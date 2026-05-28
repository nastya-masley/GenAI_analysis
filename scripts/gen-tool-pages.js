#!/usr/bin/env node
/*
 * Generator for the standalone /tool page.
 *
 * Produces public/tool/{index.html, base.css, shell.js, processing.css,
 * archive.css, live.css, processing.js, archive.js, live.js}.
 *
 * Each of the 3 workspace modes (Processing / Archive / Live) becomes a
 * fully independent section: its own prefixed ids + classes, its own CSS
 * file, and its own self-contained JS module (a tailored copy of script.js).
 * No ids/classes/styles are shared between modes. The only shared chrome is
 * the loader + mode bar, which live in shell.js / base.css with DISTINCT
 * names (tool-*) so they never collide with any mode.
 *
 * This is a one-time code generator — the OUTPUT files share no code with
 * each other. Re-run with `node scripts/gen-tool-pages.js` after editing the
 * originals in public/.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const PUB = path.join(ROOT, 'public');
const OUT = path.join(PUB, 'tool');

const htmlSrc = fs.readFileSync(path.join(PUB, 'index.html'), 'utf8');
const cssSrc = fs.readFileSync(path.join(PUB, 'styles.css'), 'utf8');
const jsSrc = fs.readFileSync(path.join(PUB, 'script.js'), 'utf8');

// ── Token sets ───────────────────────────────────────────────────────────
// IDs come from the original index.html (+ one id created dynamically in JS).
// SVG-internal ids (pointer, axes-circle, emotion-*, Circumplex_diagram) are
// NOT in index.html, so they never enter the set and are never prefixed.
const idSet = new Set();
for (const m of htmlSrc.matchAll(/\bid="([\w-]+)"/g)) idSet.add(m[1]);
idSet.add('open-archive-link'); // injected via liveStatus.innerHTML

// Classes come from every selector defined in styles.css, plus the dynamic
// classes used in JS and any class attributes in HTML.
const classSet = new Set();
for (const m of cssSrc.matchAll(/\.(-?[A-Za-z_][\w-]*)/g)) classSet.add(m[1]);
for (const m of jsSrc.matchAll(/classList\.(?:add|remove|toggle|contains)\(\s*['"]([\w-]+)['"]/g)) classSet.add(m[1]);
for (const m of jsSrc.matchAll(/\.className\s*=\s*['"]([^'"]*)['"]/g)) m[1].split(/\s+/).forEach((t) => t && classSet.add(t));
for (const m of htmlSrc.matchAll(/\bclass="([^"]*)"/g)) m[1].split(/\s+/).forEach((t) => t && classSet.add(t));

// ── Prefixers ──────────────────────────────────────────────────────────────
const pfxId = (pfx, t) => (idSet.has(t) ? `${pfx}-${t}` : t);
const pfxClass = (pfx, t) => (classSet.has(t) ? `${pfx}-${t}` : t);

// Prefix .class / #id tokens inside a CSS-selector-ish string.
function pfxSelector(pfx, s) {
  s = s.replace(/#([A-Za-z][\w-]*)/g, (m, t) => (idSet.has(t) ? `#${pfx}-${t}` : m));
  s = s.replace(/\.(-?[A-Za-z_][\w-]*)/g, (m, t) => (classSet.has(t) ? `.${pfx}-${t}` : m));
  return s;
}

// Whole CSS file (selectors only; values/keywords like `hidden` are untouched
// because they are not preceded by . or #).
function prefixCss(pfx, css) {
  return pfxSelector(pfx, css);
}

// id=/for=/aria-controls=/aria-labelledby= and class= attribute values, in
// both real HTML and HTML embedded in JS template literals.
function prefixAttrs(pfx, text) {
  text = text.replace(/\b(id|for|aria-controls|aria-labelledby)="([\w-]+)"/g,
    (m, a, v) => `${a}="${pfxId(pfx, v)}"`);
  text = text.replace(/\bclass="([^"]*)"/g,
    (m, v) => `class="${v.split(/\s+/).map((t) => pfxClass(pfx, t)).join(' ')}"`);
  return text;
}

// Full JS-aware prefixing.
function prefixJs(pfx, js) {
  // 1. HTML attributes inside template literals / innerHTML.
  js = prefixAttrs(pfx, js);
  // 2. document.getElementById('id')
  js = js.replace(/(getElementById\(\s*['"])([\w-]+)(['"]\s*\))/g,
    (m, a, id, c) => `${a}${pfxId(pfx, id)}${c}`);
  // 3. classList.add/remove/toggle/contains('class', ...)
  js = js.replace(/(classList\.(?:add|remove|toggle|contains)\(\s*['"])([\w-]+)(['"])/g,
    (m, a, cls, c) => `${a}${pfxClass(pfx, cls)}${c}`);
  // 4. el.className = 'a b c'
  js = js.replace(/(\.className\s*=\s*['"])([^'"]*)(['"])/g,
    (m, a, v, c) => `${a}${v.split(/\s+/).map((t) => pfxClass(pfx, t)).join(' ')}${c}`);
  // 5. closest('.class' | '#id')
  js = js.replace(/(closest\(\s*['"])([.#][\w-]+)(['"]\s*\))/g,
    (m, a, sel, c) => `${a}${pfxSelector(pfx, sel)}${c}`);
  // 6. querySelector / querySelectorAll(<selector>) — selector may contain
  //    inner quotes (e.g. '[id^="emotion-"]'); pfxSelector only touches .x/#x.
  js = js.replace(/querySelector(All)?\(([^)]*)\)/g,
    (m, all, arg) => `querySelector${all || ''}(${pfxSelector(pfx, arg)})`);
  return js;
}

// ── HTML fragment extraction ────────────────────────────────────────────────
function slice(src, startMarker, endMarker, includeEnd) {
  const s = src.indexOf(startMarker);
  if (s < 0) throw new Error(`marker not found: ${startMarker}`);
  const e = src.indexOf(endMarker, s + startMarker.length);
  if (e < 0) throw new Error(`marker not found: ${endMarker}`);
  return src.slice(s, includeEnd ? e + endMarker.length : e);
}

const workspaceBody = slice(htmlSrc, '<div class="workspace-body">', '</div><!-- .workspace-body -->', true);
const fullscreen = slice(htmlSrc, '<div id="fullscreen-overlay"', '</main>', false).trim();
const popups = slice(htmlSrc, '<!-- Time range selection popup -->', '<script type="module"', false).trim();

const loaderMarkup = slice(htmlSrc, '<div id="loader-overlay"', '<div id="workspace-root"', false).trim();

function buildSection(pfx) {
  const fragment = `<div class="workspace">\n${workspaceBody}\n</div>\n${fullscreen}\n${popups}`;
  return `    <section id="${pfx}-section" class="tool-section" hidden>\n${prefixAttrs(pfx, fragment)}\n    </section>`;
}

// ── Per-mode JS module ───────────────────────────────────────────────────────
const MODE_OF = { proc: 'edit', arch: 'archive', live: 'live' };

function buildModule(pfx) {
  let js = prefixJs(pfx, jsSrc);

  // Hand the global keyboard / fullscreen handlers to the shell (avoids 3
  // modules each registering document-level keydown handlers, which would
  // make the font-size shortcut compound). Cut from the marker to EOF.
  const cut = js.indexOf('const fsPlayerCard = document.querySelector(');
  if (cut < 0) throw new Error('fsPlayerCard marker not found in script.js');
  js = js.slice(0, cut).trimEnd() + '\n';

  // The shell owns the loader; neutralise the module's auto-boot.
  js = js.replace(/\n[^\n]*loaderOverlay\?\.addEventListener\('click', endLoader\);/, '');
  js = js.replace(/\n\s*setTimeout\(endLoader, 2000\);/, '');

  // Cross-mode navigation: the "Open in Archive" link must move the SHELL,
  // not the live section. (live module only.)
  if (pfx === 'live') {
    js = js.replace(/switchMode\('archive'\)/g, "(window.__toolGoMode && window.__toolGoMode('archive'))");
  }

  const mode = MODE_OF[pfx];
  js += `
// ── /tool bootstrap (${pfx}) — activate when this section is visible ──
(function () {
  const sec = document.getElementById('${pfx}-section');
  if (!sec) return;
  const ws = document.querySelector('.${pfx}-workspace');
  let activated = false;
  function activate() {
    if (showAnalyticsBtn) { showAnalyticsBtn.style.display = 'inline-flex'; showAnalyticsBtn.textContent = 'View Analytics'; }
    if (outputsPanel) outputsPanel.hidden = true;
    if (submitBtn) submitBtn.style.display = 'none';
    if (ws) ws.classList.remove('${pfx}-analytics-visible');
    appState = 'workspace';
    workspaceMode = null;
    switchMode('${mode}');
    activated = true;
  }
  function deactivate() {
    if (typeof stopWebcam === 'function') stopWebcam();
  }
  function sync() { if (!sec.hidden) activate(); else if (activated) deactivate(); }
  new MutationObserver(sync).observe(sec, { attributes: true, attributeFilter: ['hidden'] });
  sync();
})();
`;
  return js;
}

// ── base.css (shared chrome only) ────────────────────────────────────────────
const baseHead = cssSrc.slice(0, cssSrc.indexOf('.container {'));
const baseLoader = cssSrc.slice(cssSrc.indexOf('.loader-overlay {'), cssSrc.indexOf('/* ============ PANEL TABS ============ */'));
const baseChrome = `
/* ── /tool page frame ── */
.tool-stage {
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
  position: relative;
  overflow: hidden;
}
.tool-section {
  flex: 1 1 auto;
  min-height: 0;
  width: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.tool-section[hidden] { display: none; }

/* ── /tool mode bar (distinct from any mode's classes) ── */
.tool-bar {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.25rem;
  padding: 0.45rem 1rem;
  background: #000;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
.tool-bar-btn {
  padding: 0.3rem 1.2rem;
  border: 1px solid rgba(255, 255, 255, 0.25);
  border-radius: 0;
  background: transparent;
  color: rgba(255, 255, 255, 0.6);
  font-size: 0.8rem;
  font-weight: 150;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}
.tool-bar-btn:hover { background: rgba(255, 255, 255, 0.08); color: #fff; }
.tool-bar-btn--on { background: #fff; color: #000; border-color: #fff; }
`;
const baseCss = baseHead + baseLoader + baseChrome;

// ── shell.js (loader + mode bar + global keyboard/fullscreen) ────────────────
const shellJs = `// /tool shell: owns the loader, the mode bar, and page-global keyboard
// shortcuts. It dynamic-imports each mode module the first time that mode is
// shown, and toggles section [hidden] on switch. It knows nothing about a
// mode's internals — each module self-activates via a MutationObserver on its
// own section's [hidden] attribute.

const loader = document.getElementById('loader-overlay');
const bar = document.getElementById('tool-bar');

const SECTION = { edit: 'proc-section', archive: 'arch-section', live: 'live-section' };
const MODULE = { edit: './processing.js', archive: './archive.js', live: './live.js' };
const loaded = {};
let current = null;

function show(mode) {
  if (!SECTION[mode] || mode === current) return;
  current = mode;
  for (const [m, id] of Object.entries(SECTION)) {
    const el = document.getElementById(id);
    if (el) el.hidden = m !== mode;
  }
  bar?.querySelectorAll('.tool-bar-btn').forEach((b) => {
    b.classList.toggle('tool-bar-btn--on', b.dataset.mode === mode);
  });
  if (!loaded[mode]) {
    loaded[mode] = true;
    import(MODULE[mode]).catch((e) => console.error('tool module failed to load:', mode, e));
  }
}
window.__toolGoMode = show;

bar?.addEventListener('click', (e) => {
  const b = e.target.closest('.tool-bar-btn');
  if (b) show(b.dataset.mode);
});

// Loader: spinning AEMA logo. Dismiss on click or after 2s, then boot Live.
let booted = false;
function boot() {
  if (booted) return;
  booted = true;
  if (loader) loader.hidden = true;
  show('live');
}
loader?.addEventListener('click', boot);
setTimeout(boot, 2000);

// ── Page-global keyboard shortcuts (owned here, once) ──
const isTypingTarget = (t) =>
  !!t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);

const FONT_SIZE_KEY = 'fontSizePx';
const FONT_SIZE_BASE = 16;
const FONT_SIZE_MIN = 10;
const FONT_SIZE_MAX = 32;
const applyFontSize = (px) => { document.documentElement.style.fontSize = px + 'px'; };
try {
  const stored = parseInt(localStorage.getItem(FONT_SIZE_KEY), 10);
  if (Number.isFinite(stored) && stored >= FONT_SIZE_MIN && stored <= FONT_SIZE_MAX) applyFontSize(stored);
} catch {}

document.addEventListener('keydown', (e) => {
  if (!(e.metaKey || e.ctrlKey)) return;
  if (isTypingTarget(e.target)) return;
  const k = e.key;
  if (k !== '+' && k !== '=' && k !== '-' && k !== '0') return;
  e.preventDefault();
  const cur = parseInt(getComputedStyle(document.documentElement).fontSize, 10) || FONT_SIZE_BASE;
  let next = cur;
  if (k === '+' || k === '=') next = Math.min(FONT_SIZE_MAX, cur + 1);
  else if (k === '-') next = Math.max(FONT_SIZE_MIN, cur - 1);
  else if (k === '0') next = FONT_SIZE_BASE;
  applyFontSize(next);
  try { localStorage.setItem(FONT_SIZE_KEY, String(next)); } catch {}
});

const PAGE_FS_KEY = 'pageFullscreen';
document.addEventListener('keydown', (e) => {
  if (e.key !== 'f' && e.key !== 'F') return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (isTypingTarget(e.target)) return;
  if (e.shiftKey) {
    e.preventDefault();
    if (document.fullscreenElement) document.exitFullscreen?.();
    else document.documentElement.requestFullscreen?.();
    return;
  }
  const sec = current && document.getElementById(SECTION[current]);
  const card = sec && sec.querySelector('[class*="player-card"]');
  if (!card) return;
  e.preventDefault();
  if (document.fullscreenElement) document.exitFullscreen?.();
  else card.requestFullscreen?.();
});

document.addEventListener('fullscreenchange', () => {
  try {
    if (document.fullscreenElement === document.documentElement) localStorage.setItem(PAGE_FS_KEY, '1');
    else localStorage.removeItem(PAGE_FS_KEY);
  } catch {}
});
try {
  if (localStorage.getItem(PAGE_FS_KEY) === '1') {
    const restore = () => {
      document.removeEventListener('keydown', restore, true);
      document.removeEventListener('pointerdown', restore, true);
      if (!document.fullscreenElement) document.documentElement.requestFullscreen?.().catch(() => {});
    };
    document.addEventListener('keydown', restore, true);
    document.addEventListener('pointerdown', restore, true);
  }
} catch {}
`;

// ── index.html ───────────────────────────────────────────────────────────────
const indexHtml = `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AEMA — Tool</title>
    <link rel="icon" type="image/svg+xml" href="/assets/AEMA_logo.svg" />
    <link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin />
    <link rel="dns-prefetch" href="//cdn.jsdelivr.net" />
    <link rel="dns-prefetch" href="//storage.googleapis.com" />
    <link rel="dns-prefetch" href="//generativelanguage.googleapis.com" />
    <link rel="preload" as="image" href="/assets/AEMA_logo.svg" type="image/svg+xml" />
    <link rel="preload" as="image" href="/assets/checkbox.png" />
    <link rel="preload" as="image" href="/assets/checkbox_crossed.png" />
    <link rel="stylesheet" href="base.css" />
    <link rel="stylesheet" href="processing.css" />
    <link rel="stylesheet" href="archive.css" />
    <link rel="stylesheet" href="live.css" />
  </head>
  <body>
    ${loaderMarkup}

    <div id="tool-bar" class="tool-bar">
      <button type="button" class="tool-bar-btn" data-mode="edit">Processing</button>
      <button type="button" class="tool-bar-btn" data-mode="archive">Archive</button>
      <button type="button" class="tool-bar-btn tool-bar-btn--on" data-mode="live">Live</button>
    </div>

    <div class="tool-stage">
${buildSection('proc')}
${buildSection('arch')}
${buildSection('live')}
    </div>

    <script type="module" src="shell.js"></script>
  </body>
</html>
`;

// ── Write everything ─────────────────────────────────────────────────────────
fs.mkdirSync(OUT, { recursive: true });
const files = {
  'index.html': indexHtml,
  'base.css': baseCss,
  'shell.js': shellJs,
  'processing.css': prefixCss('proc', cssSrc),
  'archive.css': prefixCss('arch', cssSrc),
  'live.css': prefixCss('live', cssSrc),
  'processing.js': buildModule('proc'),
  'archive.js': buildModule('arch'),
  'live.js': buildModule('live'),
};
for (const [name, content] of Object.entries(files)) {
  fs.writeFileSync(path.join(OUT, name), content);
  console.log('wrote', path.relative(ROOT, path.join(OUT, name)), `(${content.length} bytes)`);
}
console.log(`\nidSet: ${idSet.size} ids, classSet: ${classSet.size} classes`);
