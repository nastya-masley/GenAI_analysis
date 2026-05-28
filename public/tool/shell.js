// /tool shell: owns the loader, the mode bar, and page-global keyboard
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
