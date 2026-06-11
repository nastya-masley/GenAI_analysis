#!/bin/bash
#
# AEMA — Exhibition kiosk launcher (macOS). Double-click this file in Finder.
#
# Installs pm2 the first time, starts the server SUPERVISED by pm2 (auto-restart on
# crash + ~800 MB memory cap — see ecosystem.config.js), waits for it to come up,
# then opens Google Chrome fullscreen in kiosk mode. The server keeps running as a
# pm2 daemon even after you close Chrome or this window.
#
#   Manage it later:  pm2 status  |  pm2 logs aema  |  pm2 stop aema
#

# Keep this Terminal window open with a clear message if a startup step fails.
trap 'echo; echo "✗ Startup failed — see the messages above."; read -n 1 -s -r -p "Press any key to close…"; echo; exit 1' ERR
set -e

# Always run from the folder this file lives in (the project root) so a double-click works.
cd "$(dirname "$0")"

URL="http://localhost:3000"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

echo "============================================"
echo "  Starting AEMA (exhibition / kiosk mode)…"
echo "============================================"
echo "Project: $(pwd)"

# 1. Node.js present?
command -v node >/dev/null 2>&1 || { echo "✗ Node.js 18+ is required (not found). Install it, then re-run."; false; }

# 2. Dependencies (first run only).
if [ ! -d node_modules ]; then
  echo "▶ First-time setup — installing dependencies (this can take a few minutes)…"
  npm install
fi

# 3. pm2 (install once, globally, if missing).
if ! command -v pm2 >/dev/null 2>&1; then
  echo "▶ Installing pm2 (one-time, global)…"
  npm install -g pm2 || { echo "✗ pm2 install failed. Try once manually:  sudo npm i -g pm2"; false; }
fi

# 4. (Re)start the supervised server cleanly.
echo "▶ Starting server under pm2…"
pm2 delete aema >/dev/null 2>&1 || true
pm2 start ecosystem.config.js --update-env
pm2 save >/dev/null 2>&1 || true

# 5. Wait until the server answers (up to ~30 s).
printf "▶ Waiting for %s " "$URL"
for _ in $(seq 1 60); do
  if curl -fs -o /dev/null "$URL/healthz"; then printf " ✓\n"; break; fi
  printf "."; sleep 0.5
done

# 6. Open Chrome in kiosk mode (dedicated profile so the kiosk flags always apply,
#    even if a normal Chrome window is already open). A later Chrome quit is normal,
#    so stop treating non-zero exits as a startup failure from here on.
trap - ERR
set +e
echo "▶ Opening Chrome kiosk → $URL   (Quit with Cmd+Q; the server stays up via pm2.)"
if [ -x "$CHROME" ]; then
  "$CHROME" \
    --kiosk \
    --app="$URL" \
    --user-data-dir="$HOME/.aema-kiosk-chrome" \
    --autoplay-policy=no-user-gesture-required \
    --noerrdialogs \
    --disable-session-crashed-bubble \
    --disable-infobars \
    --check-for-update-interval=31536000 \
    >/dev/null 2>&1
else
  echo "  Chrome not at the standard path — opening in the default browser instead."
  open -a "Google Chrome" "$URL" >/dev/null 2>&1 || open "$URL"
fi

echo "✓ Chrome closed. The server is still running under pm2 — stop it with:  pm2 stop aema"
