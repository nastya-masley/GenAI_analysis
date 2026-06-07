#!/bin/bash
# AEMA — double-click launcher (macOS).
# Starts the app and opens it fullscreen in Chrome. Keep the window that appears OPEN —
# that window IS the running app. To stop the app: close the window (or press Ctrl+C).

# Always run from the folder this file lives in.
cd "$(dirname "$0")" || exit 1

echo "============================================"
echo "  Starting AEMA…  (keep this window OPEN)"
echo "============================================"

# First run only: install dependencies.
if [ ! -d node_modules ]; then
  echo "First-time setup — installing… this can take a few minutes."
  npm install
fi

URL="http://localhost:3000"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Wait for the server to be ready, then open it fullscreen in Chrome.
(
  for _ in $(seq 1 60); do
    if curl -s -o /dev/null "$URL/healthz"; then
      break
    fi
    sleep 0.5
  done
  if [ -x "$CHROME" ]; then
    # Dedicated fullscreen window (works even if Chrome is already open).
    "$CHROME" --user-data-dir="$HOME/.aema-chrome" --app="$URL" --start-fullscreen >/dev/null 2>&1 &
  else
    # Fallback: open in the default Chrome window.
    open -a "Google Chrome" "$URL" >/dev/null 2>&1 || open "$URL"
  fi
  echo ""
  echo "  AEMA is open in Chrome. If you don't see it, open Chrome and go to: $URL"
  echo "  Press  Shift + F  for fullscreen.  (Esc to exit fullscreen.)"
) &

# Run the server in the foreground so this window stays = the app.
npm start
