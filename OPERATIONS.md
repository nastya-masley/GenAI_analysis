



## 3. Go fullscreen

- Press **Shift + F** (this hides the browser bars).
- Clicking the intro logo when it appears also goes fullscreen.
- To **exit** fullscreen: press **Esc** or **Shift + F** again.

---

## 4. Restart if it freezes or fails

1. Click the **black window**, press **Ctrl + C** to stop.
2. **Double-click `start.command`** again.

Other quick fixes:
- **Only the picture is stuck?** Click the app and press **Cmd + R** to reload.
- **Still broken?** Restart the Mac, then double-click `start.command`.

---

## 5. Get the latest fix from GitHub

1. Stop the app: click the black window, press **Ctrl + C**.
2. In **Terminal**, run:
   ```
   cd /Users/yuriismolii/Developer/GenAI_analysis
   git pull
   npm install
   npm start
   ```

> **If `git pull` shows an error**, force-load the developer's version (this throws away any local changes):
> ```
> cd /Users/yuriismolii/Developer/GenAI_analysis
> git fetch origin
> git reset --hard origin/feature/ver4
> npm install
> npm start
> ```

---

## 6. Update the API key

Do this when the AI shows an error like **rate-limited** or **invalid key**.

1. Press **Ctrl + Alt + K**  (or open **http://localhost:3000/#set-api-key**).
2. A small box appears. Type the **passphrase**, then the **new key**.
3. Click **Save key**.

The new key works **right away** and stays after a restart. ✅

> The passphrase is set up once by the developer (the `ADMIN_TOKEN` in the `.env` file).
> If the box says "Key replacement is disabled", the developer needs to set that up first.
>
> **Manual way** (developer): edit `GEMINI_API_KEY` in the `.env` file, then restart the app.

---

## 7. Where files are saved

Everything is inside the app folder, under `assets/`:

| What | Folder |
|------|--------|
| Recorded live clips (and what **ARCHIVE** shows) | `assets/archive/library/` |
| Captured frames (the **Capture frame** button) | `assets/export/frames/` |
| Background images you can pick — **drop new ones here** | `assets/archive/background_images/` |
| Clips/photos for **Select media** — **drop new ones here** | `assets/archive/media/` |
| The intro / loading video | `assets/loading/Intro_logo.mp4` |

---

## 8. Quick fixes

- **AI error / "rate limited"** → update the API key (see step 6).
- **No camera / black webcam** → in Chrome allow camera access, then press **Cmd + R**.
- **Nothing loads in the browser** → the black window was closed. Double-click `start.command` again.
