#!/usr/bin/env python3
"""
Record3D USB → WebSocket bridge
- numpy vectorised (no per-pixel Python loops)
- event-driven broadcast (zero polling latency)
- foreground-only depth filter (percentile + range cap)
- binary wire format: uint32 n | float32[n*3] xyz | uint8[n*3] rgb
"""

import asyncio
import json
import time

import numpy as np
import record3d
import websockets

PORT = 8888
STEP = 2                   # sample every Nth pixel (2 = quarter res)
MIN_DEPTH = 0.10           # metres – ignore noise closer than this
MAX_DEPTH = 5.00           # metres – clip far background

# ── foreground isolation ──────────────────────────────────────────────────────
# Strategy: keep only the nearest FOREGROUND_PERCENTILE % of valid depth pixels,
# AND never accept anything more than FOREGROUND_DEPTH_RANGE metres beyond the
# closest point in the frame. The stricter of the two limits wins.
#
# Tune FOREGROUND_PERCENTILE (0–100):
#   lower  → tighter crop (less wall, less of you)
#   higher → looser crop (more wall bleeds in)
# Tune FOREGROUND_DEPTH_RANGE (metres):
#   smaller → removes more background (try 0.6 if wall still bleeds)
#   larger  → keeps more of your body/clothes
FOREGROUND_PERCENTILE  = 45    # keep only the nearest 45 % of depth pixels
FOREGROUND_DEPTH_RANGE = 0.90  # never accept anything > 0.9 m beyond closest pt

connected_clients: set = set()
latest_frame: bytes | None = None
running = True

# asyncio loop reference – set once the server starts
_loop: asyncio.AbstractEventLoop | None = None


def log(msg: str) -> None:
    print(msg, flush=True)


# ── broadcast helper (called from event loop thread) ─────────────────────────
def _broadcast_now() -> None:
    if latest_frame and connected_clients:
        websockets.broadcast(connected_clients, latest_frame)


# ── Record3D handler ──────────────────────────────────────────────────────────
class Record3DApp:
    def __init__(self) -> None:
        self.session: record3d.Record3DStream | None = None
        self.frame_no = 0

    # runs on the C++ Record3D thread at up to 60 fps
    def on_new_frame(self) -> None:
        global latest_frame

        try:
            depth = self.session.get_depth_frame()
            rgb   = self.session.get_rgb_frame()
            intr  = self.session.get_intrinsic_mat()
            if depth is None:
                return

            h, w = depth.shape
            fx, fy = float(intr.fx), float(intr.fy)
            cx, cy = float(intr.tx), float(intr.ty)

            # ── sample grid ──────────────────────────────────────────────────
            u_idx = np.arange(0, w, STEP, dtype=np.int32)
            v_idx = np.arange(0, h, STEP, dtype=np.int32)
            uu, vv = np.meshgrid(u_idx, v_idx)          # shape (H//S, W//S)
            dd = depth[vv, uu].astype(np.float32)        # depth values

            # ── foreground mask (percentile + range cap) ─────────────────────
            valid = (dd > MIN_DEPTH) & (dd < MAX_DEPTH) & np.isfinite(dd)
            if not np.any(valid):
                return

            depths_flat = dd[valid].ravel()
            d_min = float(depths_flat.min())

            # Percentile cutoff: keep only the nearest N % of pixels
            thresh_pct = float(np.percentile(depths_flat, FOREGROUND_PERCENTILE))
            # Hard range cap: never more than FOREGROUND_DEPTH_RANGE beyond closest
            thresh_range = d_min + FOREGROUND_DEPTH_RANGE
            # Use whichever is stricter (smaller depth = closer to camera)
            thresh = min(thresh_pct, thresh_range)

            fg = valid & (dd <= thresh)

            n = int(np.sum(fg))
            if n == 0:
                return

            u_fg = uu[fg].ravel().astype(np.float32)
            v_fg = vv[fg].ravel().astype(np.float32)
            z_fg = dd[fg].ravel()

            # ── 3-D projection ────────────────────────────────────────────────
            x_fg = (u_fg - cx) * z_fg / fx
            y_fg = (v_fg - cy) * z_fg / fy

            xyz = np.column_stack([x_fg, y_fg, z_fg]).astype(np.float32)   # (n,3)

            # ── colour from RGB frame ─────────────────────────────────────────
            if rgb is not None:
                rh, rw = rgb.shape[:2]
                su = np.clip((u_fg * (rw / w)).astype(np.int32), 0, rw - 1)
                sv = np.clip((v_fg * (rh / h)).astype(np.int32), 0, rh - 1)
                colors = rgb[sv, su, :3].astype(np.uint8)
            else:
                colors = np.full((n, 3), [80, 180, 255], dtype=np.uint8)

            # ── binary pack: [uint32 n][float32 xyz * n][uint8 rgb * n] ──────
            header = np.array([n], dtype=np.uint32).tobytes()
            latest_frame = header + xyz.tobytes() + colors.tobytes()

            self.frame_no += 1
            if self.frame_no % 60 == 0:
                kb = len(latest_frame) / 1024
                log(f"frame {self.frame_no:5d} | {n:5d} pts | {kb:.0f} KB | clients {len(connected_clients)}")

            # ── wake the event loop immediately ──────────────────────────────
            if _loop is not None and connected_clients:
                _loop.call_soon_threadsafe(_broadcast_now)

        except Exception as e:
            log(f"frame error: {e}")

    def on_stream_stopped(self) -> None:
        log("stream stopped")

    def start(self) -> bool:
        log("searching for Record3D devices…")
        devs = record3d.Record3DStream.get_connected_devices()
        if not devs:
            log("no devices found  –  connect iPhone via USB and open Record3D")
            return False
        log(f"found {len(devs)} device(s)")
        self.session = record3d.Record3DStream()
        self.session.on_new_frame      = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(devs[0])
        log(f"connected to device {devs[0].product_id}")
        log("→ tap the stream button in Record3D on your iPhone")
        return True


# ── WebSocket server ──────────────────────────────────────────────────────────
async def handle_client(ws) -> None:
    log(f"client connected: {ws.remote_address}")
    connected_clients.add(ws)
    try:
        await ws.send(json.dumps({"type": "connected"}))
        async for _ in ws:
            pass
    except websockets.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(ws)
        log(f"client disconnected: {ws.remote_address}")


async def run_server() -> None:
    global _loop
    _loop = asyncio.get_running_loop()
    log(f"WebSocket server on ws://localhost:{PORT}")
    log("open browser → 'Connect Record3D' → Connect\n")
    async with websockets.serve(handle_client, "0.0.0.0", PORT):
        await asyncio.Future()   # run forever


def main() -> None:
    log("=" * 52)
    log("Record3D USB → WebSocket  (numpy · event-driven)")
    log("=" * 52 + "\n")

    app = Record3DApp()
    if not app.start():
        log("retrying in 3 s…")
        time.sleep(3)
        if not app.start():
            log("could not connect – exiting")
            return

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        log("\nstopped")


if __name__ == "__main__":
    main()
