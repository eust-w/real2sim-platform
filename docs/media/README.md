# Launch media sources

The release videos use only this repository's original procedural sample, actual offline UI recordings, the native Genesis camera recording and original explanatory text. No private task history, provider-generated mesh, cloned voice or third-party music is included.

`storyboard.json` is the bilingual narration and shot list. `scripts/capture-media.mjs` requires a running isolated `npm run demo` server. `scripts/render-launch.py` uses FFmpeg and macOS's built-in Tingting synthetic voice. It is a local production helper, not part of the cross-platform application runtime. Voice synthesis is explicitly disclosed in the video.

```bash
node scripts/capture-media.mjs
# Put the verified simulation.mp4 and metrics.json under artifacts/media/.
uv run --with pillow==13.0.0 python scripts/render-launch.py
uv run --with pillow==13.0.0 python scripts/render-launch.py --language zh --vertical
```

The media tool runs with a separate, current Pillow environment. Do not confuse that environment with the separately audited Genesis dependency chain.

Outputs in `artifacts/release/` are intentionally excluded from Git. Publish only inspected final files as Release assets. Keep platform IDs, public URLs and publication states in a separate ledger; an upload or draft is not public playback.
