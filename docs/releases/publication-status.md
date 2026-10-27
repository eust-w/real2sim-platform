# Alpha Publication Status

Verified on 2026-10-28 (Asia/Shanghai). Uploads, drafts and review queues are not counted as publication.

| Channel | State | Evidence |
| --- | --- | --- |
| GitHub | Public MIT repository and prerelease | [Repository](https://github.com/eust-w/real2sim-platform), [v0.1.0-alpha.1](https://github.com/eust-w/real2sim-platform/releases/tag/v0.1.0-alpha.1) |
| Gitea | Public mirror and prerelease | [Repository](https://gitea.d-robotics.cc/longtao.wu/real2sim-platform), [v0.1.0-alpha.1](https://gitea.d-robotics.cc/longtao.wu/real2sim-platform/releases/tag/v0.1.0-alpha.1) |
| X | Published; 90-second video playback verified | [Public post](https://x.com/eustancewu/status/2115131684236800501) |
| Douyin | Published; 45-second video playback verified | [Public video](https://www.douyin.com/video/7701399089530916159) |
| Bilibili | Not published; awaiting uploader agreement confirmation | Chinese introduction and upload copy prepared |
| YouTube | Not published; awaiting account verification and agreement confirmation | English-captioned introduction and upload copy prepared |
| Xiaohongshu | Not published; draft saved, account action required | Vertical video and upload copy prepared |
| WeChat Channels | Not published; publishing interface unavailable to the current tool session | Vertical video and upload copy prepared |

Both releases contain 15 assets. GitHub asset SHA-256 metadata matches the local files; all 15 Gitea assets were downloaded without authentication and their SHA-256 hashes matched. Neither private task history nor local credentials are release assets.

## Media

- [90-second Chinese introduction](https://github.com/eust-w/real2sim-platform/releases/download/v0.1.0-alpha.1/real2sim-intro-zh.mp4)
- [90-second English-captioned introduction](https://github.com/eust-w/real2sim-platform/releases/download/v0.1.0-alpha.1/real2sim-intro-en.mp4), with Chinese synthetic narration
- [45-second Chinese vertical version](https://github.com/eust-w/real2sim-platform/releases/download/v0.1.0-alpha.1/real2sim-short-zh.mp4)
- [Native Genesis solver recording](https://github.com/eust-w/real2sim-platform/releases/download/v0.1.0-alpha.1/genesis-native-simulation.mp4)
- Subtitles, covers, the runnable example, metrics, license and security notes are attached to both releases.

The introduction uses actual UI captures and the original procedural sample. Slowed simulation replay is labeled. Synthetic narration and AI assistance are disclosed. The English-captioned version does not have an English voice track.

## Verification Boundary

The original release tag is `56f124f`; it has not been moved. Follow-up main-branch fixes pin the supported CI runner, add numerical-metric CI tests, and ensure the API and Vite receive the same `.env.local` port. [CI at 518917d passed](https://github.com/eust-w/real2sim-platform/actions/runs/40551404979), including browser smoke tests and numerical-metric tests. Current launcher regression checks also pass locally.

The recorded Genesis example completed 800 CPU steps with finite output and measured shape change. It is not a physical-accuracy benchmark, GPU validation or proof of arbitrary mesh compatibility. See [release notes](v0.1.0-alpha.1.md) and [numerical results](../verification/genesis-cpu.json).

The optional Python stack still has unresolved Pillow advisories due to an upstream MoviePy version constraint. Use only trusted sample assets in an isolated environment. A clean Node audit does not make the Python runtime safe for untrusted input; see [SECURITY.md](../../SECURITY.md).

Platform-specific copy is kept in [the media guide](../media/platform-copy.md). Publication records do not promise reach, stars or virality.
