# Security

Real2Sim Studio v0.1 is a single-user local workbench. The API binds to `127.0.0.1`, rejects non-local Host/Origin values, and does not enable permissive CORS. It is not a multi-user hosted service. Do not expose the API or Vite dev server through a public tunnel or reverse proxy.

Keep provider credentials in `.env.local` or the process environment. Never put them in a browser build, URL, issue, recording or commit. `npm run demo` disables provider access and separates demo tasks from normal user data.

Only run exported scripts and load model files from trusted sources. Material presets are not safety-certified or calibrated for real hardware.

## Known optional runtime risk (2026-10-28)

The separate Python simulation environment is **not vulnerability-clean**. `pip-audit` reported 72 advisory entries for Pillow 11.3.0 (some identifiers overlap). Genesis 1.4.5 depends on MoviePy >=2, whose current 2.2.1 release requires Pillow <12; the audited fixes extend through Pillow 13.0.0. Installing Pillow 13 alone makes that environment dependency-inconsistent. We do not hide or suppress those audit results.

This Alpha demonstrates only the bundled, trusted procedural mesh. Do not use this Python environment to process untrusted images, fonts or models, and do not expose it as an upload service. Use an isolated environment without credentials or private files. The recording validates the sample workflow, not immunity from dependency vulnerabilities. Track upstream MoviePy compatibility before using the optional runtime on external assets.

The Node dependency audit was clean at release preparation. These are point-in-time results, not a security guarantee.

Report vulnerabilities privately through [GitHub private vulnerability reporting](https://github.com/eust-w/real2sim-platform/security/advisories/new). Do not post credentials or sensitive reproductions in public issues.
