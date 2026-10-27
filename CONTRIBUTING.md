# Contributing

Use Node.js 22.12+ and `npm ci`. Start with `npm run demo`; it uses an isolated local data directory and cannot call the paid generation API.

Before submitting a pull request:

```bash
npm run test:run
npm run lint
npm run build
npm run test:e2e
npm audit
```

The browser smoke test needs Chromium (`npx playwright install chromium`). It starts and stops its own isolated servers. For changes to physical parameter mapping or the Python export, also run the example through Genesis and include `metrics.json`, the environment, and reproduction commands.

Keep changes focused. Include a regression test for changed behavior and explain visible limitations. Do not commit `.env.local`, `.data`, virtual environments, credentials, private generated assets, or account sessions.

Useful first contributions include installation reports, a reproducible cloth example, unit-aware material presets, accessible keyboard controls and additional generation-provider adapters. Open an issue before adding a new simulator backend.

Existing optional Git hooks can be enabled with `git config core.hooksPath .githooks`. Their commit-message convention requires an emoji prefix. Contributions are licensed under MIT.
