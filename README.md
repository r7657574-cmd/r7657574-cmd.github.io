# Ruilin Wang — research portfolio

Source for a small, static GitHub Pages portfolio focused on three research threads:

- **FUSE:** concepts only; no manuscript or implementation is released.
- **Theory2Method:** public abstract plus a clean reconstruction of the theory-to-method core; no paper PDF.
- **Graph connectivity restoration:** manuscript plus a compact, safety-hardened reconstruction of the graph-repair core.

The site has no analytics, remote JavaScript, build system, or external font dependency.

## Public site

The published homepage is available at <https://r7657574-cmd.github.io/>.

## Optional local preview

After cloning the repository, run the following command from the repository root:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`. This address works only while the local server is running; it is separate from the public GitHub Pages URL.

## Publication boundaries

This repository intentionally excludes unpublished manuscripts, private prompts and validators, credentials, raw run traces, datasets, full experiment harnesses, and model-provider configuration. The code directories are research snapshots rather than complete reproduction packages.

No software license is declared in this snapshot.
