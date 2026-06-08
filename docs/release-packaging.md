# Release Packaging

JARVIS Code releases are built from the development repository into a separate
clean release folder.

Recommended local layout:

```text
C:\jarvis-code_v1.01        development repository
C:\jarvis-code-release      GitHub release repository
```

The release folder is generated. Do not develop directly inside it.

## Public Surface

Release packages should expose JARVIS Code documentation only:

- `README.md`
- `README-INSTALL.md`
- `JARVIS_REPO_CONTEXT.md`
- `NOTICE.md`
- `docs/`
- `install.ps1`
- `install.sh`
- `uninstall.ps1`
- `uninstall.sh`
- `jarvis.ps1`
- `jarvis.sh`
- `jarvis-resources/`

`JARVIS_REPO_CONTEXT.md` is generated from
`jarvis-resources/context/JARVIS_REPO_CONTEXT.md` during release builds so
installed JARVIS sessions can reason about the product layout even when
launched from the bundled `pi` engine.

The internal `pi/` folder is included as engine source, but Pi README files and
Pi user documentation are removed from the release surface to avoid product
confusion.

## Excluded Runtime Files

Never include:

- `.git/`
- `_internal/`
- `data/`
- `pi-agent/auth.json`
- `pi-agent/models.json`
- `pi-agent/settings.json`
- `pi-agent/prompts/`
- `pi-agent/sessions/`
- `pi-agent/skills/`
- `pi-agent/themes/`
- `pi/node_modules/`
- `sidecar/.venv/`
- caches and test scratch files

## Build

From the development repository:

```bash
python scripts/build-release.py --release-dir C:\jarvis-code-release --github-repo <owner>/<repo> --zip
```

If the GitHub repository is not final yet, omit `--github-repo`. The generated
README files will keep `<owner>/<repo>` placeholders until the next release
build.

Review the release folder before pushing:

```bash
cd C:\jarvis-code-release
git status
```

## Publish Flow

1. Build the release folder from the development checkout.
2. Review `RELEASE_MANIFEST.txt`.
3. Check that no runtime state was copied:

```bash
git status --short
find . -path './data' -o -path './pi-agent/auth.json' -o -path './pi-agent/skills' -o -path './pi-agent/themes' -o -path './pi/node_modules' -o -path './sidecar/.venv'
```

4. Commit and push from the release folder, not from the development checkout.

The release folder can be updated repeatedly by re-running the build command.
It is generated output, so local edits inside the release folder should be
treated as temporary unless they are copied back to the development repository.
