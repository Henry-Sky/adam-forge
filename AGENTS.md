# Repository Guidelines

## Project Structure & Module Organization

- `checkpoint/` (ckpt) stores model weights; `data/` stores datasets. Both are usually symlinks to external storage. Track only a README in each; do not commit weights or datasets.
- `configs/` holds project loading and runtime configuration. `docs/` holds project documentation.
- `interface/` contains independently runnable task interfaces. Comment each entry point's parameters and provide an example command.
- `setup/` contains environment initialization and weight download scripts. Try a suitable domestic mirror first, then fall back to the official source. Report failures clearly.
- `tests/` contains tests. Running tests must not modify files elsewhere in the project; put generated outputs in temporary locations.
- `tools/` contains standalone utilities. Document parameters and an example invocation in comments or an adjacent README.
- `utils/` contains reusable modular code. `renders/` contains scripts called by Blender.

## Build, Test, and Development Commands

No build system or test framework is configured yet. Do not assume commands such as `npm test` or `make build` are available. When adding a toolchain, document exact setup, run, and test commands in the README. Include required configuration and input paths in interface and tool examples.

## Coding Style & Naming Conventions

Follow the conventions and formatter of each language, use descriptive names, and keep reusable logic in `utils/`. Add formatter or linter configuration when adopting a tool and run it before submitting changes.

## Testing Guidelines

Use the chosen framework's conventional test naming and document how to run the suite. Cover bug fixes and significant behavior changes. Isolate temporary outputs; tests must not alter source, configuration, data, checkpoints, or other project files.

## Commit & Pull Request Guidelines

Git history contains only `Initial commit`, so no convention is established. Use short, imperative subjects (for example, `Add parser module`). PRs should describe the change and motivation, list validation performed, link related issues when applicable, and include screenshots for visual changes.

## Security & Configuration

Do not commit credentials, private keys, machine-specific settings, model weights, or datasets. Provide secret-free example configuration and document required environment variables. Use configured external paths for data and checkpoints instead of hard-coded local paths.
