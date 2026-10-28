# Repository Guidelines

## Project Structure & Module Organization

- `setup/` contains Bash installers for Qwen, Harvester, and SAM3; shared helpers belong in `setup/common.sh`.
- `third_party/` holds model source submodules. Keep upstream changes separate from repository tooling.
- `configs/`, `interface/`, and `docs/` reserve configuration, task entry points, and documentation.
- `data/` and `checkpoint/` expose external datasets and weights through local symlinks; only their READMEs are tracked. These READMEs document current sources.
- Write generated previews, inventories, and reconstruction results to ignored `outputs/`. Add reusable Python code under `utils/`, standalone utilities under `tools/`, and Blender scripts under `renders/` when needed; these directories are currently absent.

## Build, Test, and Development Commands

Refer to `setup/README.md` for module Conda environments, prerequisites, installation commands, options, and validation steps. Verify referenced scripts exist before use; the root README includes references to removed tooling.

When developing `web_server/`, keep dependency configuration and installation scripts synchronized with the code. Any addition, removal, or version change to Python or frontend dependencies must update `setup/install_web.sh` and/or `tools/install_web_vendor.py` as applicable, together with `setup/README.md` and `web_server/README.md`. Verify the affected installation steps and service imports so a freshly installed environment can run the updated service; do not rely on manually installed local packages.

## Download Sources and Proxy Traffic

Prefer trusted domestic mirrors for dependency packages, source archives, model weights, and other downloads to reduce traffic through the default proxy. Check existing local caches before downloading, and use resumable downloads when supported. Verify mirrored files against upstream checksums or pinned versions/commits where available.

Before large downloads or Git LFS checkout, inspect the required assets and select a domestic source when available. Avoid automatically fetching unnecessary LFS assets when only source code is needed. For domestic sources reachable directly, bypass the proxy for that command only; do not change global proxy settings. If domestic sources are unavailable, incomplete, or fail verification, use the official source through the default proxy and report the fallback; do not leave required downloads incomplete solely because no domestic source is available. Keep mirror URLs and download commands in the owning directory README rather than here.

## GPU Execution

Run GPU tasks outside the Codex sandbox, including GPU diagnostics, CUDA validation, CUDA extension compilation, and model inference. Request escalated execution through the tool's approval mechanism when required. The sandbox can hide GPU device nodes; do not infer a driver or device failure from sandboxed `nvidia-smi` or CUDA checks. Verify outside the sandbox before diagnosing GPU access problems.

Prefer 64 CPU threads and all 8 GPUs when the workload and program support parallel execution. Check available resources and configure the program's supported thread and multi-GPU options explicitly. For single-image or other workloads without multi-GPU support, use a suitable GPU and explain the limitation; do not launch duplicate jobs merely to occupy all devices.

## Documentation References

Keep `AGENTS.md` focused on contributor rules. Reference the corresponding directory README for setup procedures, environment versions, commands, symlink inventories, and external sources; do not copy those details here. Update the owning README when such details change.

## Data and Checkpoint Symlinks

Refer to `data/README.md` and `checkpoint/README.md` for recorded symlinks and their external sources. Use repository-relative paths and verify targets exist before use. Links and target contents are ignored by Git. Update the corresponding README when authorized link changes occur; do not replace existing links or add unrecorded links automatically.

## Default Video Format

Use the default video encoding format documented in `configs/README.md` for project video outputs unless a task explicitly requires otherwise. Resolution is unrestricted. Reference that document rather than duplicating encoding parameters here.

## Coding Style & Naming Conventions

Follow existing Bash formatting: four-space block indentation, `set -euo pipefail`, quoted paths, and descriptive names. Use `install_<module>.sh` for installers. Document entry-point parameters and example commands, including input and configuration paths. No formatter or linter is configured; document and run any introduced tools.

## Testing Guidelines

No test framework or coverage threshold is configured. Validate installer changes with syntax checks, help output, and dry runs. Actual installation checks dependencies and imports, but does not verify model inference. Add significant behavior tests under `tests/` with conventional names such as `test_<module>.py`; document the exact command and review ignore rules before tracking them. Tests must write only to temporary locations.

## Commit & Pull Request Guidelines

History uses short imperative subjects, such as `Add model submodules and Conda environment installers`. PRs should explain the problem and behavior, list validation, link related issues, and include screenshots for visual changes.

## Data & Configuration Safety

Treat data symlinks and their targets as read-only unless explicitly authorized. Never commit datasets, weights, credentials, or machine-specific configuration. Do not add undocumented model links automatically; keep link changes within the requested scope.
