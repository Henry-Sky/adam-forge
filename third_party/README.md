# Third-party Dependencies

| Model checkpoint | Source submodule | Official repository |
| --- | --- | --- |
| `checkpoint/Qwen-Image-Edit-2511` | `third_party/Qwen-Image` | [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image) |
| `checkpoint/asset-harvester` | `third_party/asset-harvester` | [NVIDIA/asset-harvester](https://github.com/NVIDIA/asset-harvester) |
| `checkpoint/sam3` | `third_party/sam3` | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) |
| `checkpoint/GEM-X` | `third_party/GEM-X` | [NVlabs/GEM-X](https://github.com/NVlabs/GEM-X) |
| `checkpoint/SOMA-X` | `third_party/SOMA-X` | [NVlabs/SOMA-X](https://github.com/NVlabs/SOMA-X) |
| `checkpoint/Wan2.1-VACE-14B` | `third_party/DiffSynth-Studio` | [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) |

Fetch the source code and nested submodules after cloning this repository:

```bash
git submodule update --init --recursive
```

Model weights remain in `checkpoint/`. Follow each upstream repository's
installation instructions and license terms before use. DiffSynth-Studio is the source
code used with the local Wan2.1-VACE-14B checkpoint.

`DiffSynth-Studio` is initialized at pinned commit
`98a9e478e259a76b473f5dfdcc8442dca8d5bb41` (DiffSynth 2.2.0). Fetch only
its source with `GIT_LFS_SKIP_SMUDGE=1 git submodule update --init third_party/DiffSynth-Studio`.
Use `setup/install_wan_vace.sh` for the separate Python 3.10 environment;
the local-only configuration and explicit GPU integration test are documented in
[setup/README.md](../setup/README.md#wan21-vace-本地配置与测试).

`GEM-X` provides monocular human body pose and motion estimation using the SOMA
body model. See [its README](GEM-X/README.md) for installation and inference.
Use `setup/install_gem_x.sh` for its environment; local weights and matching
SOMA 0.3.3 assets are documented in [the checkpoint inventory](../checkpoint/README.md).

`SOMA-X` provides a unified parametric human body model with identity-dependent
skeleton fitting. See [its README](SOMA-X/README.md) for installation and usage.
Use `setup/install_soma_x.sh` for its separate environment. Model assets are
documented in [the checkpoint inventory](../checkpoint/README.md).
