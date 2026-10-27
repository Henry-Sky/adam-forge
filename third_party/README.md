# Third-party Dependencies

| Model checkpoint | Source submodule | Official repository |
| --- | --- | --- |
| `checkpoint/Qwen-Image-Edit-2511` | `third_party/Qwen-Image` | [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image) |
| `checkpoint/asset-harvester` | `third_party/asset-harvester` | [NVIDIA/asset-harvester](https://github.com/NVIDIA/asset-harvester) |
| `checkpoint/sam3` | `third_party/sam3` | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) |

Fetch the source code and nested submodules after cloning this repository:

```bash
git submodule update --init --recursive
```

Model weights remain in `checkpoint/`. Follow each upstream repository's
installation instructions and license terms before use.
