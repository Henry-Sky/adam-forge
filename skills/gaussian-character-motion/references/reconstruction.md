# Scene selection, reference preparation, and binding

Read this file for workflow stages 1–7. Paths below are relative to the repository root. Resolve environment names, sources and asset versions from the owning READMEs before execution; ignored historical examples may be absent on another machine.

## 1. Inventory and select a person

Build an inventory from the actual source dataset. Record scene ID, camera, person ID, frame index base, timestamp units, image path, image dimensions, bounding-box convention, occlusion and truncation flags. Verify frame files exist and timestamps are monotonic. Do not assume a field named `bbox` uses corner coordinates.

Prefer full-body coverage, enough original pixels to distinguish limbs, visible feet and head, and separation from nearby people. Review contact sheets rather than selecting from annotations alone. A large box can contain an occluding vehicle rather than a clear person. Check consistent identity across cameras before using multiple views.

Choose the reconstruction reference for appearance and complete anatomy; choose the animation interval for continuity and articulation. They need not use the same frame. Record both choices. When the user requests the clearest N frames for animation, prefer a continuous window: minimize occluded/truncated/missing frames, then compare original pixel area and visual sharpness. Do not join unrelated best-looking frames into a discontinuous motion.

For center-size boxes `(cx,cy,w,h)`, crop with floor on `(cx-w/2,cy-h/2)` and ceil on `(cx+w/2,cy+h/2)`, clipped to image boundaries. Add padding if necessary to retain hands/feet. Save original resolution crops and exact integer crop coordinates; upscaling does not add measured detail. Keep a map from every derived frame back to source path/index/timestamp.

## 2. Segment and generate the A-pose

SAM3 can isolate the target before Qwen when the background is crowded. Select the target mask using its box/position and visual review, not only the highest score when multiple people are present. Retain the binary mask, transparent cutout and a composited preview suitable for Qwen. Inspect fingers, footwear, hard-hat edges, straps and held objects.

Use Qwen Image Edit with the local checkpoint and existing environment. Prompt for a neutral A-pose, straight torso, relaxed symmetric limbs and complete feet; explicitly preserve face, hairstyle/headwear, clothing construction, logos/colors, accessories, camera viewpoint and lighting as appropriate to the task. An example starting prompt is:

> Change this person's posture to a neutral A-pose, standing upright with arms about 30 degrees below horizontal and feet shoulder-width apart. Preserve the person's identity, clothing, colors and headwear. Keep the full body inside the image with clear separation between the arms and torso and between the legs.

Adapt object handling to the actual character; do not silently remove a distinctive object to make segmentation easier. Review the result for duplicated limbs, fused hands, asymmetry, cropped shoes, changed identity and invented details. Save failed attempts separately when retrying; explain any appearance compromise.

The person32 example used `QwenImageEditPlusPipeline`, bf16, seed 42, 40 steps, `true_cfg_scale=4`, `guidance_scale=1`, and the 2511 checkpoint with `zero_cond_t` support. These are a reproducible starting point, not universal quality settings. Record generation dimensions separately from any final image resize. Use the Qwen pipeline for this stage as requested; do not replace it with another image generation service.

## 3. Prepare and run Harvester

Segment the generated A-pose again: its silhouette differs from the source crop. Keep the RGB reference and mask in the same pixel coordinates. Resize with preserved aspect ratio and pad both with identical offsets; use nearest-neighbor interpolation for the mask. The prior example fit the image's longest side to 460 pixels in a white 512×512 canvas, leaving margin for limbs. Record dimensions, offsets and mask score. Inspect the final model input, not only the high-resolution Qwen image.

Run Harvester in its own environment with local diffusion, camera-estimator and lifting weights. Check the current upstream CLI before reusing flags. The prior run used 30 diffusion steps, bf16, seed 42 and one image. Preserve input image/mask, invocation, checkpoint identifiers, logs, generated multi-views, PLY and rotation preview. Use single-GPU execution when that entry point does not support multiple devices, following repository GPU rules.

If local C-RADIO/DC-AE assets are available, resolve them locally and keep writable generated code caches under outputs or the normal cache directory; do not write into checkpoint targets. The prior pure inference wrapper disabled the unused LPIPS training term to avoid an unnecessary VGG download. Only reuse this change if that term is actually unused in the current inference mode; it is not appropriate for a requested optimization/training run.

Inspect front, back and side views for disconnected limbs, large floaters, fused legs, doubled headwear, missing back surfaces and severe silhouette mismatch. A visually plausible front image alone does not establish an animatable reconstruction.

## 4. Export the reference SOMA input

Use the same final reference framing supplied to Harvester, such as the padded 512×512 RGB image. In the GEM-X environment, a typical body inference command is:

```bash
gem-infer pose-image --task body --input <reference-image> \
    --release body --checkpoint checkpoint/GEM-X/body_s240000.ckpt \
    --body-detector checkpoint/GEM-X/yolox_humanart.onnx \
    --soma-assets checkpoint/GEM-X/soma-v0.3.3 \
    --device cuda --output <new-output-directory>/body.pt
```

Export the selected person's native parameters and geometry into a numeric NPZ for the web server. Do not assume the detector produced exactly one instance without checking. Important fields for the currently matched implementation:

| Field | Single-person representation |
| --- | --- |
| `body_pose` | 76 local rotation matrices, shape `(76,3,3)` |
| `global_orient` | root rotation matrix, `(1,3,3)` or supported `(3,3)` |
| `identity_coeffs` | 128 identity coefficients |
| `scale_params` | GEM-X packed57, first GS1 value fixed to 1 |
| `camera_translation` | 3-vector |
| `vertices_camera`, `faces` | SOMA MID geometry in its original vertex order |
| `joints_camera`, `joint_names`, `joint_parent_ids` | matching public joint positions/names/hierarchy |
| `intrinsics`, `keypoints_2d` | original image camera/projections when available |

Keep the original prediction bundle and a readable parameter summary. Obtain names/parents from matched SOMA assets, never a hand-written assumed joint order. The example MID geometry had 18,056 vertices, 36,108 triangles and 77 public joints; inspect the loaded version rather than forcing these counts on another release. Check finiteness, `R.T @ R ≈ I`, positive rotation determinants, face indices and overlay alignment. Record whether intrinsics are calibrated or estimated; a single uncalibrated image does not establish metric body size/depth.

## 5. Present the web alignment task

Start the local web server using `web_server/README.md`; provide its URL and the exact NPZ/PLY input paths. Explain that alignment uses overall translation, rotation and uniform scale. Inspect pelvis height, shoulders, elbows, knees, ankles and feet from multiple views. A global similarity cannot fix a wrong limb pose or incorrect anatomy; return to reference preparation if necessary.

The project world is right-handed X-front, Y-left, Z-up. Original Harvester and GEM-X frames may differ. Use the web initialization only for original imports; exported aligned inputs already contain the common basis. Saving bakes parameters and geometry together. A temporary session is not a durable artifact: export before closing it.

Wait for explicit human alignment confirmation before binding. Confirmation already supplied for the exact current export is sufficient. Record the export path and source hashes so later runs do not silently select a different `aligned-*` directory.

## 6. Bind, inspect and confirm LBS

Save the confirmed alignment, then generate weights with the web binding panel (default K=8). The worker recreates the full matched SOMA rig, finds nearest mesh triangles, interpolates vertex weights barycentrically, prunes tiny weights and renormalizes retained influences. The complete deformation rig currently has 110 joints, including procedural joints, while the public display has 77; these index spaces are different.

Inspect parameter/geometry reconstruction errors, binding-pose replay, weight row sums, surface-distance quantiles, discarded weight and `needs_review`. The current review flag uses distance relative to character height; it is a triage marker, not proof of semantic correctness. Close limbs and clothing can transfer to the wrong bone even at small distances.

Test moderate rotations on both arms, knees, hips and shoulders, including an asymmetric pose. Look for the opposite limb moving, clothing sticking to another body part, disappearing splats, stretched covariance, and clipping. Reset and check return to the original bound pose. The interactive web FK test does not recompute procedural constraints/correctives; production motion must use full SOMA target transforms.

After human LBS confirmation, export `soma.npz`, `gaussians.ply`, `gaussian_binding.npz`, and `metadata.json`. Check binding state and SHA-256 association. Any later alignment edit requires save and rebind; never animate with the stale binding. If review fails, preserve diagnostics and fix alignment/weights before generating a sequence.
