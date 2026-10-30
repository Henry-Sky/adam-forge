---
name: gaussian-character-motion
description: Build an animatable Gaussian character from a selected person in a scene, from Qwen A-pose and Harvester reconstruction through GEM-X/SOMA alignment, skinning, LBS review, and source-motion recovery.
metadata:
  short-description: Reconstruct and animate a Gaussian character
---

# Gaussian character reconstruction and motion

Use this workflow when asked to create a Gaussian character from a person in scene footage and animate it from source motion. Work in the repository's existing `outputs/` layout and use its installed tools, checkpoints, and README instructions. Treat source datasets and checkpoint directories as read-only. Put generated artifacts under a new `outputs/<project>/` directory and record inputs, settings, and validation there. Resolve repository paths in this skill from the repository root, not from the skill directory.

## Workflow

1. **Choose a suitable scene and person.** Inspect the scene inventory, frame coverage, timestamps, resolution, person annotations, occlusion, and camera views. Prefer a continuous interval with a clearly visible target, useful body articulation, stable identity, and minimal occlusion. Confirm the chosen person ID and source file paths; do not rely on an automatic primary-person detector when annotations can select the target explicitly. Save a compact inventory and representative frames.
2. **Crop and make an A-pose reference.** Crop the selected person from a clear source frame, retaining enough context for the full body. Use Qwen Image Edit to produce a clean A-pose reference while preserving identity, clothing, colors, and distinctive objects. Save the exact source crop, prompt, model/settings metadata, and generated reference. Check anatomy, full-body framing, and identity before proceeding.
3. **Reconstruct the reference with Harvester.** Prepare the A-pose image and mask as required by the local Harvester workflow, run reconstruction, and retain the Gaussian PLY and its provenance. Inspect multiple views for missing limbs, floaters, severe holes, scale, and appearance. Treat this PLY as the character surface representation; it is not a conventional triangle mesh.
4. **Estimate the reference SOMA model with GEM-X.** Run the matching GEM-X body pose model on the same A-pose reference. Preserve native SOMA parameters, identity, local scales, global scale, joint names/topology, geometry, camera assumptions, and model/version metadata. Validate finite parameters, rotation matrices, mesh topology, and overlays. Keep the matching SOMA assets and procedural profile available.
5. **Manually align both inputs in the web server.** Import the Harvester Gaussian PLY and GEM-X SOMA NPZ into `web_server`. Check the coordinate convention and axis initialization, then manually align translation, rotation, and uniform scale using front, side, and other useful views. Prepare the inputs and working web session for the user to align. Obtain the user's confirmation that alignment is correct before transferring weights; an existing explicit confirmation for the same exported bundle satisfies this gate. Save/export the confirmed aligned bundle and record its path. Keep the source model artifacts unchanged.
6. **Transfer skinning weights.** Use the web server binding panel on the saved aligned pair. It rebuilds the matched complete SOMA deformation rig and transfers surface weights to Gaussian centers. Wait for `ready`; inspect the reported surface distances, review flags, major joints, and file hashes. Export the aligned bundle including `gaussian_binding.npz`. Do not use a binding whose source hashes or aligned pose no longer match the Gaussian and SOMA files.
7. **Confirm LBS before motion recovery.** Use the web server's LBS interaction to drive representative joints, including elbows, shoulders, hips, and knees. Check both Gaussian splat and point inspection views when available. Confirm the deformation is finite, correctly oriented, and returns exactly to the binding pose after reset. Inspect tight limb/clothing regions where nearest-surface transfer may cross-assign weights. Regenerate binding after any alignment change. Obtain the user's confirmation that LBS is correct before recovering motion; reuse an explicit confirmation already given for the same binding. Record which aligned bundle and binding were confirmed.
8. **Recover motion and create an in-place animation.** Select the clearest continuous source interval for the same person. Keep source timestamps and physical frame rate accurate; use annotated boxes or tracks to avoid identity switches. Run GEM-X temporal Body Motion with the checkpoint-matched body pose and keypoint detector configuration. Fix character identity and local scale to the reconstructed character; use predicted body pose and a deliberate root policy. Decode each frame through the matching SOMA version/profile to the complete deformation skeleton (including procedural joints), then apply the saved binding using `target_world @ bind_inverse`. Do not pass the 77 public joints in place of the bound complete skeleton. For an in-place result, explain how horizontal root movement, turning, and vertical placement are handled. Preserve a camera-aligned option if world trajectory is also needed.
9. **Render and validate deliverables.** Produce a fixed-view preview, an animation, and optionally a source-video comparison. Keep a machine-readable skeleton/motion sequence and, when useful, per-frame Gaussian PLYs. Validate frame count, FPS, duration, video encoding per `configs/README.md`, finite transforms and Gaussian attributes, root behavior, and unchanged input hashes. Review representative frames across the whole sequence, including occlusions and transitions. Document known limits such as foot sliding, clothing deformation, missing view-dependent appearance, and non-looping endpoints.

## Detailed operating guides

- For stages 1–7, read [reconstruction and binding](references/reconstruction.md): selection criteria, crop conventions, segmentation, Qwen settings, Harvester preparation, SOMA NPZ export, human alignment and LBS review.
- Before motion inference, read [motion recovery and rendering](references/motion.md): target tracks, physical timing, parameter conversion, identity/scale retention, complete rig decoding, in-place root policies, Gaussian covariance deformation and deliverable checks.
- Historical scripts under ignored `outputs/` are optional examples. Use these guides and tracked repository tools when those artifacts are unavailable; do not hard-code person32 paths or infer new APIs from the examples.

## Repository references

- Environment setup and weight locations: `setup/README.md`, `checkpoint/README.md`, and `data/README.md`.
- Existing Qwen, Harvester, and GEM-X examples: `outputs/clip306-person32/README.md` and `outputs/clip306-person32/gem-x-soma/README.md` (example provenance only; discover the current scene rather than assuming these paths).
- Web alignment/export and binding workflow: `web_server/README.md`.
- Gaussian binding, LBS driving and target-transform format: `tools/README.md`.
- Default video encoding: `configs/README.md`.
- For a concrete end-to-end artifact layout, inspect the ignored example `outputs/clip306-person32/source-motion-30f-20261028/README.md`; adapt it to the current asset, source clip, and user-requested frame rate.

## Important invariants

- Keep upstream model submodule changes separate from repository tooling. Do not edit third-party sources to work around an inference issue; use a local wrapper or script.
- Use the exact GEM-X/SOMA version and procedural settings compatible with the stored SOMA parameters and binding. The repository's GEM-X environment pins `py-soma-x==0.3.3`; do not substitute another SOMA runtime without validating compatibility.
- Preserve the reconstructed character's identity and scale during motion transfer unless the user requests shape changes. GEM-X predicted identity/scale drift can change the Gaussian silhouette and invalidate the intended character.
- Maintain consistent coordinate bases from alignment through target transforms. Do not reapply axis conversion or global transforms to an already aligned bundle.
- Keep data and checkpoint sources read-only. Do not create undocumented symlinks, download unnecessary model assets, or put large outputs into tracked source directories.
- A successful binding-pose replay and mouse-driven LBS check are prerequisites for spending effort on a motion sequence.
