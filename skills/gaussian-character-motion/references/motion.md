# Motion recovery, retargeting and rendering

Read after the aligned bundle and LBS have been confirmed. Use the exported bundle as the source of truth. The following describes the current matched GEM-X/SOMA path; inspect current code/API contracts before adapting historical scripts.

## 1. Establish input lineage and timing

Record aligned bundle path/hashes, original reference SOMA path, selected person ID, camera, source paths and chosen interval. Check that reference body pose, identity and local scales agree with the aligned character where expected. Hash differences alone cannot establish or disprove lineage after alignment bakes geometry/parameters; retain the reference-to-aligned transform and provenance.

Choose a continuous N-frame interval. The person32 demonstration selected zero-based frames 4–33 from 45 frames by minimizing occluded frames, then maximizing summed annotated box area. All 30 chosen frames were marked unoccluded; timestamps spanned about 2.9 seconds and the 10 FPS output video duration was 3 seconds. This is an example selection, not a default person/interval for future work.

Read timestamps in their actual units and subtract the first timestamp before conversion to floating point. Record original source times and output times separately. A 10 FPS source must not simply be declared 30 FPS: this changes motion speed by three times. If resampling is needed, map both frame content and annotations to the same timeline and document interpolation/duplication. Resampling adds no observed detail.

## 2. Choose the motion input path

Prefer original full frames plus calibrated intrinsics and person-specific annotations. Independent moving tight crops discard image position, background and camera projection information, affecting root motion. When resizing full images, scale intrinsics and all boxes/keypoints by the same factors. Account for offsets if cropping or letterboxing; do not reuse original intrinsics unchanged.

`motion-video` normalizes its selected interval to 30 Hz and automatically chooses the person with largest cumulative detection area. It is suitable when that is clearly the target:

```bash
gem-infer motion-video --task body --input <source-video> \
    --config checkpoint/GEM-X/body_video_native77.yaml \
    --start <seconds> --duration <seconds> --device cuda \
    --output <new-output-directory>/motion.pt
```

For a specific annotated target, construct cached conditions from the supplied track, then use `motion-cache` or the Python Motion API. Do not assume `motion-video` accepts `pose-video`'s `--boxes` option; check help. In the local person32 wrapper, a process-local replacement of primary tracking supplied `TrackResult` and `square_crops` to `build_body_conditions`; this avoided third-party edits. Such private API adapters must verify current signatures and keep all track fields consistent with frame coverage.

The matched native77 condition set uses body checkpoint `body_s240000`, original `vitpose.pth` with release ID `vitpose77-dinov3-huge-metrosim-256x192-v1`, and the matching motion checkpoint/statistics sidecar. Configuration paths resolve relative to their YAML. Do not substitute another detector or joint layout merely because its output has 77 entries. Preserve release IDs and schema with cached conditions.

The Python path can preserve native 10 Hz observations using `frame_times` and `sampling_fps=10` with the matched physical-time configuration. However, `build_body_conditions` assumes 30 Hz: override its timeline before motion inference, and check camera angular-velocity interval conventions as well. Changing `sampling_fps` alone does not correct camera rotations/increments computed on the wrong clock. Prefer the supported normalized 30 Hz path when world rollout or Contact/IK requires it; sample the final animation to the user-requested output FPS.

Camera handling must reflect the recording: background mode estimates rotation approximately, static requires a genuinely fixed camera, and provided rotations must match the normalized timeline and conventions. Camera-space body turning can include camera motion; do not describe it as world-stable turning without compensation. Save raw conditions and raw camera/world predictions before retargeting.

## 3. Adapt predicted parameters to the bound character

Current motion outputs use axis-angle: `body_pose[1,T,228]` represents 76 three-component rotation vectors, and `global_orient[1,T,3]` represents the root rotation vector. Reference NPZ parameters instead use rotation matrices. Inspect actual shapes, then explicitly convert axis-angle to matrices; reshaping axis-angle values into matrices is invalid.

Start each frame from the aligned character parameter dictionary. Keep `identity_coeffs`, local `scale_params`, and `global_scale` fixed. Replace local body rotation matrices and the chosen root rotation/translation. Motion-predicted identity or scale should not resize the reconstructed character.

The current decoder `utils.soma_binding.SomaBindingRuntime` uses the matched SOMA runtime, `reference_pose=0.2.0` and `legacy_pose021` procedural profile. Packed57 has GS1=1: remove the first entry, then append four unit Foot/ToeBase scales to form local60 when using the native runtime. Use the stored native `global_scale` separately. Avoid a fixed-GS1 adapter path that silently ignores the aligned global scale.

A reference-derived orientation mapping can be computed, when the two references differ only by the documented common global transform:

```text
Q = R_aligned_reference @ transpose(R_original_reference)
R_target_project(t) = Q @ R_motion_camera(t)
```

Check `Q` is a proper rotation and confirm source camera axes/reference conventions match. This mapping converts basis; it does not remove time-varying camera rotation. For scene trajectory recovery use the calibrated world transform/scale and appropriate camera compensation rather than this orientation-only mapping.

Evaluate the unmodified aligned parameters first and compare the returned complete rig to `bind_world`, and geometry to cached aligned geometry. Stop on significant discrepancy; do not hide version/coordinate errors by fitting another transform afterward. The prior example used a `1e-4` maximum transform-element tolerance and obtained zero replay error; choose tolerances with the stored precision/scale in mind.

## 4. Define in-place behavior

Describe the root policy explicitly. A simple verified preview policy is:

1. Evaluate the target rotations with zero translation using the fixed character identity/scale.
2. Find Hips by name in the full rig, not an assumed index.
3. Set horizontal translation so Hips X/Y equals the bound Hips X/Y.
4. Set Z translation so the lowest target SOMA mesh vertex lies on the bound mesh floor, then evaluate again.

This preserves body turn and lets vertical position change with pose while removing horizontal travel. It is a floor placement heuristic: it is unsuitable for jumping and may introduce height jitter or foot sliding. For improved contact, solve foot contact/IK on the target character after shape/scale substitution, using temporal contact constraints. GEM-X's built-in Contact/IK currently requires continuous 30 Hz input; do not run it on native 10 Hz predictions without a supported conversion. Replacing identity/scale after source-body IK can undo contact correction.

Do not make a walking clip loop by blindly repeating frames; a seamless loop requires compatible endpoint pose, velocity and root handling, and should be requested or stated as an additional operation.

## 5. Drive Gaussian geometry

Decode every target frame to the complete deformation rig in the same world basis and exact joint-name order as the binding. The rig must include procedural transforms generated by SOMA, not only FK of displayed joints.

```text
D_j(t) = target_world_j(t) @ bind_inverse_j
p_i(t) = sum_j w_ij * (D_j(t) @ [p_i_bind,1])
A_i(t) = sum_j w_ij * D_j(t)[:3,:3]
C_i(t) = A_i(t) @ C_i_bind @ transpose(A_i(t))
```

Use `utils.gaussian_lbs.deform_gaussians` or the existing GPU covariance path. Always deform from the original bound Gaussians per frame; cumulative deformation introduces drift. Preserve row order, all stored K influences, DC color and opacity. PLY uses wxyz quaternions, natural-log scales and opacity logits; renderer inputs may require exp scales, sigmoid opacity and DC-to-RGB conversion. Do not pass logged scales or opacity logits directly to gsplat.

For standalone PLY frames, decompose the transformed covariance consistently and preserve finite positive eigenvalues. Do not rotate centers without transforming covariance, or truncate K=8/16/32 bindings to four influences.

## 6. Deliver and inspect

A useful output set is:

```text
<new-motion-directory>/
  README.md                   # choices, commands, assumptions and limits
  selection.json              # source indices/times/boxes and selection rationale
  conditions.pt / motion.pt   # cached conditions and raw predictions
  animation.npz               # target complete skeleton and fixed character parameters
  animation.mp4               # fixed-view rendered character
  comparison.mp4             # synchronized source and character (if useful)
  preview.jpg                 # representative start/middle/end frames
  validation.json             # hashes, finite checks, counts, drift, encoder metadata
  frames/                     # rendered PNGs, when retained
  ply/                        # optional per-frame Gaussian assets
```

For `animation.npz`, record FPS/output frame times, source times/indices, ordered joint names, `joint_world[T,J,4,4]`, body/root parameters, fixed identity/scales, coordinate description, and transform provenance. Include geometry/topology when useful but avoid duplicating large data unnecessarily. These fields describe an output contract to implement, not a claim that all historical example files already contain every field.

Use the already installed Harvester environment's gsplat renderer for real Gaussian previews; GEM-X's optional mesh renderer is not a Gaussian renderer. Set fixed cameras/framing from the whole sequence bounds with clothing margin. A camera following each pose can hide root drift or make size changes hard to assess. Retain camera extrinsics/intrinsics with the result.

Use FFmpeg from its installed environment, e.g. `conda run -n adam-gem-x ffmpeg ...`; verify its location before launching a long render. Encode per `configs/README.md`, with the requested frame rate if overridden. Check both encoded videos with ffprobe for codec/pixel format, dimensions, count, rate, duration and absent audio. Verify all target transforms and resulting Gaussian attributes are finite, binding hashes still match, weight sums are normalized, and in-place Hips horizontal drift is within expected precision.

Inspect start, middle, end and maximum-articulation frames against source crops. Note left/right mistakes, missing gesture timing, penetration, floating feet and stretching. A zero root drift metric does not establish motion accuracy. Clearly distinguish estimated motion, contact correction and cloth/appearance limits in the result summary.
