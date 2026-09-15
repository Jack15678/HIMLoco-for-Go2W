# S10 asset source

Copied on 2026-09-15 from the official-material copy in
`D:/Desktop/Code/goai26-s10-racing/goai_embodied_future_material-main/goai_embodied_future_material-main/src/S10_sdk_deploy/S10_description/s10_mjcf`.
The containing repository HEAD was `bee3595b92c910d15822d5e2707bdf888436d7d6`;
the official-material directory is untracked, so that commit does **not** identify its contents.

Only `urdf/S10.urdf`, `mjcf/S10.xml`, and the 17 referenced STL meshes are copied.
Their relative references are preserved. No geometry or inertia is edited.

SHA-256 of the copied source files:

- URDF: `64903c21f925583b99d69220aef8214ec458498c23eacb7b83e8e48afd70f3e8`
- MJCF: `7855e80eb93e55f6bff63dbce195210ee0febf80de6cdfa40baaff6f55f43804`

The maintained `upstream` copies have one geometry difference: the left rear wheel
joint/body Y offset is 0.054891 m instead of the official copy's 0.045891 m.
No explanation was found in its patch script or project notes. This task uses the
matched official URDF/MJCF pair. The 17 meshes match both copies byte for byte.

Both formats specify total mass 18.987425 kg, wheel radius 0.081 m, and leg/wheel
effort limits 50/14 Nm. URDF velocity fields are 25.76/65.50 rad/s; they are model
parameters, not verified hardware ratings. MJCF has force limits but no velocity
limit. Base MJCF inertia omits the URDF's small off-diagonal terms; limb inertias
agree after rotating to their principal axes. Three massless fixed sensor links
are merged for Gym; actor inputs use body-frame angular velocity and gravity.

`S10_track.xml` includes `S10.xml`, `scene.xml`, and `track_overlay.xml`; it also
contains another free body and is not used as the training robot asset.

Control source: the same repository's
`artifacts/s10-speedturn-48-20260912/model-metadata.json` and `deployment.patch`,
cross-checked with `upstream/.../run_policy/s10_policy_runner.hpp` and
`docs/S10_SPEEDTURN_2000_TRIAL_ZH.md`. The selected steady RL parameters are leg
Kp/Kd=80/2, wheel Kp/Kd=0/0.6; hipx default 0, front hipy/knee=-0.3/0.6,
rear hipy/knee=0.3/-0.6; action scales 0.125/0.25/5 for hipx/other legs/wheels.
This selects actuator semantics only; no old actor or checkpoint is imported.

Control-source SHA-256 (paths relative to the source repository):

- `artifacts/s10-speedturn-48-20260912/model-metadata.json`: `239e12968395bb73cf05a04b2617a2d0d8b54b3e7ec288e7252ff83d3f8932ef`
- `artifacts/s10-speedturn-48-20260912/deployment.patch`: `055ec6447883bc9f764a2cb5ace542f00166f36ea450f74f9122261e0523e681`
- Official `interface/robot/hardware/s10_interface.hpp`: `777067e7cd338768442ff42715eb7cec21fc07576ff3ee8d5e04be068280a1f9`
- Official `run_policy/s10_policy_runner.hpp`: `ccf4436a2dc8a6d70a1ad54fc083fc735976addd0088c11d24a034cf253d37e1`

The original runner uses a 150-policy-frame (3-second) entry ramp in action and
gains. The current maintained runner removed it. HIM preparation starts in the
configured pose with the steady gains, as the existing Gym task does; deploying
HIM through that hardware state machine will require an explicit entry adapter.

Full checks, selected timing and train/export mapping are recorded in
`docs/S10_HIM_PREPARATION_20260915.md` at repository root.
