# Bundled Kinova Gen3 model

The robot MJCF and mesh files in `assets/kinova_gen3/` come from the Kinova Gen3 model in Google DeepMind's MuJoCo Menagerie. They are pinned to commit `d7b481bac63d3d600ab404ee74f7474c5efe8a67` so model changes upstream do not silently change this project. The upstream BSD 3-Clause license is included as `assets/kinova_gen3/LICENSE`.

The project-specific `scene.xml` adds a lightweight table, pedestal and mocap target ball. The collision-enabled task uses these as solid obstacles, while legacy checkpoint settings disable their collision masks. A base/shoulder overlap is explicitly excluded. `gen3.xml` adds an 8 mm spherical contact proxy at the unchanged `pinch_site`; this proxy is not a detailed gripper model. The upstream meshes, link inertias and seven joints remain the vendored Gen3 assets. The reinforcement-learning environment and Python scripts are the local CPU implementation; they do not contain Isaac Lab code.

The Python environment can retune actuator damping and add inverse-dynamics bias compensation inside the existing actuator force limits. Contact stiffness and the extra scene geometry are local simulation changes, not a claim of exact Isaac Lab or real-hardware dynamics.

Upstream model: <https://github.com/google-deepmind/mujoco_menagerie/tree/d7b481bac63d3d600ab404ee74f7474c5efe8a67/kinova_gen3>
