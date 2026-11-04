# 来源与发布范围

- 原Isaac Lab项目：[Kinova Gen3 RL & Sim2Real Toolkit](https://github.com/louislelay/kinova_isaaclab_sim2real)，固定提交 `d97c80ed8f532de5ab2609366f12e9d4c62e1ea3`。本仓库展示复现训练和对照评估，不将原作者算法、模型或工程架构归为原创。
- Kinova模型来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/d7b481bac63d3d600ab404ee74f7474c5efe8a67/kinova_gen3)，上游BSD-3-Clause许可证保留于 `simulations/Kinova_Gen3_MuJoCo_CPU_Lite/assets/kinova_gen3/LICENSE`。场景与控制修改见同目录的 `MODEL_SOURCE.md`。
- JAKA模型来源为 [JAKARobotics/jaka_ros2](https://github.com/JAKARobotics/jaka_ros2)，固定提交 `f6b28c6be93f289f101c6cc3907caf7804fb1a47`。上游description包许可证字段为TODO，因此本仓库不再分发其URDF、网格或派生MJCF，复现者需自行从来源取得资产并核实适用条件。
- 本仓库不包含厂商SDK、真实设备连接配置、真机运动程序或原作者预训练权重。
- 本地训练的MuJoCo checkpoint与配套仿真代码一并保存，仅用于对应仿真任务。JAKA checkpoint依赖精确场景哈希，重建场景不同会拒绝加载。
- 除明确附带第三方许可的文件外，本仓库暂不授予统一的开源许可证；公开展示不等于授权任意再分发。第三方权利不受仓库公开状态改变。
