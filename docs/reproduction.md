# 运行与复现

## 指标复核

Python 3.10及以上，无第三方依赖：

```bash
python tools/verify_results.py
```

程序从逐窗口记录重算Isaac Lab计数，复核MuJoCo汇总及发布文件哈希。进一步从原始状态数组复核Isaac Lab保持计数、目标与重置一致性，需要NumPy：

```bash
python -m pip install numpy
python tools/verify_results.py --raw
```

这是既有实验数据的重统计，不会重新训练或执行策略。

## Kinova CPU仿真

在仓库根目录创建独立Python 3.12环境；以下Windows PowerShell命令使用环境内解释器，不依赖激活脚本：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r simulations/Kinova_Gen3_MuJoCo_CPU_Lite/requirements.txt
Set-Location simulations/Kinova_Gen3_MuJoCo_CPU_Lite
..\..\.venv\Scripts\python.exe check.py
..\..\.venv\Scripts\python.exe play.py
..\..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

记录中的依赖版本保留在requirements中；如目标平台无法安装对应版本，应使用兼容版本单独验证，不将版本替换后的结果视为原实验复现。

在Kinova目录重新评估，输出使用新文件名：

```powershell
..\..\.venv\Scripts\python.exe evaluate.py --episodes 100 --seed 410000 --report-json reports/repeat_short.json
..\..\.venv\Scripts\python.exe evaluate.py --episodes 20 --episode-seconds 60 --seed 510000 --report-json reports/repeat_long.json
```

## JAKA CPU仿真

先按NOTICE说明自行取得官方模型。`third_party/`与重建资产已被Git忽略。在仓库根目录运行：

```powershell
git clone https://github.com/JAKARobotics/jaka_ros2.git third_party/jaka_ros2
.\.venv\Scripts\python.exe simulations/JAKA_ZU5_Real/import_model.py --source-repo third_party/jaka_ros2
.\.venv\Scripts\python.exe -m pip install -r simulations/JAKA_ZU5_Real/requirements_sim.txt
Set-Location simulations/JAKA_ZU5_Real
..\..\.venv\Scripts\python.exe check_zu5.py
..\..\.venv\Scripts\python.exe -m unittest discover -s . -p "test_*.py" -v
..\..\.venv\Scripts\python.exe play_zu5.py --model runs/zu5_first_300k/best_model.zip --continuous --seed 510000
```

重建脚本固定上游提交；模型校验通过后才可加载随附checkpoint。场景序列化或模型资产变化可能导致哈希不同，此时应检查差异或重新训练，不移除校验。

## Isaac Lab

本仓库提供配对评估的原始状态和逐窗口记录，支持重统计；不包含完整服务器训练环境、Seed 44权重或作者权重。执行训练或重新采集rollout需另行配置上游固定版本。不得将 `--raw` 数据验证描述为重新完成了训练或策略运行。

## 验证记录

发布前实际执行的检查与结果保存在 `VALIDATION.md`。仿真测试不连接硬件。
