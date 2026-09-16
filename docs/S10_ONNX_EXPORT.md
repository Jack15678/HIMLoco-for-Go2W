# S10 HIM ONNX 导出

接口见 [共同契约](S10_HIM_ONNX_CONTRACT_V1.md)。导出完整 encoder → 原始 L2 normalize → actor，固定 FP32 `obs[1,342]` → `actions[1,16]`，opset 17，输出尚未裁剪/物理缩放。原 `PolicyExporterHIM` 和 JIT 一维接口保持不变。

## 环境

离线导出和保存钩子检查仅需 CPU，不加载 Isaac Gym。复用现有 PyTorch；安装导出依赖：

```powershell
python -m pip install onnx==1.16.2 onnxruntime==1.19.2
```

Dockerfile 已加入相同依赖。已存在的训练镜像需安装上述包或重新构建后使用。保存钩子检查还使用项目原有 tensorboard 依赖。

本机 Anaconda 的 ONNX 原生 DLL 存在加载/校验崩溃，本次实际验证使用独立 CPython 3.12.13 环境：`artifacts/onnx-clean/Scripts/python.exe`，PyTorch 2.5.1+cpu、ONNX 1.16.2、ONNX Runtime 1.19.2、NumPy 1.26.4、protobuf 4.25.8、tensorboard 2.14.0。下面的 PowerShell 命令使用此已准备的解释器；正常 Linux 训练环境可使用原 Python。

## 补导出已有 model_2000

在本仓库工作区根目录运行；下面用原仓库只读 checkpoint、保存配置和已评估 sidecar：

```powershell
$python = './artifacts/onnx-clean/Scripts/python.exe'
$run = 'D:/Desktop/Code/HIMLoco-for-Go2W/artifacts/s10-obsfix-scratch2000-20260916'
$handoff = 'D:/Desktop/Code/HIMLoco-for-Go2W/artifacts/s10-onnx-handoff/model_2000'
& $python legged_gym/scripts/export_s10_onnx.py --checkpoint "$run/model_2000.pt" --config "$run/config.json" --metadata "$run/assessments/checkpoint_2000/flat_gym/policy.json" --reference-jit "$run/assessments/checkpoint_2000/flat_gym/policy.pt" --observations "$handoff/mujoco_observations.npy" --output $handoff
```

旧 checkpoint 没有资产导入后的关节限值，因此要求提供匹配评估的 `--metadata`。其他控制参数逐项核对保存的 `config.json`，不使用当前默认训练配置。没有额外观测时可省略 `--observations`；仍执行零输入、初始化帧加零历史、32 组各帧不同的历史输入比较。额外观测必须为 FP32 `.npy [N,342]`。

导出将生成 `policy.onnx`、同 stem `policy.json`、`export_verification.json`。临时文件通过 ONNX checker、ONNX Runtime 与 checkpoint `act_inference` 比较后才发布；提供 `--reference-jit` 时也会比较已评估 JIT。报告记录样本、版本、来源、最大绝对/相对误差；通过条件为 `atol=1e-5, rtol=1e-4` 的逐元素比较，不是声称相对误差单独小于阈值。

## 后续训练自动保存

S10 的 `HIMOnPolicyRunner.save()` 对 model_0、定期保存、最后保存及脚本直接调用均生效。checkpoint 先保存，其中 `s10_export` 记录该次实际环境控制参数、资产限值、网络配置及维度；随后从磁盘相同权重快照在 CPU 重建模型，导出至：

```text
<checkpoint目录>/onnx/<checkpoint文件名去掉.pt>/policy.onnx
<checkpoint目录>/onnx/<checkpoint文件名去掉.pt>/policy.json
<checkpoint目录>/onnx/<checkpoint文件名去掉.pt>/export_verification.json
```

不同 checkpoint 不覆盖彼此。保存钩子使用相同 Python 解释器的独立进程，隔离实际遇到的 ONNX 原生 DLL 崩溃；导出不触碰训练模块设备、模式、权重、优化器或 RNG。直接调用离线函数时也会恢复 CPU RNG，验证样本使用独立 NumPy RNG。失败会打印 `Checkpoint saved ... ONNX export FAILED`、退出码/错误，保留 checkpoint，不报告成功。可以离线重试：

```powershell
python legged_gym/scripts/export_s10_onnx.py --checkpoint '<新checkpoint绝对路径>'
```

新 checkpoint 使用内嵌配置，拒绝 `--config/--metadata` 覆盖。Go2W 等其他机器人不自动导出。JIT play/evaluate 仍按原行为导出，ONNX 不改变网络、奖励、观测或训练周期。

## 无 PPO 的保存钩子检查

```powershell
& ./artifacts/onnx-clean/Scripts/python.exe legged_gym/scripts/check_s10_onnx.py
```

使用小型随机网络和环境参数替身调用真实 `save()`，检查导出、不同迭代目录、非法形状拒绝、导出失败仍保留 checkpoint、非 S10 不调用导出，以及权重/子模块模式/设备/优化器/CPU Torch、NumPy、Python RNG 不变。测试不创建真实 PPO 或仿真器。

本次交接中的 `mujoco_observations.npy` 为原已评估 JIT 在现有 `deploy/s10_mujoco.py` 观测/控制链下采集的 150 帧，采集信息见 `observation_capture.json`；`deployment_fixture.json` 提供由相同参考函数生成的观测、零历史、历史滚动、裁剪及腿/轮目标值，供比赛端接口检查使用。

本次 model_2000 实测共 184 个输入，最大绝对误差 `7.152557373046875e-6`、最大相对误差 `4.662840801756829e-4`，逐元素容差检查通过；已评估 JIT 对 checkpoint `act_inference` 最大误差为 0。输入输出名称/形状/FP32/有限值与 ONNX checker 均通过。150 帧采集对应约 3 秒，MuJoCo 警告计数全为 0。保存钩子检查已通过，日志在工作区 `artifacts/onnx-validation/save_hook_check.log`。未运行新训练、PPO 更新、真机、策略性能 A/B，也未验证 GPU 训练或重建 Docker 镜像。
