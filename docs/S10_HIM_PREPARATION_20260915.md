# S10 接入准备与证据（2026-09-15）

状态：准备及零 PPO 检查已完成；独立环境任务 `s10` 已注册，正式训练尚未启动。
本次正式 PPO 更新预算尚未选择；实际 PPO 更新数为 **0**。

## 1. 工作区和环境

- 原仓库基准为 `0116937`，开始时只有未跟踪的交接文档。服务器原始检查使用该提交的独立归档 `/data/him-runtime/original`，只添加检查脚本。
- 新工程目录 `/data/HIMLoco-for-Go2W`；Gym 包、构建文件和原始检查日志放在 `/data/him-runtime`。已有 `/data/rl_training` 及其镜像不作修改。
- 本次服务器实测：Ubuntu 22.04.5、RTX 4090（23028 MiB）、驱动 580.173.02、Docker 29.1.3、Compose 2.40.3。开机后没有运行中的容器或 GPU 计算进程，`/data` 初始可用约 51 GB。
- 独立 Python 3.8 / PyTorch 2.0.1+cu118 容器已通过原版 Go2W GPU 步进与 HIM 推理。上游公布的测试组合为 Python 3.7.16 / PyTorch 1.10.0+cu113 / Gym Preview 4，与本次实测组合不同。[上游安装说明](https://github.com/InternRobotics/HIMLoco#installation)
- Gym 从 [NVIDIA 官方归档](https://developer.nvidia.com/isaac-gym/download) 下载，未采用同名 PyPI 替代品。Docker Hub 超时后通过公开 ECR mirror 获取 Python 镜像；Debian 11 仓库包缺失后改用 Debian 12 用户态。主机驱动未调整。

最终镜像：`him-gym:py38-torch201-cu118`，Docker inspect ID `sha256:5b046b36ca01bb17012bd472c0ee99b3c75cb6c72eb5335cc497c1e62b53f27c`。完成时 `/data` 可用 31.67 GiB。

| 组件 | 实测版本 |
|---|---|
| Python / GCC | 3.8.20 / 12.2.0 |
| PyTorch / torchvision / CUDA runtime | 2.0.1+cu118 / 0.15.2+cu118 / 11.8 |
| Isaac Gym | NVIDIA Preview 4；包元数据版本 1.0rc4；`gym_38` 与 `gymtorch` 加载通过 |
| 本仓库包 | legged-gym 1.0.0；rsl-rl 1.0.2；实际加载 `/workspace/him/legged_gym`、`/workspace/him/rsl_rl/rsl_rl` |
| NumPy / SciPy / Matplotlib | 1.23.5 / 1.10.1 / 3.7.5 |
| TensorBoard / PyYAML / Ninja | 2.14.0 / 6.0.2 / 1.11.1.4 |
| MuJoCo | 3.2.3（Python 3.8 可用版本；3.2.7 无对应包） |

详见 [环境与逐关节地址](evidence/s10-20260915/environment.json)、[完整 pip freeze](evidence/s10-20260915/pip-freeze.txt)。`pip check` 通过。旧 Ninja 1.11.1.1 的 wheel 标签元数据导致平台检查失败；改用 [1.11.1.4](https://pypi.org/project/ninja/1.11.1.4/) 后复核 S10 GPU 检查通过。早先原版/回归/容量检查使用的是同一运行库和旧 Ninja 构建工具。

PyTorch 官方 wheel 已校验：`torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl` 的 SHA-256 为 `2ce38a6e4ea7c4b7f5baa51e65243a5f687f6e19ab7915ba5b2a431105f50bbe`，与官方索引一致。服务器构建复用该下载文件，普通依赖通过清华镜像获取；最终版本以以上清单为准。

## 2. 接口判断和处理

| 项目 | 实际代码行为与依据 | 处理 |
|---|---|---|
| 轮速奖励 | `_reward_dof_vel` 原来直接清零 Gym view；本仓库 MuJoCo 端读取真实轮速。Git 历史中这一行为只见于首个 Go2W 导入，没有解释为何应屏蔽观测轮速。 | 只在平方速度的局部张量中屏蔽轮子，奖励仍排除轮速；状态、观测、速度缓存保留真实值。 |
| 轮角观测 | 连续轮绝对角不进入 actor，与 S10 runner 一致；但三个观测函数还额外写原状态。 | 保留 `q-q_default` 的轮切片为零，删除三处原状态写入。 |
| reset 历史 | 上游和此 fork 都保留旧历史，但没有找到刻意跨物理 reset 保留历史的说明；本仓库部署入口从零历史开始。原先仅清 `last_actions`，当前 `actions` 又写回了上一动作缓存。 | reset 环境的历史和当前/上一动作清零；新帧在前、后五帧为零。按重置后的 root state 更新角速度、重力和速度，终止观测仍先保存。 |
| hip 索引 | 原来是 `[0,4,8,12]`；S10 实际导入后仍对应四个 hipx。 | 保留原实现。轮和接触索引继续使用原有按名称查找机制。 |
| 连续轮内部驱动 | Preview 4 将 S10 连续轮导入为刚度 `3.402823×10^38`，实际外加 3 Nm 时轮速只有约 `10^-7` rad/s。腿关节导入刚度为 0，轮 hasLimits=false，质量/惯量正确。 | 在统一力矩控制入口显式设 effort 模式、内部 stiffness/damping=0；PD 仍由 `_compute_torques` 每 5 ms 计算。原资产和 S10 的 80/2、0/0.6 增益不改；修后 Go2W 回归通过。 |
| HIM 目标输入 | 上游帧以命令开头，估计器 `3:...` 去掉命令后接真实速度；Go2W 把命令移至 6:9，却未改此切片，实际丢掉角速度。 | 按当前帧去掉 6:9，保留角速度等本体量并接真实速度。网络规模、损失、PPO 和估计器算法不改。 |
| 恢复/导出入口 | `train.py` 原来强制关闭 resume；训练循环中间保存的迭代计数未更新；load 后自适应学习率标量仍是初始值。play 提前取得的 obs 在 runner reset 后过时。 | 恢复 CLI 传递，保存实际完成的迭代数和运行计数，恢复学习率；play 从 reset 后取 obs 并在推理前更新命令切片；导出增加与实际 DOF 顺序对应的 `policy.json`。 |

上游对照：[环境帧与 reset](https://github.com/InternRobotics/HIMLoco/blob/main/legged_gym/legged_gym/envs/base/legged_robot.py)、[HIM 估计器](https://github.com/InternRobotics/HIMLoco/blob/main/rsl_rl/rsl_rl/modules/him_estimator.py)。这里以调用关系和训练/部署一致性作判断，不以“源码一直如此”证明意图。

注意：`dof_acc` 按奖励名称排序先于 `dof_vel`。原轮速写零还导致下一步轮加速度使用了错误的上一轮速；修复后恢复相邻物理采样的差分。旧 Go2W checkpoint 的输入与该副作用可能耦合，本次接口版本为 2，不能宣称与旧策略行为等价。本次不加载旧 Go2W checkpoint，不开展 A/B 训练。

## 3. S10 来源与最终配置

资料库：`D:/Desktop/Code/goai26-s10-racing`，HEAD `bee3595b92c910d15822d5e2707bdf888436d7d6`，工作区已有修改，未改动该库。没有发现额外 AGENTS.md；已读 CONTRIBUTING.md。官方资料目录为未跟踪文件，来源版本另以复制文件摘要记录，见 [资产来源](../resources/robots/s10/SOURCE.md)。

| 项目 | 选定值 / 语义 | 来源与实现 |
|---|---|---|
| 模型 | 官方资料副本的配套 URDF、基础 MJCF、17 个 STL；18.987425 kg | `resources/robots/s10/`；维护副本左后轮横向多 9 mm，未查到修改依据，未继承此变更。 |
| 轮径 | 半径 0.081 m，直径 0.162 m | 两个源模型的碰撞圆柱一致；网格米制尺寸检查。 |
| 默认姿态 | hipx=0；前 hipy/knee=-0.3/+0.6；后=+0.3/-0.6；轮=0 | 48 号隔离部署的 `model-metadata.json`、`deployment.patch`，与当前维护 runner 一致；新配置观测和解码共用同一默认角。 |
| 初始/目标高度 | 初始 z=0.45 m；奖励目标 0.425 m | 选定姿态 FK 支撑高度 0.424921 m；初始轮底约高出地面 0.025079 m。 |
| PD | 腿 80/2；轮 0/0.6 | 部署 metadata 和补丁；官方原 runner 的轮 Kd=0.8 属于另一配置。 |
| 动作 | hipx 乘 0.125 rad，hipy/knee 乘 0.25 rad，轮乘 5 rad/s | `s10_config.py` → 共享混合 PD；腿 q_target=q_default+scale×raw，轮 dq_target=5×raw，tau_ff=0。 |
| 限幅 | raw action 按 HIM 原规则裁到 ±100；力矩腿/轮 ±50/±14 Nm | URDF effort 与 MJCF actuator limit 相符；未添加 raw ±1 或轮速 ±5 限幅。 |
| 速度字段 | 腿 25.76、轮 65.50 rad/s | URDF velocity；MJCF 没有对应速度硬限。该字段及 12/30 rad/s 诊断停止线均不能当作已证实的硬件额定值。 |
| 时序 | Gym dt=5 ms，decimation=4；策略和历史间隔 20 ms（50 Hz） | 保留 HIM；旧 S10 SDK 状态循环 5 ms、策略每 4 次执行。新 MuJoCo runner 显式设置 dt 并每物理步重算 PD。 |
| 动作延迟 | 训练每环境每周期采样 0–3 物理步，即 0–15 ms | 保留原 HIM `step()`；确定性检查关闭；MuJoCo 验证为零附加延迟。未继承旧平地 2.5 ms×8 / 0–2 步设定。 |
| 接触 | wheel 为轮端；hipy/knee/base 为惩罚；base 为终止 | 按 Gym 导入名称获得真实索引；critic 仍是 262D，含 4×3 轮端接触力和 187 高度点。 |
| 自碰撞 | `self_collisions=1` 禁用全部机器人自碰撞 | Gym Preview 4 `docs/programming/assets.html` 明确 `>0` 的含义；保留 HIM 配方，新 MuJoCo runner 用碰撞位掩码对齐。 |
| 固定关节 | 合并三个无质量的 IMU/前后雷达固定 link | `collapse_fixed_joints=True`；模型 IMU 固定 link 的 RPY 不用于再次旋转已是机身坐标的 actor 角速度。 |
| actor / critic | actor=6×57=342，最新帧在前；critic=262 | actor 只含本体历史；高度、接触、外扰仍只在训练侧。 |

### 命令与 SDK 转换

- 48 号隔离部署：手动轴 → DDS 原样传递 → 官方 runner 乘 `[1.5,0.5,0.6]` → 57D actor 直接接收物理指令。
- 当前维护副本的 ROS 路径：`Twist` 的 m/s、rad/s → `RosCmdInterface` 原样传递 → 修改过的 runner 原样写入观测。
- 新 HIM：最终物理 `[vx,vy,wz]` 再乘 `[2,2,0.25]` 进入 actor。新 MuJoCo 命令行直接接收物理速度，不重复做手柄缩放。
- 保留原 heading 回调：`clip(0.5*wrap_to_pi(error), -2, 2)`。由于 wrap 范围为 ±π，通常实际上界约 ±1.571 rad/s，不能把配置的 yaw 采样范围直接当作 heading 模式范围。
- SDK 原始反馈到模型坐标：`q_model=q_sdk*dir+offset_rad`、`dq_model=dq_sdk*dir`；下发反向转换为 `q_sdk=(q_target-offset_rad)*dir`，速度和前馈力矩乘 dir。转换只在 SDK 边界执行，Gym/MuJoCo 模型坐标不再转换。
- SDK 顺序的 dir：`[1,1,-1,1, 1,-1,1,-1, -1,1,-1,1, -1,-1,1,-1]`；初始 offset（度）：`[-35,-145,156,0, 35,-145,156,0, -35,145,-156,0, 35,145,-156,0]`。源码还会根据读回角调整 hipy/knee 的 ±360° 分支，不能在模型内固化一份未经实机初始化的偏置。
- 48 号实测 IMU 姿态已经是弧度；隔离副本移除了重复度转弧度。现有 ROS MuJoCo 仿真器却发布角度制 Euler，与它自身的旧 DDS 读端配对。新 MuJoCo runner 直接取模型旋转和机身角速度，避开该消息单位差异。

旧 57D ONNX runner 不能直接加载本次完整 HIM TorchScript。当前交付是仿真推理端；AGX/SDK 的模型格式及进入 RL 的 3 秒增益/动作渐变适配仍属于后续部署工作，不在服务器检查中驱动实机。

### 实际关节和接触映射

Gym 的 16 DOF 与 SDK 均按 `fl, fr, hl, hr`、每腿 `hipx, hipy, knee, wheel` 交错排列。MuJoCo 通过关节名和 actuator 名查找，且逐个验证 actuator 指向对应 joint；Gym/MuJoCo 各完成 16 个单物理步正向力矩响应检查。

| 腿 | Gym / HIM / SDK 索引 | 对应官方 57D actor 动作索引 |
|---|---|---|
| fl | 0, 1, 2, 3 | 0, 1, 2, 12 |
| fr | 4, 5, 6, 7 | 3, 4, 5, 13 |
| hl | 8, 9, 10, 11 | 6, 7, 8, 14 |
| hr | 12, 13, 14, 15 | 9, 10, 11, 15 |

HIM → 官方动作排列：`him[[0,1,2,4,5,6,8,9,10,12,13,14,3,7,11,15]]`。SDK/仿真之间仍须按前述方向、偏置转换，顺序相同不等于数值原样透传。

Gym 合并后共 17 个 body：base_link=0；轮端 body `[4,8,12,16]`；惩罚 body `[2,6,10,14,3,7,11,15,0]` 对应 hipy、knee、base_link；终止 body `[0]`。轮 DOF `[3,7,11,15]` 均无角度限位，轮碰撞保持圆柱（`replace_cylinder_with_capsule=False`）。

## 4. 检查结果

机器可读结果见 [results.json](evidence/s10-20260915/results.json)，导出的控制契约见 [policy.json](evidence/s10-20260915/policy.json)。检查脚本为 `legged_gym/scripts/check_zero_ppo.py`；下表 GPU/接口检查最终退出码为 0，预算拒绝测试按预期返回错误。PPO 与估计器优化器更新次数均为 0。

| 检查 | 实测结果 |
|---|---|
| 原版 Go2W，2 环境 | GPU PhysX / GPU pipeline 均开启；48 步物理采样和 HIM 推理通过；342D actor、262D critic；JIT 最大误差 `1.12e-8`。 |
| 原版接口复现 | 奖励改写轮速=true，观测改写轮角=true；强制 reset 后旧历史非零数 285、上一动作非零数 16；终止观测 `[1,262]` 和终止动作保留。 |
| 修正版 Go2W，2 环境 | 两项状态改写均=false；reset 后旧历史/上一动作非零数均为 0；未 reset 的另一环境历史连续；终止观测保留；三个观测入口均保持轮角屏蔽而不改状态。 |
| HIM 目标与恢复 | 编号输入确认 target 为列 `0:6 + 9:60`；在 target 前向钩子处中止检查，未执行 optimizer.step。临时 checkpoint 的模型/探索标准差、两个优化器、学习率、合成计数器往返通过；未调用 learn。 |
| S10 导入与方向 | 17 body / 16 DOF；Gym 和 MuJoCo 各 16 个正向单步力矩响应通过；连续轮无角度限位，显式 effort 驱动，内部 stiffness/damping=0。 |
| S10 零动作 5 秒 | Gym 两环境最终高度 `0.422055 / 0.422059 m`，无终止；最后一秒最大关节速度 `0.111638 rad/s`。MuJoCo 高度 `0.420153 m`，状态有限、无警告。 |
| S10 同输入接口 | 57D 观测最大误差 0；完整 estimator+actor 的 raw action 最大误差 `1.86e-8`；解码/PD 力矩最大误差 `9.54e-7 Nm`。raw 大于 1 时轮目标可超过 5 rad/s。 |
| 模型格式 | MuJoCo 3.2.3 中带非零关节角的 URDF/MJCF FK 最大位置误差 0；引用资源齐全，惯量检查通过。 |
| S10，4096 环境 | 保留混合地形和域随机化，完成 48 步真实采样、critic 推理、存储写入和回报计算；观测、奖励、回报均有限。Torch 分配峰值约 797 MiB；检查时 GPU 总占用 5357 MiB、剩余 17197 MiB。 |
| 完整 MuJoCo 命令入口 | 使用临时随机 HIM 导出运行约 1 秒：200 次物理/PD 步、50 次策略调用，实际 dt `0.004999999888 s`；无警告、无非有限状态。 |
| 训练预算入口 | `--initialization scratch --max_iterations 0` 被入口拒绝，未创建训练环境；S10 默认预算为 None，不能静默使用 20000 次。 |

方向检查取施加正力矩后的首个物理步；PD 一个周期末的速度可能因制动变负，不能用它判断关节轴反向。最初连续轮被内部弹簧锁住的原始属性和响应保存在 [continuous-drive-before.txt](evidence/s10-20260915/continuous-drive-before.txt)。未用训练或 A/B 测试诊断这些问题。

以上证明环境和接口可执行。4096 检查没有进行反向传播或优化器更新，不能作为完整 PPO 的显存峰值或收敛能力证明；随机策略也不作为运动能力验收。当前没有新的训练 checkpoint，`artifacts/s10-preparation/policy.pt` 仅是零 PPO 检查产物。

服务器原始日志：`/data/him-runtime/{go2w-original,go2w-fixed,s10-two,s10-capacity,s10-mujoco-cli}.log`。

此前独立本地准备检查（与服务器结果分别记录）：

- MuJoCo 3.11.0 加载官方 URDF、基础 MJCF，以及维护副本赛道 XML；赛道有额外自由体，不采用为训练资产。
- 所有有质量 link 的惯量正定、满足三角不等式；两格式质量一致。带非零关节角的 URDF/MJCF FK 最大位置误差为 0。
- 官方模型 5 ms PD、所选控制参数、零动作 5 秒：高度 0.420153 m、末时刻最大关节速度 0.003461 rad/s，无 MuJoCo 警告。
- 本地 CPU PyTorch 2.5.1 的 HIM 推理和导出 forward 一致，最大误差 0；编号输入确认估计器去掉命令并保留角速度，未执行 optimizer.step。它不能代替服务器组合实测。

## 5. 入口和剩余选择

在服务器新工程目录运行：

```bash
cd /data/HIMLoco-for-Go2W
docker compose run --rm him
docker compose run --rm him python legged_gym/scripts/check_zero_ppo.py --task s10 --num_envs 4096 --headless --strict
```

确定初始化和预算后，正式入口为：

```bash
# N 必须由本次新实验明确选择。以下命令尚未执行。
docker compose run --rm him python legged_gym/scripts/train.py --task s10 --headless --initialization scratch --max_iterations N --run_name RUN_NAME
# 中断恢复：N 是本次追加迭代数；计数、优化器和学习率从指定 checkpoint 恢复。
docker compose run --rm him python legged_gym/scripts/train.py --task s10 --headless --resume --load_run RUN_DIRECTORY --checkpoint K --max_iterations N
docker compose run --rm him python legged_gym/scripts/play.py --task s10 --headless --load_run RUN_DIRECTORY --checkpoint K
docker compose run --rm him python deploy/s10_mujoco.py --policy logs/S10_HIM/exported/policies/policy.pt --seconds 10 --command 0 0 0
```

训练保存 `config.json` 和可恢复 checkpoint；导出包含 estimator encoder+actor 的 `policy.pt` 及真实导入顺序的 `policy.json`。恢复不是物理状态/RNG 的逐位重放。

还需用户选择：

1. **初始化**：HIM 完全从零；或官方 actor 热启动。热启动尚未自动实现，需要按本次关节、命令、默认姿态语义适配并验证初始数值一致。
2. **PPO 预算**：正式迭代数，以及是否包含明确次数的非零 PPO 接入短测。未采用默认 20000 或任何旧任务预算。

未使用子代理，未开展 A/B 测试。随机网络只用于临时零 PPO 接口检查，其输出不作为运动能力验收。
