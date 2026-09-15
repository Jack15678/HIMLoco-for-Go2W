# S10 接入 HIMLoco-for-Go2W：新对话交接

日期：2026-09-15。交接时本仓库 HEAD：`0116937`。

## 目标与当前状态

在 `D:\Desktop\Code\HIMLoco-for-Go2W` 中接入 S10，先完成旧环境运行检查、必要的接口修正和 S10 控制核对，再按本仓库 HIM 的训练流程建立独立实验。第一阶段尽量保留原算法、奖励、地形课程和命令生成方式，减少同时变化的内容。

本文件是准备与执行交接，不是运行结果。本次对话只检查了本地代码和资料、创建了本文档；没有安装服务器环境、修改训练代码或启动 PPO。下文明确区分已看到的代码、需要验证的问题和建议。

- 未经用户明确同意，不使用子代理；做 A/B 测试前先询问用户是否有必要。确定性的接口检查不需要以两条训练链代替。
- S10 模型和控制参数以 `goai26-s10-racing` 中选定并核验的资料为来源，`rl_training` 仅用于对照和复用经验。
- HIM 原生从零训练是前面对话中的建议，用户尚未明确选定；不能把“旧 checkpoint 不能直接恢复”理解成“必须随机初始化全部网络”。先完成不依赖该选择的准备。
- 本 HIM 实验尚未确定正式 PPO 预算。不得直接使用仓库默认的 20000 次，也不得继承旧 M20/TurnStop 的 800、1000、5000 等预算。非零 PPO 接入短测同样需要明确次数并计入新实验。
- 新对话如获得初始化方案、训练预算或其他明确指令，以用户最新指令为准，不重复询问已经确认的事项。

## 1. 机器人和控制参数需要真实适配

### 1.1 已有基础

Go2W 和 S10 都是 12 个腿关节加 4 个轮关节。本仓库 `_compute_torques()` 已实现腿部位置目标与轮部速度目标混合控制，所以不需要重新发明轮腿控制框架。

但“都是 16 个动作”不代表可以只换 URDF。先核对：

| 内容 | 必须得到的结果 |
|---|---|
| 模型 | S10 URDF、MJCF、引用网格可以加载；单位、质量、惯量、轮径、关节轴、碰撞体和固定关节合并方式明确 |
| 关节映射 | Isaac Gym 实际导入顺序、策略顺序、MuJoCo 顺序和 SDK 顺序逐项对应；按实际关节名称生成映射，不靠模型文件排列猜测 |
| 方向和零位 | 核对 SDK 的关节方向、零位偏置及角度单位；转换只执行一次 |
| 默认姿态 | 观测中的 `q_default` 和动作解码中的 `q_default` 一致，前后腿方向正确，初始机身高度与所选姿态匹配 |
| 执行器 | 12 腿位置控制、4 轮速度控制；PD、动作缩放、力矩限制、速度约束与选定部署配置相符 |
| 接触 | 轮端、腿和机身索引有效；惩罚与终止识别的对象正确，自碰撞掩码含义已核对 |
| 时序 | 策略周期、物理步长、PD 更新频率、动作延迟和观测历史时间间隔一致 |

本仓库 Go2W 配置是腿 `Kp=40/Kd=1`、轮 `Kp=0/Kd=0.5`、腿动作乘 `0.25`、轮动作乘 `10`；不能作为 S10 参数照搬。参见 [Go2W配置](D:/Desktop/Code/HIMLoco-for-Go2W/legged_gym/envs/go2w/go2w_config.py)。

### 1.2 S10 参数的含义与已发现差异

腿部控制关系为 `tau = Kp * (q_target - q) + Kd * (dq_target - dq) + tau_ff`，再执行与模型/驱动相符的限幅。`Kp/Kd` 是可调控制增益，动作缩放是网络输出的物理解释，均不是电机不可改变的硬件常数。

| 项目 | `goai26-s10-racing` 官方资料副本 | 该仓库当前 `upstream` 部署副本／现有训练对照 |
|---|---|---|
| 腿 Kp/Kd | 80 / 2 | 80 / 2 |
| 轮 Kp/Kd | 0 / 0.8 | 0 / 0.6 |
| 动作缩放 | hipx 0.125；hipy/knee 0.25；轮 5 | 相同 |
| 前腿默认姿态 | 左右 hipx 为 +0.05/-0.05；hipy=-0.35、knee=0.65 | hipx=0；hipy=-0.3、knee=0.6 |
| 后腿默认姿态 | 左右 hipx 为 +0.05/-0.05；hipy=0.35、knee=-0.65 | hipx=0；hipy=0.3、knee=-0.6 |

这些是本地代码中看到的值，不代表已经核验当前实机正在运行哪份配置。具体文件见第 6 节。不要把两套值拼成没有出处的新配置，也不能因为路径叫 `upstream` 就认为其中没有本队修改。

建议以选定 S10 配置的腿 `80/2`、动作缩放 `0.125/0.25/5` 为起点；轮 Kd、默认姿态及其他参数先结合实际准备部署的 runner、模型和既有记录决定。如有明确来源足以解决差异，直接按证据选定并记录理由；证据不足且会改变部署行为时，再向用户提出具体选择。

`wheel_target = raw_action * 5 rad/s` 不等于轮速上限 5 rad/s。现有接口不把 raw action 限制在 ±1；不能额外加这种裁剪。PD 的选择还受负载、惯量、延迟、控制周期和力矩饱和影响，不能把“站得住”当作所有动态控制已验证。

现有 `rl_training` 的腿/轮限扭 50/14 N·m、速度参数 25.76/65.5 rad/s 仅作对照：新任务需从第 6 节源文件核验，并区分 URDF/MJCF 限制、软件诊断停止线和已证实的硬件能力。不得将曾经的轮速 12/30 rad/s 诊断停止线直接写成电机硬限。

## 2. 现有 S10 模型不能直接拿来续训

### 2.1 网络和数据接口不同

| 项目 | 当前 S10 57D actor | 本仓库 HIM Go2W |
|---|---|---|
| 部署输入 | 当前一帧 57D | 6 帧 × 57D = 342D，最新帧在前 |
| 动作网络 | 57 → 512 → 256 → 128 → 16 | 当前57D + 估计速度3D + 内部特征16D，共76D进入动作网络 |
| 估计器 | 无 HIM 估计器 | 342D历史输入，输出3D速度及16D特征 |
| critic | 当前平地60D | 当前Go2W配置262D，含速度、外扰、187点高度和12维轮端接触力 |
| 部署产物 | 现有 ONNX runner | 当前导出 TorchScript，包含估计器编码器和动作网络 |

参见 [HIM网络](D:/Desktop/Code/HIMLoco-for-Go2W/rsl_rl/rsl_rl/modules/him_actor_critic.py)、[估计器](D:/Desktop/Code/HIMLoco-for-Go2W/rsl_rl/rsl_rl/modules/him_estimator.py)、[训练runner](D:/Desktop/Code/HIMLoco-for-Go2W/rsl_rl/rsl_rl/runners/him_on_policy_runner.py)。

### 2.2 关节排列和命令缩放不同

当前 S10 官方策略顺序是四条腿的12个关节在前，4轮在后：

```text
fl_hipx, fl_hipy, fl_knee,
fr_hipx, fr_hipy, fr_knee,
hl_hipx, hl_hipy, hl_knee,
hr_hipx, hr_hipy, hr_knee,
fl_wheel, fr_wheel, hl_wheel, hr_wheel
```

S10 SDK 常用顺序则是每条腿插入一个轮。现有 S10 policy → SDK 重排为：

```text
[0,1,2,12, 3,4,5,13, 6,7,8,14, 9,10,11,15]
```

HIM 的训练观测/动作跟随其实际 DOF 顺序；Go2W MuJoCo 配置按每腿四关节排列，轮索引 `[3,7,11,15]`。接 S10 后必须重新核对 Isaac Gym 导入顺序，不能直接假定这个索引组或上面的 S10 重排仍适用。

HIM 当前将最终物理指令 `[vx, vy, wz]` 乘 `[2, 2, 0.25]` 后放入观测。我们现有 S10 57D 契约把物理单位的这三项直接放入观测。还需区分：

1. 手柄/网页轴值 → 物理速度指令；
2. 航向模式下 heading error → 最终 yaw-rate；
3. 最终物理指令 → actor 中的观测缩放。

官方资料副本在 runner 内做手柄轴乘 `1.5/0.5/0.6`，当前 `upstream` 副本的 runner 直接传递轴值；调用端还可能先做转换。必须追踪完整调用链，避免重复缩放或遗漏，不能仅看变量名 `*_scale`。Go2W 的命令范围也不等于 heading 模式下最终 yaw-rate 的实际范围，应检查回调中的限幅。

### 2.3 初始化怎么处理

- 完全从零：新建 actor、critic、估计器、优化器、探索分布和计数器，最接近此仓库原流程。
- 官方 actor 热启动：技术上可行，但需要做关节排列、命令缩放、默认姿态/动作语义等适配，并初始化新增输入和估计器；必须验证初始动作数值一致。不能只因输入维度相同就复制权重。
- 旧 TurnStop 或其他 S10 完整 checkpoint 不能直接恢复为 HIM。官方 ONNX 也只有 actor，没有 HIM critic、估计器或优化器。

准备阶段不自动实现两套初始化或开展对照训练。正式启动前依据用户选定方案执行。现有训练入口还硬编码 `args.resume=False`，如果后续需要中断恢复，要核对参数传递与保存/加载行为，不得把“恢复失败后从零重跑”当作续训。

## 3. 原仓库接口问题：先核实意图，再决定修正

前面对话指出的是静态代码疑点，不能把它们整体当成已完成诊断的 bug 清单。对每项给出：当前实际行为、是否有原设计依据、训练与部署是否一致、处理决定及最小验证。查相关调用方、仓库历史和必要的上游说明；没有找到设计说明时如实记录，不无限追溯。

### 3.1 轮速奖励原地修改状态

已看到 [LeggedRobot](D:/Desktop/Code/HIMLoco-for-Go2W/legged_gym/envs/base/legged_robot.py) 中：

```python
def _reward_dof_vel(self):
    self.dof_vel[:, self.wheel_indices] = 0
    return torch.sum(torch.square(self.dof_vel), dim=1)
```

`dof_vel` 是 Gym 状态张量的 view；当前配置该奖励非零，`post_physics_step()` 先算奖励再组装观测，之后还保存 `last_dof_vel`。静态调用顺序显示轮速值会被清零并影响后续读取，不能把这段理解为仅对奖励的临时屏蔽。

- **可能的合理意图**：不惩罚轮子正常转动，这不等于需要惩罚轮速，也不等于必须改奖励目标。
- **需要确认的另一层意图**：是否有意让策略不读轮速？目前没有看到说明；现有 [MuJoCo runner](D:/Desktop/Code/HIMLoco-for-Go2W/mujoco/pdandrl.py) 读取真实轮速，存在输入不一致的线索。
- **验证**：非零轮速状态下记录奖励调用前后状态、下一帧观测与缓存；关闭噪声以分清真实信号与噪声，不需要 PPO。
- **处理原则**：若仅为排除轮速惩罚，用局部选择/副本计算，保留原奖励数学含义；若确有有意屏蔽观测的依据，将其显式放在观测组装里并对齐部署，不靠奖励副作用实现。改变输入语义会影响旧 Go2W checkpoint，明确其适用版本。

### 3.2 轮角观测置零与原始状态写入

`compute_observations()`、`get_current_obs()`、`compute_termination_observations()` 中都存在轮位置写零。连续轮的绝对转角不进入观测可以是合理设计，S10 现有 actor 也这样处理；不要据此恢复不需要的轮角输入。

核对“只对观测误差切片置零”和“直接修改 `self.dof_pos` 状态 view”的区别。若后者无必要且影响其他读取，保留观测置零语义，仅消除状态副作用；同时覆盖上述三个调用路径。

### 3.3 回合结束后的6帧历史

`reset_idx()` 清理部分动作/速度缓存，但未看到清理 `obs_buf`；随后 `compute_observations()` 把新帧接在旧历史前。静态上存在上一回合历史进入新回合的情况。

- 核对上游是否刻意这样训练、实际是否发生了物理状态重置，以及部署进入 RL 状态时怎样初始化历史。
- 区分终止状态用于估计器训练/价值处理，与 reset 后供新动作使用的历史。当前 HIM runner 专门保存 reset 前的 `termination_privileged_obs`，不要在修历史时破坏这条路径。
- 若实际重置造成不连续且没有匹配的设计依据，明确采用零填充还是首帧填充，并统一训练、play、导出使用端；正常未重置环境的历史继续滚动。
- 最小验证只需两个环境，其中一个 reset，检查其新历史、另一个连续历史、上一动作和终止观测是否符合约定。

### 3.4 写死的 Go2W 索引

`_reward_hip_default()` 使用 `[0,4,8,12]`。如果 S10 实际导入后这四项仍准确对应 hipx，这本身不构成必须重构的 bug；如果不同，改为核验过的 S10 关节索引。类似检查轮索引、接触 body 名称和 MuJoCo 关节映射即可，不做无关的全仓库重构。

**完成标准：有明确依据的设计予以保留；必要修正保持原奖励/算法意图，并留下一个可运行的最小接口检查。不能用“源码里一直这么写”证明正确，也不能用“看起来奇怪”直接更改训练目标。**

## 4. 现有服务器连接方法

从本机 PowerShell 使用已有 SSH 密钥别名：

```powershell
ssh paratera-rl-new
ssh paratera-rl-new "nvidia-smi"
```

别名保存在本机 `C:\Users\Lenovo\.ssh\config`。只使用别名；公网地址、密码、私钥内容不得写入仓库、日志或聊天。不要连接已停用的 `paratera-isaac`，不要使用旧密码文件或连接脚本。

最近已记录的服务器配置（不是本次重新实测）：

| 项目 | 已有记录 |
|---|---|
| 系统/GPU | Ubuntu 22.04.5；单张 RTX 4090 24GB |
| CPU/内存 | 10 vCPU；32GB RAM |
| 磁盘 | 50GB系统盘；150GB数据盘挂载 `/data` |
| 驱动 | 580.173.02 |
| 容器 | Docker 29.1.3；Compose 2.40.3；NVIDIA Container Toolkit 1.20.0 |
| 容器数据 | Docker `/data/docker`；containerd `/data/containerd` |
| 已有工程 | `/data/rl_training`；`/home/ubuntu/rl_training` 是它的符号链接 |

新对话连接后先做当前状态检查，尤其磁盘可用量和其他训练进程：

```powershell
ssh paratera-rl-new "cat /etc/os-release"
ssh paratera-rl-new "nvidia-smi"
ssh paratera-rl-new "df -h / /data"
ssh paratera-rl-new "docker version"
ssh paratera-rl-new "docker compose version"
ssh paratera-rl-new "docker ps"
```

建议 HIM 使用独立目录 `/data/HIMLoco-for-Go2W` 和独立镜像；这是建议路径，尚未创建或检查，不得把它写成已部署。已有 Isaac Lab 工程不能当作 HIM 工作目录；不要停止其他实验或覆盖其依赖。

## 5. 服务器需要安装什么

复用现有主机驱动、Docker 和 NVIDIA Container Toolkit，在独立容器内准备旧 Gym 软件栈。不要把两套同名 `rsl_rl` 装进当前 Isaac Lab 环境。

| 组件 | 用途与版本状态 |
|---|---|
| Linux用户态 + Python 3.8 | 建议候选基础，需与 Gym 二进制匹配 |
| PyTorch 2.0.1 + CUDA 11.8、匹配 torchvision | 针对4090的候选组合，尚未在这套 HIM 环境验证；不是上游公布的原测试组合 |
| Isaac Gym Preview 4 | 从 NVIDIA 官方归档获取，安装 Python 包；不是 Isaac Lab，也不能仅用一个同名 PyPI 包代替 |
| 本仓库 `rsl_rl` | 安装包含 HIM 修改的本地版本，不能替换成当前最新版 RSL-RL |
| 本仓库根目录包 | 此 Go2W fork 的根目录有 `setup.py`，需按它的目录结构安装 `legged_gym`，不能机械照抄上游不同层级的路径 |
| NumPy、SciPy、Matplotlib、TensorBoard | 按旧代码/Python约束固定兼容版本，必要时处理废弃API；不要整体升级到最新版再排错 |
| GCC/G++、Ninja与必要系统运行库 | 编译/加载 Gym PyTorch 扩展；按实际错误补必要依赖 |
| MuJoCo、PyYAML、pynput | 后续运行现有 MuJoCo 验证程序时使用，可放单独验证环境；无头训练不依赖它们 |

PyTorch 的 CUDA 运行时、Gym 二进制与主机驱动是不同层。不要看到版本号不同就降级主机驱动或改系统 CUDA；只在具体依赖需要时给独立容器补 Toolkit。保存最终实测可用的 Python、PyTorch/CUDA、Gym 和依赖版本。

服务器训练不需要 S10 SDK、ROS 2、AIRY 驱动或 AGX 的部署环境。Isaac Gym 已停止官方支持；兼容问题以实际导入、GPU张量计算、Gym物理步进和HIM推理检查定位，环境没有通过前不启动正式训练。

参考：[HIMLoco上游安装说明](https://github.com/InternRobotics/HIMLoco#installation)、[NVIDIA Gym归档入口](https://developer.nvidia.com/isaac-gym)、[PyTorch历史版本](https://docs.pytorch.org/get-started/previous-versions/#v201)。上游公布的测试组合是 Python 3.7.16/PyTorch 1.10.0+cu113/Isaac Gym Preview 4，与上述候选组合不同。

## 6. S10 模型和控制参数从 goai26-s10-racing 获取并检验

本机仓库：`D:\Desktop\Code\goai26-s10-racing`。开始前读取该目录中实际存在的协作说明。

### 6.1 必查来源

| 来源 | 文件 |
|---|---|
| 官方资料副本：模型 | [S10.urdf](D:/Desktop/Code/goai26-s10-racing/goai_embodied_future_material-main/goai_embodied_future_material-main/src/S10_sdk_deploy/S10_description/s10_mjcf/urdf/S10.urdf)、[S10.xml](D:/Desktop/Code/goai26-s10-racing/goai_embodied_future_material-main/goai_embodied_future_material-main/src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10.xml)及其相对网格资源 |
| 官方资料副本：策略控制 | [s10_policy_runner.hpp](D:/Desktop/Code/goai26-s10-racing/goai_embodied_future_material-main/goai_embodied_future_material-main/src/S10_sdk_deploy/run_policy/s10_policy_runner.hpp) |
| 当前维护副本：模型 | [S10.urdf](D:/Desktop/Code/goai26-s10-racing/upstream/goai_embodied_future_material/src/S10_sdk_deploy/S10_description/s10_mjcf/urdf/S10.urdf)、[S10.xml](D:/Desktop/Code/goai26-s10-racing/upstream/goai_embodied_future_material/src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10.xml) |
| 当前维护副本：策略控制 | [s10_policy_runner.hpp](D:/Desktop/Code/goai26-s10-racing/upstream/goai_embodied_future_material/src/S10_sdk_deploy/run_policy/s10_policy_runner.hpp)，包括默认姿态、增益、重排、命令及输出解码 |
| 实机接口 | [s10_interface.hpp](D:/Desktop/Code/goai26-s10-racing/upstream/goai_embodied_future_material/src/S10_sdk_deploy/interface/robot/hardware/s10_interface.hpp)，核对方向、偏置、轮模式和SDK命令字段 |
| MuJoCo控制实现 | [mujoco_simulation_ros2.py](D:/Desktop/Code/goai26-s10-racing/upstream/goai_embodied_future_material/src/S10_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py) |
| 部署与实测记录 | [实机指南](D:/Desktop/Code/goai26-s10-racing/docs/S10_REAL_ROBOT_QUICKSTART_ZH.md)、[48号测试记录](D:/Desktop/Code/goai26-s10-racing/docs/S10_SPEEDTURN_2000_TRIAL_ZH.md) |

不要默认赛道版 `S10_track.xml` 与基础 `S10.xml` 相同；核对实际模型的 include、碰撞和坐标调整。也不要将旧步态/站起状态的增益或 `s10_control_parameters.cpp` 中其他控制器参数混作 RL 状态的参数。

### 6.2 对照资料

- [实机硬件与传感器约束](D:/Desktop/Code/rl_training/docs/finals/s10_hardware_and_sensors.md)：涉及实机观测或部署时先读。
- [当前 S10 接口契约](D:/Desktop/Code/rl_training/docs/finals/s10_interface_contract.md)：用来解释已有57D/16D及历史174D接口，不能替代本次源码核验。
- [现有 S10 资产配置](D:/Desktop/Code/rl_training/source/rl_training/rl_training/assets/deeprobotics.py)、[官方 model0 说明](D:/Desktop/Code/rl_training/pretrained/s10/README.md)：对照当前训练参数和actor来源。

S10 已确认前后双 AIRY、AGX/ARM、ROS 2 Jazzy 等硬件信息，但 HIM actor 首阶段只用本体观测历史。仿真高度、外扰和接触可以留在训练侧 critic/监督中，不能作为实机已有传感器直接加入 actor；HIM也不等于提前看见未接触的台阶。

核验后记录一个简明的“来源 → 最终值 → 训练/部署对应位置”表，至少含默认姿态、轮径、关节顺序/方向、PD、限幅、动作和命令缩放、时序。记录实际采用的源文件和版本；复制所需模型与网格时保持相对引用，避免依赖本机绝对路径才能在服务器加载。

## 7. 新对话执行顺序与交付

### 阶段一：运行原有环境

1. 查看本仓库当前代码及工作区状态，确认本交接的静态证据仍适用。
2. 检查服务器，建立独立 HIM 环境，核验实际加载的是本仓库 `rsl_rl` 和 `legged_gym`。
3. 先做小规模 Go2W 环境导入、GPU步进、HIM推理和终止/重置检查，PPO更新为0。若使用已有Go2W checkpoint，记录其来源及与接口版本是否匹配；没有可用checkpoint也不需要先重训Go2W来证明环境能步进。
4. 区分依赖/驱动错误、原始接口问题和 S10 尚未适配的问题，不靠同时修改奖励、控制参数来掩盖环境错误。

### 阶段二：必要接口修正与 S10 核对

1. 按第3节逐项判断并作最小修正；复用现有结构，修正共享行为时检查受影响的调用路径。
2. 从第6节获取并核验模型、控制和观测定义，新增独立 S10 任务，保留 Go2W 配置。
3. 检查静态姿态、逐关节方向/索引、腿位置和轮速度目标解码、状态反馈及reset历史。不要求未训练随机策略先通过运动能力验收。
4. 用一致的传感器输入验证训练侧与导出推理侧的观测、raw action、解码目标。MuJoCo核对实际生效步长和PD频率，不只读配置字段：本仓库 `config.yaml` 的 `timestep` 没有被当前脚本应用到模型，实际周期由加载XML的 `m.opt.timestep` 与物理步数决定。
5. 原Gym配置为5ms×4，策略50Hz；现有S10平地曾用2.5ms×8。先明确选定的物理/控制配置与来源，只有必要的数值稳定性证据才调整，不静默从另一个任务继承。策略周期和历史间隔与部署保持一致。

### 阶段三：按 HIM 流程建立独立实验

准备通过后，冻结本次 S10 适配配置，按 `train → play/export → MuJoCo验证` 的流程建立独立 run。正式启动前只补齐尚未确定的初始化方式和PPO预算；已经可执行的准备不必等待这些答案。

- 原Go2W默认是4096环境、48步采样/更新；可作为HIM正式规模的准备目标，需验证S10资产和混合地形下的实际资源。小规模检查不是正式容量上限；资源不足时报告实测原因再决定规模。
- 第一阶段保留HIM估计器、PPO、原地形课程、命令逻辑及适用奖励，仅做有依据的S10适配和必要接口修正。
- 不自动加入旧M20/TurnStop奖励、专家轨迹、BC、额外KL停止规则或两条A/B训练。原配方没有明确保证对角抬腿/长期停稳，不能为了展示这些动作悄悄改成另一套目标。
- 保存可恢复checkpoint、实际配置和环境版本；导出完整估计器+actor，维护342D历史输入及reset语义。当前57D ONNX runner不能直接加载它，部署格式可按选定运行端适配。
- 评价要包含实际三轴速度、关节/轮速、跌倒与接触、命令切换和停车行为；单纯奖励上升不能证明实机需求达标。正式训练及后续评估按新任务明确的范围执行。

准备阶段交付：环境可运行证据、每个接口问题的判断与最小检查、S10参数/来源表、独立任务配置、零PPO检查结果，以及可直接执行的训练入口与尚缺的初始化/预算选择。不得把未执行的GPU检查、MuJoCo验证或训练写成已完成。

## 新对话可直接使用的开场文字

> 请读取本仓库 `docs/S10_HIM_HANDOFF_20260915.md`，按文档准备 S10 接入 HIMLoco-for-Go2W。先完成服务器旧环境运行检查，核实原仓库接口疑点是否属于有意设计并仅修必要项，从 `D:\Desktop\Code\goai26-s10-racing` 获取和核验 S10 模型、关节映射、控制参数及部署语义，再建立独立 S10 任务。先推进不依赖初始化和正式预算选择的准备与零PPO检查，准备完成后汇报证据和剩余选择。未经我同意不用子代理，做A/B测试前先询问。
