# 共享参考形状、三轴伸缩与运动补偿的 3D-GTR

本版本将静态 3D-GTR 扩展为：**一组共享的参考高斯，加上随采集时间变化的整体三轴伸缩，在真实照明／接收几何下预测每次曝光的 transient，并联合反演参考场景和伸缩参数。**

实现日期：2026-10-29。入口为 [deformable_train.py](../deformable_train.py)。

这已经是基线代码的扩展，而不是 math_tests 中独立的单峰 Fisher 实验。动态模型实际调用仓库的 NLOS_3DGS.forward 和 batch_render_nonconfocal，使用相同的几何幅度、时间中心与 Gaussian 时间核。

## 1. 本版本实现了什么

| 内容 | 实现范围 |
| --- | --- |
| 共享参考物体 | 原始 NLOS_3DGS 参数；支持从原版 checkpoint 或参考高斯 NPZ 初始化 |
| 整体形变 | 固定世界坐标轴上的三个正伸缩比例；可选择共享一个比例 |
| 时间参数 | log-scale 时间结点插值，参考结点严格固定为单位变换 |
| 完整高斯变换 | 中心、协方差、精度矩阵与体积因子一同变换 |
| 运动补偿 | 将时变几何直接写入前向算子，支持曝光期间的数值积分 |
| 测量 | 非共焦／合法共焦配置、墙面法向、逐点或固定非负照明编码、剂量和背景 |
| 联合反演 | 可优化参考中心／反射率，或全部参考高斯参数，同时优化伸缩 |
| 验证 | 留出一部分照明图案＋接收位置组合，跨时刻整组留出 |
| 输出 | 可重载动态 checkpoint、任意覆盖时间内的预测和 Gaussian 状态 |
| 运行 | 小规模纯 PyTorch CPU 参考；保留原 CUDA/Triton 时间核接口 |

暂未实现：自由局部形变场、任意旋转／剪切、未知膨胀中心的估计、事件滤波、自适应照明或 SLM 相位设计、首光子动态反演、LCT/RSD 最终体积重建。

**三轴模型严格包含均匀膨胀**：令 ax=ay=az=a 即可。isotropic 模式直接共享参数，不是通过惩罚项近似相等。

## 2. 从参考高斯到当前状态

参考高斯为

$$
\mathcal G^0=\{\mathbf x_j^0,\Sigma_j^0,\rho_j^0\}_{j=1}^N.
$$

固定膨胀中心 c，定义

$$
A(s)=\operatorname{diag}(a_x(s),a_y(s),a_z(s)),\qquad a_i(s)>0.
$$

可选附加平移 b(s)，默认关闭。当前几何为

$$
\boxed{
\mathbf x_j(s)=\mathbf c+A(s)(\mathbf x_j^0-\mathbf c)+\mathbf b(s),
\qquad
\Sigma_j(s)=A(s)\Sigma_j^0A(s)^\top.
}
$$

对全局仿射变换，Gaussian 协方差推送是精确的；这里不需要自由形变场的局部线性近似。

渲染还使用

$$
P_j(s)=A(s)^{-\top}P_j^0A(s)^{-1},
\qquad
v_j(s)=\det A(s)\,v_j^0,
$$

其中 P 是基线用于方向投影的精度矩阵，v 是基线的 sqrt(det Sigma) 因子。

### 与发布版 quaternion 约定的兼容

发布版对行方向使用 u @ R.T。新增 gaussian_state() 因此按相同约定构造

$$
\Sigma^0=R^\top\operatorname{diag}(s_j^2)R,
\qquad P^0=R^\top\operatorname{diag}(s_j^{-2})R.
$$

发布版 quaternion 归一化分母含 1e-6，所以 R 在有限精度下并非严格正交。本版本分别保留基线的协方差、精度和体积因子，不用重新求逆／行列式悄悄改变静态幅度。单位形变的数值回归覆盖了非平凡旋转与各向异性 Gaussian。

反射率 rho_j 暂视为同一个 Gaussian 的共享属性。变形后的幅度由基线几何公式重新计算；这里不额外施加质量守恒或不可压缩假设。真实气球材料拉伸导致的 BRDF／反射率变化尚未验证。

## 3. 时间参数与尺度锚定

每个方向使用

$$
a_i(s)=\exp(\ell_i(s)).
$$

在时间结点之间线性插值 ell，而不是直接插值 a，保证比例始终为正。第一个时间结点固定

$$
\ell_i(s_0)=0,\qquad\mathbf b(s_0)=0.
$$

这样参考形状不会与整体尺度任意互相补偿。运动结点必须覆盖完整曝光区间；程序拒绝未声明的时间外推。

模式对应：

- static：所有时刻单位变换；
- isotropic：每个非参考结点 1 个 log-scale；
- anisotropic：每个非参考结点 3 个 log-scale。

若有 L 个时间结点，三轴伸缩的未知量只有 3(L-1)，与 Gaussian 数量无关。可选平移增加 3(L-1) 个参数。优化阶段将伸缩限制在 [0.5,2]，报告触边数量。

时间平滑项约束相邻区间的 log-scale 变化速度差；支持非等间距结点。已知充气速度未作为额外约束加入这一版。

## 4. 运动补偿的 NLOS 前向算子

在每个时刻，原始渲染器根据当前高斯状态计算：

$$
\mu_{rpj}(s)
=\frac{d_{1,pj}(s)+d_{2,rj}(s)}{c\,\Delta t},
$$

$$
\sigma_{rpj}(s)
=\frac{\sqrt{(\mathbf u_1+\mathbf u_2)^\top\Sigma_j(s)(\mathbf u_1+\mathbf u_2)+\epsilon}}
{c\,\Delta t}.
$$

几何幅度也使用当前距离、方向、Gaussian 投影面积等重新计算。这样伸缩同时影响峰位、宽度和幅度，不能简化成统一移动时间峰。

第 r 行测量使用相机墙点 d_r 和照明权重 w_rp。预测为

$$
\boxed{
\widehat y_{rk}
=m\,Q_r
\sum_p w_{rp}\sum_l\omega_l
h_{p,d_r,k}\bigl(\mathcal G(s_{rl})\bigr)
b_{rk}.
}
$$

其中：

- m：render magnification，基线强度到观测单位的标定比例；
- Q_r：整次曝光的总照明剂量；
- w_rp>=0，sum_p w_rp=1；
- s_rl、omega_l：曝光区间内的归一化 Gauss–Legendre 积分结点和权重；
- b_rk：已知背景。

权重和为 1，所以 Q 已包含曝光总剂量，不能再乘一次曝光长度。默认假设曝光内照明通量均匀；非均匀通量需要扩展时间权重。

零长度区间按瞬时状态处理。有限区间默认 3 点积分；可增加 --quadrature 核验精度。相同曝光区间的测量共享一次形变状态计算；只渲染非零照明权重。

这是一种直接在测量算子中进行的运动补偿，不是先重建图像再做配准，也没有直接把实际数据除以某个时间尺度。

### 与前面高维 lifting 讨论的关系

均匀膨胀在中心化 lifting 坐标 (y,|y|^2) 中对应 diag(a,a,a,a^2)。本版本在物理三维空间显式实施相同的均匀膨胀，以及更一般的三轴伸缩，再计算真实距离和方向。

它没有实现新的高维 Radon 解析逆变换；优化器仍是可微反演。这个取舍先保证实际几何、幅度及原始基线的一致性，再为之后研究投影域快速变换保留比较对象。

## 5. 损失函数与数据单位

默认使用与基线相容的归一化 MSE：

$$
\mathcal L_{\mathrm{data}}
=\frac{\sum_{r,k}(\widehat y_{rk}-y_{rk})^2}
{\sum_{r,k}y_{rk}^2}.
$$

再加入较弱的 log-scale 速度变化惩罚和参考中心初始化锚定项。参考锚定是一项显式先验，不是额外测量。

可选 --loss poisson 使用 Poisson NLL，要求输入为非负原始整数计数，且 magnification/dose/background 已正确标定。整数检查只是防止误用，并不能自动证明物理标定正确。

原始 3D-GTR 的时间核是 bin 中心采样的未归一化 Gaussian，本版本为保持基线一致沿用这个定义。它不自动成为严格的单光子概率质量。真实计数实验还应纳入 IRF、时间箱积分及实际曝光标定。

**强首光子 pile-up 不能直接用这里的 Poisson NLL。** math_tests/first_photon.py 已验证的观测层尚未并入动态训练；跨周期死时间、事件递推滤波同样未实现。

## 6. CPU 合成验证

```powershell
cd "F:\deformable NLOS\nlos-3d-gtr"
python deformable_train.py demo
```

默认使用 2 个 Torch CPU 线程；修改时全局参数放在子命令前，例如：

```powershell
python deformable_train.py --threads 4 demo --iterations 200
```

两种真实运动分别为：

- uniform：最终比例 (1.25,1.25,1.25)；
- anisotropic：最终比例 (1.15,1.25,1.35)。

每种情况生成 5 个时刻、16 个激光扫描点、固定接收点的测量，Gaussian 数为 24。中间三个时刻使用 0.08 s 曝光；合成数据用 7 点积分生成，反演用 3 点积分，避免数值积分完全相同。

默认每种模型训练 200 步。所有时间都应使用相对实验起点的秒数，避免将大数值 Unix 时间戳直接放入 float32 时间参数。

每个场景比较 static、isotropic、anisotropic 三种拟合，共享相同数据、初始化和空间验证划分。默认增加 Poisson 计数噪声，峰值期望信号约 2000 counts/bin，背景为 2 counts/bin；这只是合成标定，不是实验硬件预测。

参考 Gaussian 中心初始化带有 1.5 mm 标准差扰动，反射率初值是真值的 90%。反演同时优化参考中心、反射率与运动参数；参考 Gaussian 尺度、方向和数量在默认 demo 中视为已有先验。

**因此，这是有参考先验的 warm-start 联合反演，不是从任意初始化盲恢复真实气球。** 数据生成与反演共享散射模型，属于第一版模型内验证；它不能证明对模型失配、真实遮挡或复杂局部形变的泛化。

空间留出按“照明图案＋相机位置”分组，同一配置的所有时刻一起留出；不把时间重复观测随机拆分后当成独立泛化证据。留出约 20% 的空间采集配置。

默认结果与实际数值记录于下节。

## 7. 默认运行结果

验证命令为 `python deformable_train.py demo`，种子 20261029，Poisson 噪声，每种模型 200 步。两种场景均通过验收。

| 真实运动 | 静态模型留出 NMSE | 单参数模型留出 NMSE | 三轴模型留出 NMSE |
| --- | ---: | ---: | ---: |
| 均匀膨胀 | 0.231282 | 0.00277239 | 0.00277672 |
| 不等比例膨胀 | 0.297766 | 0.0122823 | 0.00342791 |

| 真实运动 | 真实最终比例 (x,y,z) | 三轴模型估计的最终比例 | 三轴模型全评估时刻最大绝对尺度误差 |
| --- | --- | --- | ---: |
| 均匀膨胀 | (1.25,1.25,1.25) | (1.24810,1.25176,1.24987) | 0.00190306 |
| 不等比例膨胀 | (1.15,1.25,1.35) | (1.16013,1.23959,1.35043) | 0.0148785 |

尺度误差是无量纲比例值，0.0148785 对应约 1.49 个百分点，不是毫米。表中最大误差覆盖全部五个评估时刻，不只看最后一帧。

均匀膨胀时两个动态模型表现接近；不等比例膨胀时，三轴模型明显降低了单参数模型的失配误差。新增自由度在这个合成设置下有价值，但这不构成对一般真实形变的可辨识性证明。

此次 CPU、2 个 Torch 线程运行中，三轴模型每次 200 步训练约 41–43 s；六次对照训练合计约 241 s，不含进程启动和其他验证。这里仍是离线联合优化，尚未实现实时增量更新，也未与原始 GPU 实现进行速度比较。所有伸缩参数均未触及 [0.5,2] 约束边界。

9 项回归测试通过，原版格式和动态 checkpoint 测试通过；另外已运行命令行的 Poisson 损失、全部参考属性＋平移的短训练，以及动态模型重载预测。短训练仅检查这些分支可执行，不代表它们已达到 demo 的反演精度。

输出目录 deformable_gtr/results/demo/ 包含：

- report.json：每种模型的训练／验证误差、尺度误差、时间与验收结果；
- comparison.png：真实与估计尺度曲线，以及空间留出误差；
- uniform/、anisotropic/：各自的 measurements.npz、canonical_init.npz、truth.npz；
- 每种拟合的 .pth checkpoint 和 *_prediction.npz。

![默认合成验证](results/demo/comparison.png)

结果目录已忽略。默认 demo 的验收条件是：三轴模型在两种场景都使留出 NMSE 低于静态模型一半，并且所有评估时刻的伸缩比例绝对误差小于 0.04。这个阈值是工程回归检查，不是普适信息界限。

## 8. 自己的数据格式

新增时间化测量使用 NPZ，allow_pickle=False。必需字段：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| laser_points | (P,3) | 可用激光墙点，单位 m |
| camera_points | (M,3) | 每行测量的接收墙点，单位 m |
| patterns | (M,P) | 非负强度权重，每行和为 1；扫描对应 one-hot |
| transients | (M,K) | 每行观测；预测时也需提供同形状占位数组 |
| times | (M,) | 实验采集时间，单位 s，不是光子飞行时间 |
| dt | 标量 | 时间箱宽度，单位 s |
| t_range | (2,) | 半开区间 [start,end)，K=end-start |

可选字段：

| 字段 | 形状 | 默认值 |
| --- | --- | --- |
| exposure_start/end | (M,) | 两者都缺省时，均等于 times；必须成对提供 |
| dose | (M,) 或标量 | 1，总曝光剂量 |
| background | (M,K)、(K,) 或标量 | 0，观测单位 |
| laser_normals | (P,3) | +z，适用于平面墙 |
| camera_normals | (M,3) | +z，适用于平面墙 |
| render_magnification | 标量 | 1，可由 --magnification 覆盖 |

所有曝光必须包含对应 times，且落在运动结点覆盖范围内。当前照明图案和接收点在一次曝光内保持不变；移动扫描需拆成多行测量。

非平面墙应提供真实法向，不能依赖平面默认值。共焦原版 checkpoint 只接受每个非零照明项都与相机墙点重合的配置；不能用一个固定相机点读取多个激光点，然后把配置声明为共焦。

旧 MAT 没有采集时间时，程序不会伪造时间或将其默认为动态序列。合成序列可用于开发，但不能当作已经采得的形变实验。

## 9. 从参考初始化训练

参考 NPZ 字段：centers=(N,3) m、scales=(N,3) m、quats=(N,4)、albedos=(N,) 或 (N,1)。四元数使用原版的 [w,x,y,z] 约定。

也可以将 --init 指向包含 config/state_dict 的原版 3D-GTR checkpoint。本版本直接读取原格式，不需要为 CPU 安装 EasyDict 或 Triton；不支持缺失完整配置的裸权重。

原版格式的兼容性已用构造的 checkpoint 回归验证。当前工作副本 real_output 下的 .pth 为 Git LFS 指针，尚未用其中的实际发布权重进行加载／动态拟合实验；使用前需要具备实际权重文件。

以下示例使用 demo 已生成的 NPZ，其中已记录合成标定系数：

```powershell
python deformable_train.py fit --data deformable_gtr/results/demo/anisotropic/measurements.npz --init deformable_gtr/results/demo/anisotropic/canonical_init.npz --pivot 0.025 -0.015 0.72 --knots 0 0.5 1 --mode anisotropic --output deformable_gtr/results/my_fit
```

--mode isotropic 可强制均匀膨胀。--fit-canonical none 固定参考场景，只估计运动；默认 centers_albedo 优化参考中心与反射率；all 还优化参考尺度与方向，但自由度更高，需重新验证可辨识性与稳定性。

--translation 额外拟合平移；它不是膨胀中心的自动估计，也可能与伸缩产生耦合，默认 demo 关闭。--batch-size 可限制每步测量行数。--validation 0 关闭空间验证；默认保留 20%。

GPU 使用 --device cuda:0，--backend auto 将使用原始 Triton 时间核；--backend torch 可选择 Torch 实现。**当前环境仅有 CPU，CUDA/Triton 的动态路径尚未实际运行验证。**

## 10. 重载模型和预测

```powershell
python deformable_train.py render --checkpoint deformable_gtr/results/demo/anisotropic/anisotropic.pth --data deformable_gtr/results/demo/anisotropic/measurements.npz --output deformable_gtr/results/reloaded_prediction.npz
```

预测数据可以包含新的照明点和时间，但时间需被原运动结点覆盖，dt/t_range 必须与 checkpoint 一致。

程序接口还可直接调用 model.motion.push_forward(model.canonical.gaussian_state(), time)，取得该时刻完整的 Gaussian 状态，供之后的虚拟墙渲染和动态 LCT/RSD 接口使用。原 output.py 仍面向静态 checkpoint，不能直接读取新的动态 checkpoint。

## 11. 数值回归与文件入口

```powershell
python -m unittest discover -s deformable_gtr/tests -v
```

9 项测试覆盖：

1. 单位变换与原版静态公式一致，包含非平凡 quaternion 和非共焦墙点；
2. isotropic 与三个相同参数的 anisotropic 在任意测试时刻一致；
3. 协方差、精度与体积因子的仿射变换；
4. 通过实际渲染器的三个运动参数解析梯度与有限差分；
5. 有限曝光积分及静态极限；
6. 固定强度图案的线性叠加；
7. 动态 checkpoint 存取一致；
8. 原版格式 checkpoint 初始化；
9. 非法图案、缺失时间和时间外推被拒绝。

| 文件 | 入口 |
| --- | --- |
| [model.py](model.py) | AxisScaling、DeformableGTR、参考初始化 |
| [data.py](data.py) | Measurements 数据检查和曝光分组 |
| [training.py](training.py) | 联合训练、留出验证、存取与导出 |
| [demo.py](demo.py) | 合成序列与三种模型对照 |
| [test_deformable.py](tests/test_deformable.py) | 物理和梯度回归 |
| [nlos_3dgs.py](../nlos_3dgs.py) | 基线新增状态接口和可选 CPU 时间核 |

原始静态参数名、checkpoint 格式、默认 CUDA 时间核和静态训练入口均保留。动态结果没有自动注册为原版静态 checkpoint。

本版本的下一步证据应来自：更差初始化、独立生成模型、局部非仿射形变、真实 IRF/计数标定及真实气球数据。当前实现完成的是整体伸缩模型与基线前向算子的接入和模型内验证。
