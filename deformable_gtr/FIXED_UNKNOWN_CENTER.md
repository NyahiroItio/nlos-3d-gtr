# 固定未知球心：测量域配准与不透光椭球可行性实验

本次尝试包含两个独立层次：

1. 为原有整体 Gaussian 伸缩接口增加一个跨全部时刻共享、可以优化的固定 pivot。
2. 用**有明确表面和单侧可见性的不透光凸椭球**检验固定未知球心的估计，避免继续利用旧 demo 中不应出现的背面返回。

第二层是带椭球形状先验的局部可行性实验，不代表已经完成任意 Gaussian 物体的自遮挡，也不能将旧 Gaussian demo 的数值重新解释为物理正确。

## 1. 它与点云配准有什么关系？

普通点云配准可以使用类似目标：

$$
\min_\theta\sum_i\|T_\theta(\mathbf x_i)-\mathbf y_{\pi(i)}\|^2,
$$

其中两组点的位置来自几何观测，pi 是给定或估计的对应关系。

这里没有观测到第二组隐藏物体点云。优化目标是

$$
\boxed{\min_\theta\mathcal L\bigl(\mathcal F[T_\theta(S_0)],y\bigr),}
$$

其中 F 是 NLOS 光输运和仪器响应，y 是瞬态计数。更准确的名称是**测量域／投影域的形变参数估计，或基于物理前向模型的配准**。

物体有参数化的对应点，不等于 SPAD 给出了点对应。程序没有 ICP、最近邻或 Chamfer 损失，也没有先将每帧重建成点云再对齐。

## 2. 固定未知的 pivot 在通用 Gaussian 模型中怎样进入优化？

原有模型为

$$
\mathbf x_j(s)=\mathbf c+A(s)(\mathbf x_j^0-\mathbf c).
$$

新增 --learn-pivot 时，c 是 nn.Parameter，而不是固定 buffer。三个坐标跨时刻共享，使用以米为单位的单独学习率。默认仍保持固定 pivot，兼容原有调用与 checkpoint。

对于独立于 c 的参考 Gaussian 位置，

$$
\frac{\partial\mathbf x_j(s)}{\partial\mathbf c}=I-A(s).
$$

所以参考时刻 A=I 时，运动关系本身不能约束 pivot；形变小也会使中心估计变弱。新增测试检查了实际渲染器对此参数的有限差分梯度。

--learn-pivot 拒绝同时启用 --translation，也拒绝 static 模式，防止将一个存在已知参数歧义的设置当成可辨识的运动中心估计。

运行示例（仅检查通用接口，旧数据仍有无自遮挡的限制）：

```powershell
python deformable_train.py fit --data deformable_gtr/results/demo/uniform/measurements.npz --init deformable_gtr/results/demo/uniform/canonical_init.npz --pivot 0.05 -0.035 0.75 --learn-pivot --mode anisotropic --knots 0 0.5 1 --fit-canonical none --output deformable_gtr/results/unknown_pivot_fit
```

此时 --pivot 是初始猜测，不是已知真值。输出报告和 prediction.npz 中记录最终 pivot。

## 3. 物理可见性实验采用更明确的椭球假设

为避免把没有表面定义的 Gaussian 集合当作不透光物体，新实验令

$$
\mathbf x(\mathbf n,s)=\mathbf c+D(s)\mathbf n,
\quad\|\mathbf n\|=1,
\quad D(s)=\operatorname{diag}(r_{0x}a_x(s),r_{0y}a_y(s),r_{0z}a_z(s)).
$$

c 固定但未知，三个参考半轴 r0 也未知，强度还有一个未知的全局增益。均匀模式共享一个伸缩系数；三轴模式各自独立。

这一模型把参考物体的几何中心和膨胀中心绑定为同一个 c。因此

$$
\frac{\partial\mathbf x(\mathbf n,s)}{\partial\mathbf c}=I.
$$

**这与上一节独立参考点云的 I-A 不同。** 本实验可以同时利用参考形状的位置与后续变化来估计中心，包含较强的椭球先验；它没有证明仅从任意参考 Gaussian 的运动就能获得相同精度。

伸缩使用 a(s)=exp(beta*s)，参考时刻 a(0)=1。时间单位是 s，beta 是 log-scale 的变化率；当前实验不学习任意复杂时间函数。

## 4. 表面法向、面积和可见性

单位球到椭球的仿射映射给出

$$
\mathbf n_{\mathrm{raw}}=D^{-\top}\mathbf n,
\qquad \widehat{\mathbf n}=\frac{\mathbf n_{\mathrm{raw}}}{\|\mathbf n_{\mathrm{raw}}\|},
\qquad
\mathrm dS=\det(D)\|D^{-\top}\mathbf n\|\,\mathrm d\Omega.
$$

方向积分采用 Gauss–Legendre 的 z 结点和周期方位角积分，dOmega=dz*dphi，不把经纬网格错误地当作等面积点。

从表面指向照明墙点和接收墙点的单位方向分别为 ell1、ell2。表面点权重为

$$
w_j\propto\frac{
\Delta S_j
[\widehat{\mathbf n}_j\cdot\boldsymbol\ell_1]_+
[\widehat{\mathbf n}_j\cdot\boldsymbol\ell_2]_+
c_{\mathrm{wall},1}c_{\mathrm{wall},2}}
{d_1^2d_2^2}.
$$

墙面为 z=0，法向 +z，两个墙面余弦分别计算。对不透光凸椭球和外部墙点，朝向该墙点的条件也保证这段路径不会穿过物体内部，因此实现自身可见性；外部障碍物未包含。

任一表面侧背向时，返回严格为零。它不是固定剔除一半结点，而是逐测量点对、逐时刻计算。法向与表面积随伸缩变化。若物体非凸、存在透射或不同 BRDF，这个条件不再构成完整模型。

时间模型使用双程光程和已知 Gaussian IRF 的时间箱积分：

$$
\mu_j=(d_{1j}+d_{2j})/c,
\qquad
\lambda_{rk}=b+M e^g\sum_j w_{rj}
\int_{t_k}^{t_{k+1}}\mathcal N(t;\mu_{rj},\sigma_{\mathrm{IRF}})\,\mathrm dt.
$$

sigma_IRF=60 ps，时间零点已知，g 是待估全局 log-gain。没有通过提供反射率真值直接锁定中心，也没有用原始 Gaussian 体投影幅度再重复乘表面面积。

这里使用理想／低通量 Poisson 计数假设，没有加入早期墙面强返回和首光子删失。实验结果不解决真实 SPAD 的 pile-up 问题。

## 5. 三组对照和预算

真值中心为 (0.025,-0.015,0.72) m。错误初值为 (0.05,-0.035,0.75) m，与真值相距约 43.87 mm。

真值参考半轴为 (0.12,0.10,0.14) m，所有拟合使用 (0.11,0.11,0.12) m 的相同半轴初值。

| 组别 | 中心处理 | 其余待估量 |
| --- | --- | --- |
| known_center | 给定正确中心并固定，作为参考 | 参考半轴、伸缩、全局增益 |
| wrong_fixed_center | 固定错误中心 | 同上 |
| estimated_center | 从同样错误初值开始，优化一个共享中心 | 同上，再加 3 个中心坐标 |

每种真实运动只生成一份数据，三组共用同一数据和划分。全局标定系数固定，不因错误模型接收更少信号而重新归一化。

两种真实运动的最终比例分别为 (1.25,1.25,1.25) 和 (1.15,1.25,1.35)。每组采集 5 个时刻、9 个激光墙点，固定接收墙点为原点，2 个墙点的全部时刻用于留出验证。

每条测量平均约 6000 个潜在目标信号计数，背景 0.2 counts/bin。它是合成采集预算，不是单脉冲通量。测量视为各状态的静态曝光，未加入曝光内运动。

真值表面积分用 32x64=2048 个结点，反演用 16x32=512 个结点，避免数值积分完全相同。但两者仍共享椭球、Lambert 反射、IRF 和无外部遮挡假设，属于模型家族内的局部实验。

## 6. 运行和结果

```powershell
cd "F:\deformable NLOS\nlos-3d-gtr"
python deformable_train.py pivot-demo
```

默认固定种子 20261029、Poisson 噪声、每组 200 次 Adam 更新，并降低后期学习率。损失为测量域归一化 MSE，实际从原始 transient 进行梯度更新。--noise none 可运行无噪声对照，--iterations 可更改迭代预算。

默认命令已运行完成，种子 20261029、每组 200 次更新。球心误差为三维欧氏距离，单位 mm。

| 真实运动 | 中心处理 | 球心误差/mm | 空间留出 transient NMSE |
| --- | --- | ---: | ---: |
| 均匀膨胀 | 给定正确中心 | 约 0 | 0.00378572 |
| 均匀膨胀 | 错误中心固定 | 43.8748 | 0.0111486 |
| 均匀膨胀 | 错误初值，中心可优化 | 1.54766 | 0.00377726 |
| 三轴膨胀 | 给定正确中心 | 约 0 | 0.00403675 |
| 三轴膨胀 | 错误中心固定 | 43.8748 | 0.0105209 |
| 三轴膨胀 | 错误初值，中心可优化 | 1.57962 | 0.00405247 |

球心优化后分别为 (0.0246486,-0.0150479,0.718494) m 和 (0.0251516,-0.0151600,0.718436) m，主要残差在深度方向。

形变参数并没有全部同样准确：

| 真实运动 | 真实最终伸缩 | 中心可优化组的估计 |
| --- | --- | --- |
| 均匀膨胀 | (1.25,1.25,1.25) | (1.25267,1.25267,1.25267) |
| 三轴膨胀 | (1.15,1.25,1.35) | (1.20724,1.18275,1.35332) |

三轴场景最大伸缩比例绝对误差为 0.06725，约 6.7 个百分点。参考半轴也有约毫米至近厘米的误差；即使给定正确中心，参考半轴和部分三轴比例仍有明显偏差。因此当前结果不能声称完整形状或全部方向的形变已被可靠恢复。

这一现象可能包含有限迭代、测量灵敏度／参数耦合、计数噪声及积分近似的共同影响，不能仅凭本次拟合判定存在严格不可辨识性。在真实参数处单独计算的粗／细积分 NMSE 分别为 0.00034713、0.00053175，说明积分误差非零，尤其不能忽略其对弱参数的影响。

两种场景均验证：所有背向相机的表面积分点预测信号贡献严格为 0。平均可见积分结点比例约 35%，它不是表面面积比例，也不是固定的半球截断。

14 项回归检查通过；通用 Gaussian 的 --learn-pivot 命令行短训练也通过。CPU、2 个 Torch 线程下，六组正式拟合合计约 462 s；这是离线局部反演，不是实时性能结果。

输出目录 results/pivot_demo/ 包含 report.json、comparison.png、每种场景的测量、三组预测和参数 checkpoint。这个不透光模型的 checkpoint 与通用 Gaussian 动态 checkpoint 使用不同格式；不应交给原 render 子命令混用。

![未知固定球心对照](results/pivot_demo/comparison.png)

误差要分开读：中心误差小不保证全部半轴／伸缩准确；数据误差小也不保证参数唯一。尤其是只观测前侧、增益未知和参考尺寸未知时，深度、曲率和增长率可能耦合。

这里只测试一个给定偏移初值和一次随机计数实现，不是全局收敛保证，也不是 Monte Carlo 置信度。背侧完整几何由椭球先验外推，未被独立观测。

## 7. 回归检查与入口

```powershell
python -m unittest discover -s deformable_gtr/tests -v
```

当前 14 项检查包括旧的 9 项，以及未知 pivot 的实际渲染梯度和存取、中心／自由平移歧义的拒绝、球面面积积分、双墙点相关的可见性、背侧零贡献和不透光表面模型的球心梯度。

| 文件 | 内容 |
| --- | --- |
| [model.py](model.py) | 通用 AxisScaling 的 learn_pivot |
| [training.py](training.py) | 独立球心学习率、中心报告与输出 |
| [opaque_ellipsoid.py](opaque_ellipsoid.py) | 解析不透光表面前向模型 |
| [pivot_demo.py](pivot_demo.py) | 三组控制实验 |
| [test_opaque.py](tests/test_opaque.py) | 表面面积、可见性和梯度检查 |

**结论边界：本次验证的是椭球形状先验下，从瞬态测量估计固定中心和整体伸缩的可能性。它不是点云 ICP，也没有完成任意不透光 Gaussian 场景的物理修复。**
