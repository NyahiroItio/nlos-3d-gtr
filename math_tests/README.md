# 从代码理解照明编码的 Fisher 信息上限

另一个独立实验已实现发射端 phase-only 整形、隐藏目标照明干涉和强度瞬态接收，见 [照明波前验证及完整推导](ILLUMINATION_WAVEFRONT.md)。它与本页的非相干墙面强度编码使用不同的照明模型，请分别阅读。

新增 [首光子探测与 Poisson 模型对照实验](FIRST_PHOTON.md)：比较低通量近似、早期强返回造成的信息损失，以及错误似然的几何估计偏差。本页与波前实验使用理想 Poisson 接收；首光子实验明确增加探测器观测层。

这是一份独立 CPU 数学实验。主程序不导入原有 3D-GTR，不需要 CUDA、Triton 或真实数据。
核心函数有中文注释，按“测量 -> 导数 -> Fisher 信息 -> 信息损失 -> 参数估计”的顺序阅读即可。

本文分为两部分：前半部分解释已运行的简化代码；后半部分给出更一般的完整数学推导。
第一次阅读可先看代码说明，再读 [完整数学推导](#完整数学推导)。
后半部分涉及的动态模型、最优编码与相干波前模型是理论扩展，不代表当前代码已经实现。

## 运行

在 PowerShell 中：

```powershell
cd "F:\deformable NLOS\nlos-3d-gtr"
python math_tests/fisher_coding.py
```

当前本地 Python 已有 NumPy、SciPy、Matplotlib。其他环境缺依赖时：

```powershell
python -m pip install -r math_tests/requirements.txt
```

默认做每种设计 100 次 Monte Carlo 重复。结果存入 `math_tests/results/`：

- `report.json`：数值断言、Fisher 矩阵、CRLB、估计偏差和随机实验结果；
- `information_bound.png`：编码保留的信息比例与位置估计误差下界。

该结果目录已在本目录的 `.gitignore` 中排除。每次运行覆盖同名结果；要保留多个实验可指定目录：

```powershell
python math_tests/fisher_coding.py --trials 500 --dose 2000000 --seed 20261026 --output math_tests/results/run_500
```

只跑数值、不生成图片时增加 `--no-plots`。`--dose` 是统一的入射剂量单位，不代表已标定的实验激光光子数。

## 第一件事：明确实际测量的是什么

隐藏散射体在二维截面中的中心为 `(x,z)`。墙位于 `z=0`，9 个照明位置沿墙分布，探测点固定在墙中心。
未知参数是：

\[
\theta=(x_{\mathrm{cm}},z_{\mathrm{cm}},\eta),\qquad \eta=\log(g).
\]

`g` 是未知的全局散射强度。使用 `log(g)` 可以保证强度始终为正。
前两个参数的单位是 cm，所以最终位置 CRLB 的单位也是 cm。

这是固定宽度的 Gaussian 时间核，模拟一个有限空间 Gaussian 的局部瞬态响应。
它与 3D-GTR 共享由光程决定时间中心的思想，但没有实现完整空间协方差、投影面积、遮挡和形变。

照明位置 `p` 对应两段距离：

\[
d_{1p}=\sqrt{(x-p)^2+z^2},\qquad d_2=\sqrt{x^2+z^2}.
\]

时间中心与幅度取为：

\[
\mu_p=(d_{1p}+d_2)/c,\qquad
A_p=10^{-3}e^\eta\frac{z^2}{d_{1p}^3d_2^3}.
\]

幅度包含两个墙面余弦和逆平方距离衰减。这里用固定效率常数控制检测光子量级，数值不是实验系统标定结果。
时间宽度固定为 60 ps，bin 宽度为 16 ps。bin 接收到的光子不能直接用一个概率密度值表示，需要积分：

\[
q_{pk}=\Phi\left(\frac{t_{k+1}-\mu_p}{\sigma}\right)
-\Phi\left(\frac{t_k-\mu_p}{\sigma}\right),\qquad h_{pk}=A_pq_{pk}.
\]

`h[p,k]` 表示单位入射剂量产生的该 bin 的期望计数。

## 第二件事：计算几何导数

`transient_and_jacobian()` 同时返回 `h` 和 `jac`。其中

\[
J_{pka}=\partial h_{pk}/\partial\theta_a.
\]

代码使用解析链式法则：

\[
\partial_a h_{pk}
=h_{pk}\partial_a\log A_p
+A_p\frac{\partial q_{pk}}{\partial\mu_p}\partial_a\mu_p.
\]

第一项表示几何位置改变造成强度变化；第二项表示光程改变造成时间峰移动。
宽度固定，因此这里没有宽度导数项。

\[
\frac{\partial q}{\partial\mu}
=\frac{\phi((t_k-\mu)/\sigma)-\phi((t_{k+1}-\mu)/\sigma)}{\sigma}.
\]

几何光程导数在代码中直接展开。相对于“米”的导数乘 0.01，才能得到相对于“厘米”的导数。
程序用中心有限差分独立检查这三个解析导数，避免把错误 Jacobian 带入 Fisher 信息。

## 第三件事：Fisher 信息为什么有除以 lambda

独立计数满足：

\[
y_i\sim\operatorname{Poisson}(\lambda_i(\theta)).
\]

省去常数项，对数似然是：

\[
\ell=\sum_i[y_i\log\lambda_i-\lambda_i].
\]

求导：

\[
\nabla\ell=\sum_i(y_i/\lambda_i-1)\nabla\lambda_i.
\]

由于 Poisson 方差等于均值，且通道独立：

\[
F=\mathbb E[\nabla\ell\nabla\ell^T]
=\sum_i\frac{(\nabla\lambda_i)(\nabla\lambda_i)^T}{\lambda_i}.
\]

`fisher()` 把每行 Jacobian 除以 `sqrt(lambda)`，再计算矩阵内积。
所以普通导数变大不等于信息一定变大：Poisson 噪声方差也会随均值增加。

## 第四件事：编码与公平参照

`w[r,p]` 是第 r 次曝光分配给第 p 个墙面点的非负剂量。
在强度非相干叠加条件下：

\[
\lambda_{rk}=\sum_pw_{rp}h_{pk},\qquad
\nabla\lambda_{rk}=\sum_pw_{rp}J_{pk}.
\]

这就是 `encode()`。同一曝光中，探测器只知道光子到达时间，不知道它来自哪个照明位置。

公平的逐点参照给每个点相同总剂量：

\[
e_p=\sum_rw_{rp},\qquad \lambda^{\mathrm{sep}}_{pk}=e_ph_{pk}.
\]

这就是 `separated_fisher()`。这样既匹配总入射剂量，也匹配各点剂量和总期望检测信号光子数。
完整扫描用 9 次曝光，两幅编码用 2 次曝光；本实验没有把曝光次数等同于采集时间，也没有计入切换开销。
场景被假设为静态，因此它们观察的是相同参数。

## 第五件事：信息上限与损失从哪里来

固定 r,k，定义：

\[
a_p=w_{rp}h_{pk},\quad s_p=J_{pk}/h_{pk}=\nabla\log h_{pk},\quad
\Lambda=\sum_pa_p,\quad\bar s=\sum_pa_ps_p/\Lambda.
\]

保留来源标签时的信息贡献是 `sum a_p s_p s_p^T`；混合以后的信息贡献是 `Lambda*s_bar*s_bar^T`。
两者相减、展开后恰好得到：

\[
\boxed{F_{\mathrm{sep}}-F_{\mathrm{code}}
=\sum_{r,k,p}a_p(s_p-\bar s)(s_p-\bar s)^T\succeq0.}
\]

因为每项都是非负系数乘向量外积，所以差矩阵必为半正定。
对任意参数方向 v 都有 `v^T F_code v <= v^T F_sep v`。
`information_loss()` 直接按右侧计算，并与两个 Fisher 矩阵相减独立比较。

这个恒等式同时解释了取等条件：被混合的通道必须有相同的 score。
程序还检查两个容易手算的例子：

- 导数抵消：`h1=1+theta,h2=1-theta`，在 theta=0 时分开观测 F=2；只看和时 F=0。
- 相同 score：`h=[1,2],J=[1,2]`，两通道 score 都为 1，混合前后 F=3。

第一个例子是完全不可辨识；第二个例子说明压缩并不总是损失任务信息。

## 第六件事：未知强度为什么降低几何信息

分块写成：

\[
F=\begin{bmatrix}F_{gg}&F_{gn}\\F_{ng}&F_{nn}\end{bmatrix}.
\]

g 表示两项几何参数，n 表示未知 log_gain。
已知强度时用 `F_gg`，未知时用：

\[
F_{g|n}=F_{gg}-F_{gn}F_{nn}^{-1}F_{ng}.
\]

`effective_geometry()` 实现该公式。
若几何变化造成的强度变化可以完全由 gain 解释，这部分几何信息会被减掉。
只有 `F_eff` 正定且数值可辨识时，才求逆得到标准差下界：

\[
\operatorname{std}(\hat x)\ge\sqrt{[F_{g|n}^{-1}]_{xx}}.
\]

程序不会用一个 ridge 或伪逆把不可辨识方向显示为有限方差。
CRLB 是局部无偏估计在模型正确与常规正则条件下的界；它不保证任意估计器、任意样本量都达到该界。

## 如何读图

两幅编码把墙面分成左右两组。alpha 从 0.5 到 1，图案从完全相同逐渐变为左右分离。
注意 alpha=1 仍混合各组内部多个点，不是 one-hot 扫描。
alpha=0.5 时虽然失去了左右编码差异，时间轴仍有很多 bin，所以不能据此断言所有几何参数一定不可辨识。

左图使用归一化矩阵：

\[
R=F_{\mathrm{sep,eff}}^{-1/2}F_{\mathrm{code,eff}}F_{\mathrm{sep,eff}}^{-1/2}.
\]

其最小特征值表示最不利几何方向保留的信息比例。
这是广义特征值比例，比直接比较混合单位下的普通最小特征值更有意义。
数学上应在 `[0,1]` 内，极小浮点误差除外。
右图为两项位置标准差的 CRLB；越小越好。两幅编码曲线可以接近但不能优于相同剂量的完整参照。

## Monte Carlo 做了什么

对真值生成两种设计下的期望计数，再分别：

1. 随机生成 Poisson 观测；
2. 在真值附近随机扰动初始参数；
3. 最小化 Poisson 负对数似然，同时估计 x,z,gain；
4. 统计标准差、偏差、RMSE、优化失败次数；
5. 与真值处计算的局部 CRLB 对比。

这里的初值有意接近真值，用于验证局部信息分析，不是在证明全局反演容易。
100 次重复的经验标准差有抽样波动，可能小于 CRLB 的数值；这不构成理论反例。
默认运行 200 次拟合全部成功，但程序会如实记录未来实验的失败次数；失败较多时，只分析成功样本会产生选择偏差。

## 第一轮本地结果（seed=20261026，100 次/设计）

两种设计均为约 6573 个期望检测信号光子：

| 设计 | x 的 CRLB 标准差 | x 的经验标准差 | z 的 CRLB 标准差 | z 的经验标准差 |
| --- | ---: | ---: | ---: | ---: |
| 9 点逐点扫描 | 0.06585 cm | 0.06554 cm | 0.01289 cm | 0.01175 cm |
| 两幅编码 alpha=0.8 | 0.14457 cm | 0.14890 cm | 0.02347 cm | 0.02286 cm |

解析 Jacobian 与有限差分的相对误差在 4e-9 以下。
30 组非负随机编码全部通过上限与损失恒等式检查。
损失矩阵最小特征值的最差相对误差约 -3.5e-16，是双精度舍入量级。
结果说明在相同逐点剂量、静态、无背景条件下，丢失照明标签会降低信息。
它没有回答在相同总采集时间、强度限制、背景或动态场景下谁更好。

## 后续扩展顺序

1. 加入半径/膨胀轨迹，让 Gaussian 中心随时间运动；
2. 显式写入每幅图案的曝光区间，建模帧内积分；
3. 加入背景与固定切换耗时，在相同 wall-clock 时间下比较；
4. 在资源约束内优化照明图案；
5. 接入完整 3D-GTR 的协方差、幅度和稠密场景。

本代码验证的是非负强度混合的局部经典 Fisher 信息结果。
它不涉及相干波前整形，也不使用之前“仅峰位移高斯模型”的位置 CRLB 作为通用物理上限。

---

## 完整数学推导

### A. 符号、时间和模型边界

本节从一般 NLOS 模型出发。除非另行说明，Fisher 信息和导数都在同一个局部真值或设计参考点 `theta_0` 处计算。

| 符号 | 含义 |
| --- | --- |
| p=1,...,P | 墙面照明位置索引，对应空间点 `p_p` |
| d=1,...,D | 墙面探测位置索引，对应空间点 `q_d` |
| k=1,...,K | 光子飞行时间 bin 索引 |
| r=1,...,R | 编码曝光索引；不要与气球半径混用 |
| j=1,...,G | 空间 Gaussian 基元索引 |
| tau | 光子飞行时间，单位 s 或 ps，必须与 c 的单位一致 |
| s | 物体状态/扫描时间，通常是 ms 或 s |
| Delta_r | 第 r 幅图案的曝光时长 |
| theta_g | 几何与形变参数，例如中心、半径、膨胀速度 |
| theta_n | nuisance 参数，例如强度、时间零点、背景 |
| theta | 完整参数向量 `[theta_g,theta_n]` |
| h_pdk(theta) | 单位入射剂量下，一个点照明对应的期望信号计数 |
| w_rp | 编码曝光 r 对照明点 p 分配的非负入射剂量 |
| lambda_rdk | 编码曝光的期望总计数 |
| J | 对参数求导得到的 Jacobian |
| F | Fisher 信息矩阵 |

默认假设：非相干强度叠加、线性探测、独立 Poisson 计数、图案已知且在求导时固定。
低通量 SPAD 是该统计模型的适用情形之一；显著 pile-up、dead time 或相关噪声需要改写 likelihood。
参数相关支撑、不可微遮挡事件等情形也不能直接套用常规 CRLB。

本文上限是给定观测模型与资源条件下的局部统计上限，不是所有 NLOS 光学系统的统一极限。

### B. 从 NLOS 光程到 Gaussian 时间核

对隐藏点 x，忽略已经校准/扣除的外部激光器与探测器光程：

\[
L_{pd}(x)=\|x-p_p\|+\|x-q_d\|,\qquad \tau_{pd}(x)=L_{pd}(x)/c.
\]

单个到达时间约束的是一个等光程椭球面。多个照明/探测位置通过不同的光程梯度约束位置。
设第 j 个 Gaussian 的中心与协方差为 `x_j,Sigma_j`，定义：

\[
d_{pj}=\|x_j-p_p\|,\quad d_{dj}=\|x_j-q_d\|,\quad
u_{pj}=\frac{x_j-p_p}{d_{pj}},\quad u_{dj}=\frac{x_j-q_d}{d_{dj}}.
\]

在 Gaussian 空间尺寸远小于传播距离时，作一阶近似：

\[
L_{pd}(X)\approx L_{pd}(x_j)+(u_{pj}+u_{dj})^T(X-x_j).
\]

若 `X ~ N(x_j,Sigma_j)`，线性函数仍是 Gaussian，于是：

\[
\mu_{pdj}=\frac{d_{pj}+d_{dj}}{c},\qquad
\sigma_{pdj}^2=\frac{v_{pdj}^T\Sigma_jv_{pdj}}{c^2},\quad v_{pdj}=u_{pj}+u_{dj}.
\]

这里 mu、sigma 是物理飞行时间。若换成 bin 坐标，要同时除以 bin 宽度 `Delta_tau`。
若系统 IRF 也近似 Gaussian 且已知，卷积后的方差为 `sigma_total^2=sigma_geom^2+sigma_IRF^2`。
一般 IRF 应显式卷积，不一定能只用一个方差表示。

用归一化时间密度 `f` 表示期望计数：

\[
f(\tau;\mu,\sigma)=\frac{1}{\sqrt{2\pi}\sigma}
\exp\left[-\frac{(\tau-\mu)^2}{2\sigma^2}\right],
\]

\[
h_{pdk}(\theta)=\sum_j N_{pdj}(\theta)
\int_{t_k}^{t_{k+1}} f(\tau;\mu_{pdj}(\theta),\sigma_{pdj}(\theta))\,d\tau.
\]

这里 `N_pdj` 是单位剂量下该 Gaussian 对完整时间轴的期望计数。
3D-GTR 代码采用非归一化 Gaussian 峰幅度 A：

\[
h_{pd}(\tau)=\sum_j A_{pdj}
\exp\left[-\frac{(\tau-\mu_{pdj})^2}{2\sigma_{pdj}^2}\right].
\]

在连续时间表达式下两者满足 `N=A*sqrt(2*pi)*sigma`。
因此固定峰高与固定总光子数是不同的约束；做光子预算比较时必须明确使用哪一种。
目前数学测试使用 bin 积分与总计数幅度，不能把其 A 与原代码的峰高直接等同。

### C. 几何参数的 sensitivity：峰位、宽度和振幅

以原代码形式的一项非归一化 Gaussian `g=A*exp(-(tau-mu)^2/(2*sigma^2))` 为例：

\[
\partial_a g=g\left[
\partial_a\log A+
\frac{\tau-\mu}{\sigma^2}\partial_a\mu+
\frac{(\tau-\mu)^2}{\sigma^3}\partial_a\sigma
\right].
\]

若改用归一化密度且幅度 N 表示总光子数，则：

\[
\partial_a\log(Nf)=\partial_a\log N+
\frac{\tau-\mu}{\sigma^2}\partial_a\mu+
\left[\frac{(\tau-\mu)^2}{\sigma^3}-\frac1\sigma\right]\partial_a\sigma.
\]

多出的 `-1/sigma` 来自归一化系数；不能在两种幅度定义之间漏掉这一项。
真实直方图的 Jacobian 应再对 bin 积分，而不是直接把密度导数当计数导数。

位置参数影响峰位的核心式子是：

\[
\boxed{\partial_a\mu_{pdj}
=\frac{(u_{pj}+u_{dj})^T}{c}\partial_a x_j.}
\]

例如球面基元 `x_j=c_0+R(s)*n_j`，其中 n_j 是固定 canonical 方向，则：

\[
\frac{\partial\mu_{pdj}}{\partial R}
=\frac{(u_{pj}+u_{dj})^Tn_j}{c}.
\]

它可以为正、为负或接近零。膨胀并不意味着所有表面点相对墙面的光程都增加：靠墙一侧向墙运动时，光程可减少。
对于时间轨迹 `R(s)=R_0+v_R*s`，在给定状态时间 s 处还有：

\[
\frac{\partial\mu}{\partial v_R}
=s\frac{\partial\mu}{\partial R}.
\]

不同状态时间提供区分初始半径与速度的观测多样性；仅有一个状态时，它们可能高度混淆。

若要计算协方差传播引起的宽度导数，令 `Q=v^T Sigma v`：

\[
\partial_a\sigma=\frac{\partial_a Q}{2c\sqrt Q},\qquad
\partial_a Q=2(\partial_a v)^T\Sigma v+v^T(\partial_a\Sigma)v.
\]

其中对固定墙面点：

\[
\partial_a u=\frac{I-uu^T}{d}\partial_a x.
\]

这把“位置/形变 -> 距离与方向 -> 峰位/宽度 -> transient”的链式过程全部连接起来。
幅度还会随距离、方向、投影面积和散射强度变化，需要相同方式求导。

### D. 编码模型与资源约束

第 r 幅图案若采用归一化空间权重 `a_rp>=0,sum_p a_rp=1`，投入剂量为 Q_r，则定义：

\[
w_{rp}=Q_ra_{rp},\qquad
\lambda_{rdk}(\theta)=\sum_pw_{rp}h_{pdk}(\theta)+b_{rdk}(\theta).
\]

光子计数满足：

\[
y_{rdk}\overset{\mathrm{ind}}\sim\operatorname{Poisson}(\lambda_{rdk}).
\]

固定图案求导得到：

\[
\nabla_\theta\lambda_{rdk}=\sum_pw_{rp}\nabla_\theta h_{pdk}+\nabla_\theta b_{rdk}.
\]

已知、与参数无关的背景导数为零，但它仍增加分母中的噪声。
若图案是根据前一轮数据自适应选择，应在给定历史后计算本轮条件信息，再对历史取期望；不能把图案对未知真值的依赖当作额外可观测导数。

常用资源约束互不等价：

\[
\sum_{r,p}w_{rp}=Q_{\rm total}\quad\text{（总入射剂量）},
\]

\[
\sum_rw_{rp}=e_p\quad\text{（逐点剂量）},
\]

\[
\sum_r\Delta_r+R T_{\rm switch}\le T_{\rm budget}\quad\text{（总采集时间）}.
\]

还可能存在总功率 `sum_p w_rp/Delta_r<=P_max` 或逐点峰值功率约束。
若两者总功率都固定且没有切换开销，同时照亮更多点只是把同样功率分散，并不会自动收集更多光子。
若每个点功率固定、允许总功率随激活点数增加，编码的 photon throughput 优势部分来自更大的总功率。
因此实验中必须分别报告入射剂量、检测计数、曝光数和时间。

### E. Poisson Fisher 信息的逐步推导

用单索引 i 合并 r,d,k。联合概率为：

\[
p(y|\theta)=\prod_i\frac{\lambda_i(\theta)^{y_i}e^{-\lambda_i(\theta)}}{y_i!}.
\]

取对数并求导：

\[
\ell(\theta)=\sum_i[y_i\log\lambda_i-\lambda_i-\log(y_i!)],
\qquad \partial_a\ell=\sum_i\frac{y_i-\lambda_i}{\lambda_i}\partial_a\lambda_i.
\]

利用独立性：

\[
\mathbb E[(y_i-\lambda_i)(y_j-\lambda_j)]=\delta_{ij}\lambda_i,
\]

所以：

\[
\begin{aligned}
F_{ab}&=\mathbb E[(\partial_a\ell)(\partial_b\ell)]\\
&=\sum_{i,j}\frac{\partial_a\lambda_i\partial_b\lambda_j}{\lambda_i\lambda_j}
\mathbb E[(y_i-\lambda_i)(y_j-\lambda_j)]\\
&=\sum_i\frac{\partial_a\lambda_i\partial_b\lambda_i}{\lambda_i}.
\end{aligned}
\]

因此：

\[
\boxed{F=J^TD_\lambda^{-1}J=\widetilde J^T\widetilde J,\qquad
\widetilde J=D_\lambda^{-1/2}J.}
\]

F 为半正定，因为对任意 v：

\[
v^TFv=\|\widetilde Jv\|^2\ge0.
\]

若该值为 0，代表沿参数组合 v 的一阶观测变化为零，即局部一阶不可辨识。
这不排除某些非正则模型由二阶变化获得信息；常规可逆 Fisher 与 CRLB 分析无法处理这类奇异点。

当完整 F 正定且估计器满足局部无偏等条件时：

\[
\operatorname{Cov}(\hat\theta)\succeq F^{-1}.
\]

这是估计误差下界，不是某个优化算法一定达到的误差。
有偏估计、先验约束和正则化必须使用相应的偏差修正界或 Bayesian 界。

### F. 编码不超过保留来源标签的观测：定理与证明

**定理条件。** h 为可微的正期望计数；w 非负、已知且对 theta 固定；暂时无背景；相同 theta 描述全部曝光；计数独立且服从 Poisson。
对不参与测量的零剂量项可直接省略。参数相关均值为零处需另行检查正则性。

固定 r,d,k，设：

\[
g_p=\nabla h_{pdk},\quad\Lambda=\sum_pw_{rp}h_{pdk}.
\]

编码信息贡献为：

\[
F^{(rdk)}_{\rm code}
=\frac{(\sum_pw_{rp}g_p)(\sum_pw_{rp}g_p)^T}{\Lambda}.
\]

对任意参数方向 v，令 `z_p=v^T g_p`。使用 Cauchy-Schwarz：

\[
\begin{aligned}
\left(\sum_pw_{rp}z_p\right)^2
&=\left(\sum_p\sqrt{w_{rp}h_{pdk}}\,
\frac{\sqrt{w_{rp}}z_p}{\sqrt{h_{pdk}}}\right)^2\\
&\le\left(\sum_pw_{rp}h_{pdk}\right)
\left(\sum_pw_{rp}\frac{z_p^2}{h_{pdk}}\right).
\end{aligned}
\]

除以 Lambda，再对 r,d,k 求和：

\[
v^TF_{\rm code}v\le
\sum_{r,d,k,p}w_{rp}\frac{(v^Tg_p)^2}{h_{pdk}}
=v^TF_{\rm sep}v,
\]

其中：

\[
e_p=\sum_rw_{rp},\qquad
F_{\rm sep}=\sum_{p,d,k}e_p\frac{(\nabla h_{pdk})(\nabla h_{pdk})^T}{h_{pdk}}.
\]

因为 v 任意，所以：

\[
\boxed{F_{\rm code}\preceq F_{\rm sep}.}
\]

`A <= B` 的矩阵含义是 `B-A` 半正定，不要求 A 的每一个元素都小于 B。
它保证任意参数方向的二次型不增大。若两者均正定，逆矩阵顺序反转：

\[
F_{\rm code}^{-1}\succeq F_{\rm sep}^{-1}.
\]

因而代码比较的对应 CRLB 也具有明确的顺序。

**参照的含义。** 可以构造独立的潜在计数：

\[
z_{rpdk}\sim\operatorname{Poisson}(w_{rp}h_{pdk}),\qquad
y_{rdk}=\sum_pz_{rpdk}.
\]

完整观测 z 保留了照明来源标签 p；编码观测 y 把标签丢弃。
从 z 到 y 的求和是与未知参数无关的数据处理。
所以该定理也可理解为 Fisher 信息的数据处理不等式。
静态无背景下，按 p 聚合各 r 不丢失信息，因此 F_sep 可由同剂量的逐点扫描实现。

### G. 精确损失恒等式、取等条件与一个反例

固定 r,d,k，定义 `a_p=w_rp*h_pdk`、`s_p=grad log h_pdk`：

\[
\bar s=\frac{\sum_pa_ps_p}{\sum_pa_p},\qquad \Lambda=\sum_pa_p.
\]

分别观测的信息为 `sum_p a_p s_p s_p^T`，混合信息为 `Lambda*bar_s*bar_s^T`。
展开中心化外积：

\[
\begin{aligned}
\sum_pa_p(s_p-\bar s)(s_p-\bar s)^T
&=\sum_pa_ps_ps_p^T-\Lambda\bar s\bar s^T.
\end{aligned}
\]

对 r,d,k 相加得到：

\[
\boxed{F_{\rm sep}-F_{\rm code}
=\sum_{r,d,k}\Lambda_{rdk}\operatorname{Cov}_{\pi_{rdk}}(s_p),\qquad
\pi_{p|rdk}=\frac{w_{rp}h_{pdk}}{\Lambda_{rdk}}.}
\]

这里 pi 可以理解为：在这个 bin 观察到一个光子后，它来自位置 p 的概率。
编码丢失的信息恰好由“来源不确定性对应的 score 差异”决定。

完整矩阵取等要求，每个混合观测中所有正权重来源的 score 向量相同。
若只要求某个方向 v 不损失，则只需 `v^T s_p` 相同。
因此高压缩率仍可能保留某个低维任务的信息，但一般不能保留任意场景的全部参数信息。

例如在 `-1<theta<1` 中取：

\[
h_1=1+\theta,\quad h_2=1-\theta.
\]

theta=0 时分别观测的信息为 2；相加后均值恒为 2，导数为 0，信息为 0。
总光子数没减少，但所有关于 theta 的信息都在来源标签中，因此混合会完全丢失它。
相反，若 `h_1=a_1*f(theta),h_2=a_2*f(theta)` 且 a1,a2 已知，则两通道 score 相同，求和不损失关于 theta 的信息。

### H. 未知反照率、增益和时间零点：有效几何信息

把参数分为 `(theta_g,theta_n)`，并令白化 Jacobian 按列分为 `(Jg,Jn)`。
当 nuisance 信息块可逆时：

\[
F_{g|n}=F_{gg}-F_{gn}F_{nn}^{-1}F_{ng}
=J_g^T(I-P_n)J_g,
\]

\[
P_n=J_n(J_n^TJ_n)^{-1}J_n^T.
\]

这说明有效几何信息是几何导数中不能被 nuisance 导数线性解释的部分。
例如深度变化与时间零点变化可能同时移动峰；如果二者导数共线，深度信息会被消去。
强度导数也类似：半径变化只表现为幅度变化时，未知反照率可能吸收该变化。

编码完整信息不增大时，有效几何信息也不增大。可以用变分形式直接证明：

\[
v^TF_{g|n}v=\min_u
\begin{bmatrix}v\\u\end{bmatrix}^T
F
\begin{bmatrix}v\\u\end{bmatrix}.
\]

既然完整二次型对每个 `(v,u)` 都满足编码不大于参照，取最小值后仍满足同样顺序。
因此：

\[
F_{\rm code,g|n}\preceq F_{\rm sep,g|n}.
\]

若 F_nn 奇异，需要检查哪些 nuisance 方向本来冗余，并在适当的可辨识子空间处理；普通逆公式不能直接使用。
伪逆、ridge 或额外先验不能被解释为新采集到的数据 Fisher 信息。

### I. 算子秩、单位与“编码创造可辨识性”的范围

将静态 h 按 p,d,k 展平，则在 theta_0 附近：

\[
\delta h\approx J_h\delta\theta,\qquad
\delta\lambda\approx M J_h\delta\theta,
\quad M=W\otimes I_{DK}.
\]

因此：

\[
\operatorname{rank}(M J_h)\le
\min\{\operatorname{rank}(J_h),\operatorname{rank}(W)DK\}.
\]

若 `J_h v=0`，则任何这样的编码都满足 `M J_h v=0`。
原始完整观测的局部零空间不能靠线性强度混合消除。
但编码可以比某个更稀疏的点扫描访问更多通道，所以相对于那个弱基准可增加秩。
仅两幅图案也不代表最多估计两个参数，因为每幅图案保留 D*K 个独立观测通道。

普通 Fisher 特征值依赖参数单位。若 `theta=S*phi`：

\[
F_\phi=S^TF_\theta S.
\]

因此比较最小特征值前要固定参数尺度。对同一参照的广义比值：

\[
\gamma_{\min}=\min_{v\ne0}\frac{v^TF_{\rm code,eff}v}{v^TF_{\rm sep,eff}v}
\]

在参照正定时等于代码所画的归一化矩阵最小特征值，且落在 `[0,1]`。
这个比例在可逆线性重参数化下不变。

### J. 固定光子数下，单峰时间测距能达到的界

这个例子刻意只研究固定宽度、固定检测光子数、仅峰位依赖参数的模型。
设已检测到 N 个独立光子，且：

\[
\tau_i\sim\mathcal N(\mu(q),\sigma_t^2),\qquad
\sigma_t\text{ 已知且与 q 无关}.
\]

单光子 score 为 `(tau-mu)*mu'(q)/sigma_t^2`，所以：

\[
F_{qq}=\frac{N}{\sigma_t^2}\left(\frac{\partial\mu}{\partial q}\right)^2.
\]

令隐藏点沿单位方向 a 移动：`x(q)=x_0+q*a`，q 单位为 m：

\[
\frac{\partial\mu}{\partial q}=\frac{(u_p+u_d)^Ta}{c},\qquad
\left|\frac{\partial\mu}{\partial q}\right|\le\frac2c.
\]

于是：

\[
\boxed{F_{qq}\le\frac{4N}{c^2\sigma_t^2},\qquad
\operatorname{std}(\hat q)\ge\frac{c\sigma_t}{2\sqrt N}.}
\]

同向共聚焦、位移沿径向时可达到最大的光程导数 `2/c`。
仅一个已知峰的理想例子中，sigma_t=60 ps、N=6500，对应约 0.112 mm 的标准差界。
它不是前面带有多照明混合、未知强度和有限 bin 的数值实验的直接预测。

binning、混合来源和额外参数通常使实际精度降低。
若宽度或计数率也携带 q 的信息，模型本身已改变，上述式子就不是完整信息的上界。
对于连续 Poisson 过程 `lambda(tau;q)=N(q)*f(tau;q)`，且 f 归一化、有固定支撑和常规可微条件时：

\[
F_{qq}=\frac{[N'(q)]^2}{N(q)}+
N(q)\int f(\tau;q)[\partial_q\log f(\tau;q)]^2\,d\tau.
\]

第一项是总计数信息，第二项是到达时间分布信息。
若 f 为均值和宽度都依赖 q 的 Gaussian，第二项是：

\[
N\left[\frac{(\mu')^2}{\sigma_t^2}+2\left(\frac{\sigma_t'}{\sigma_t}\right)^2\right].
\]

未知 gain 可能消去计数项的有效几何信息，但不能不经分析就假设其必然全部消失。

### K. 固定剂量、固定时间、背景与动态扫描的区别

严格定理比较的是相同逐点剂量、同一个 theta、无额外噪声差别的两种观测。
它允许完整参照使用更多曝光；所以不是固定曝光次数或固定总时间的可实现最优策略证明。

已知背景存在时：

\[
F=\sum_i\frac{(\nabla\lambda_{\rm signal,i})(\nabla\lambda_{\rm signal,i})^T}
{\lambda_{\rm signal,i}+b_i}.
\]

若两种实验能构造成同一组潜在 Poisson 信号和背景通道的求和，数据处理上限仍成立。
但若逐点测量要付出更多次背景、读出或切换代价，实际扫描不等同于那个理想标签参照，不能直接套用无背景的 dose-only 不等式比较硬件性能。
固定光子预算下 Hadamard 也没有自动增益：负权图案需要两次非负曝光实现，差分的方差等于两次计数方差之和。

连续形变时，第 r 次曝光应写成：

\[
\lambda_{rdk}(\theta)=
\int_{s_r}^{s_r+\Delta_r}\sum_p I_{rp}(s)
h_{pdk}(\theta;s)\,ds+b_{rdk}.
\]

I 为剂量率，静态时 `w_rp=integral I_rp(s) ds`。
动态时不能仅用总剂量替代曝光时序：相同剂量在不同时刻照明会访问不同形状。
若为基元定义 canonical 形变映射：

\[
x_j(s)=c(s)+B(s)(x_j^0-c_0),\qquad
\Sigma_j(s)=B(s)\Sigma_j^0B(s)^T,
\]

就可以把轨迹参数代入 h，并在常规条件下将求导移入积分。
应先固定共同状态时间或明确估计共同轨迹参数，再比较 Fisher 信息。

对均匀曝光、线性漂移 `mu(s)=mu_0+v_tau*s`、固定 N 和 sigma 的一个时间核：

\[
\bar f(\tau)=\frac1\Delta\int_0^\Delta
f(\tau;\mu_0+v_\tau s,\sigma)\,ds.
\]

若 `v_tau != 0`，精确结果为：

\[
\bar f(\tau)=\frac{
\Phi((\tau-\mu_0)/\sigma)
-\Phi((\tau-\mu_0-v_\tau\Delta)/\sigma)
}{v_\tau\Delta}.
\]

中心与方差分别为 `mu_0+v_tau*Delta/2` 和 `sigma^2+(v_tau*Delta)^2/12`。
把它近似成 Gaussian 是矩匹配近似，分布本身一般不严格为 Gaussian。
对固定墙面几何：

\[
v_\tau=\frac{d\mu}{ds}=\frac{(u_p+u_d)^T\dot x}{c}.
\]

共聚焦才化为 `2*u^T*x_dot/c`。两个径向速度相加后不能再多乘一次 2。
若曝光积分本身把快慢变化平均掉，SLM 强度编码也不能凭空恢复未记录的时间标签。

### L. 最优照明与“能增强多少”的可计算表达

在约束集合 C 中，针对几何任务可设计：

\[
W^*=\arg\max_{W\in\mathcal C}\log\det(F_{g|n}(W))\quad\text{（D-optimal）},
\]

\[
W^*=\arg\max_{W\in\mathcal C}\lambda_{\min}(F_{g|n}(W))\quad\text{（E-optimal）},
\]

\[
W^*=\arg\min_{W\in\mathcal C}\operatorname{tr}(F_{g|n}(W)^{-1})\quad\text{（A-optimal）}.
\]

参数必须先按任务尺度归一化。奇异设计的无正则 logdet 为负无穷，逆不存在；数值稳定项不能作为实验信息计入 CRLB。
设计依赖 theta_0，未知场景可用 pilot measurement 初始化，再做局部自适应设计；或对参数范围做 Bayesian/最坏情形设计。

若仅估计一个标量 theta，无 nuisance、无背景，令单点单位剂量 Fisher 信息为：

\[
I_p=\sum_{d,k}\frac{(\partial_\theta h_{pdk})^2}{h_{pdk}}.
\]

在总剂量 Q 固定且允许任意逐点分配时：

\[
F_{\rm code}\le F_{\rm sep}=\sum_pe_pI_p
\le Q\max_pI_p,\qquad\sum_pe_p=Q.
\]

这给出明确的标量任务设计上限：没有额外约束时，把所有剂量投给单位剂量信息最高的照明位置即可达到最优参照。
该结论解释了为什么多点混合并不自动提高 shot-noise-limited 的单参数信息。
多参数任务不同：单点可能只约束一个组合，必须在不同灵敏度方向间分配剂量以保持矩阵可逆。
未知 nuisance 也可能使单点策略不可辨识，所以必须优化有效信息而不是套用上述单参数公式。

固定逐点剂量条件下，任意编码的上限就是 F_sep；不同可实现约束下，最优设计值是对应约束集合内的最优化问题。
目前代码未求解最优 W，只扫描了一族左右互补图案，不能把图中的最好结果称作所有编码的最优上限。

### M. 相干 wavefront shaping 的另一条上界

若需要控制光学复场而非墙面强度，改用：

\[
E(\theta)=H(\theta)e,\qquad
\lambda_q=|(H(\theta)e)_q|^2,
\quad\|e\|^2=P_{\rm in}.
\]

H 是复数场传输矩阵，e 是 SLM 的入射复振幅；探测模式 q 可以包含空间、时间或频率，但必须先定义清楚对应的独立 Poisson 观测。
输入功率或曝光能量到计数的固定效率系数在此吸收到 H 中。
设只估计标量 theta，求导时 e 固定：

\[
\partial_\theta\lambda_q=2\operatorname{Re}
[(He)_q^*(H_\theta e)_q].
\]

利用 `Re(z)^2<=|z|^2`：

\[
\begin{aligned}
F_{\theta\theta}
&=\sum_q\frac{4\operatorname{Re}[(He)_q^*(H_\theta e)_q]^2}{|(He)_q|^2}\\
&\le4\sum_q|(H_\theta e)_q|^2
=4\|H_\theta e\|^2\\
&\le4P_{\rm in}\|H_\theta\|_2^2.
\end{aligned}
\]

这是假设正均值及常规可微条件下的经典 Poisson 直接探测上界。严格暗模式处要单独处理零均值带来的非正则性。
它不是自动可达到的 quantum Fisher bound。
取等既需要输入匹配最大奇异向量，也需要探测强度对场变化满足相位条件。
例如 `H(theta)=exp(i*theta)*H0` 只引入全局相位，直接强度观测完全不变、F=0，但上述上界仍为正。
所以最大导数奇异值只给一个宽松上界，不保证能用强度探测把该场信息全部读出来。

phase-only SLM 还要求各像素振幅固定，只能改变相位；其可行输入是固定范数球面的子集，因此最优值可能低于该界。
实验上还需校准 H 并维持光路相干稳定；动态气球可能改变 H。
当前 CPU 测试没有实现这套相干模型，非负混合定理也不能直接约束由相干交叉项产生的新观测模式。

### N. 当前代码覆盖与研究结论边界

| 数学部分 | 当前是否已实现 |
| --- | --- |
| 静态单 Gaussian 的 bin 积分与解析 Jacobian | 已实现，并用有限差分检查 |
| 独立 Poisson Fisher 信息 | 已实现 |
| 固定逐点剂量的非负编码上限 | 已实现，检查 30 组随机编码 |
| score 协方差损失恒等式与取等反例 | 已实现 |
| 未知 gain 的 Schur complement | 已实现 |
| 局部最大似然估计与 Monte Carlo | 已实现 |
| 半径、速度、曝光积分和异步扫描 | 尚未实现 |
| 背景、dead time、峰值功率和切换耗时 | 尚未实现 |
| 任意约束下的最优图案搜索 | 尚未实现 |
| 复数传输矩阵与相干波前整形 | 尚未实现 |

当前能够支持的结论是：在指定静态、无背景、线性 Poisson 模型与相同逐点剂量下，编码不会超过保留照明标签的完整观测。
要证明其对膨胀气球具有实际增益，必须进一步指定时间、功率和噪声约束，再比较相同可用资源内的编码和点扫描策略。
