# 发射端波前整形与几何信息：可解析的最小验证

本实验回答一个限定的问题：**仅改变发射端 SLM 相位，能否改变隐藏目标的照明分布，并使某个几何参数比“照得最亮”时更容易估计？**

模型在照明传播中保留干涉；接收端只记录光子到达时间直方图，按强度叠加，不测量返回光的光学相位。程序不使用 RSD，也不调用完整 3D-GTR。它通过与 3D-GTR 相同的“光程决定时间峰”的关系建立几何计数模型。

## 1. 模型范围与光路

```text
发射激光（固定输入光子预算）
    ↓
phase-only SLM：只改变各输入模式的相位 phi
    ↓
固定传播 A：包含理想相位散射与模式混合
    ↓
隐藏区域 j 的照明强度 w_j = |(A e)_j|²
    ↓
几何状态 q 决定各区域的到达时间分布 h_jk(q)
    ↓
返回强度相加、探测效率、时间分箱
    ↓
SPAD：y_k ~ Poisson(lambda_k)
```

这里控制的始终是照明，接收端没有可调的光学相位。

三个层面应分别理解：

| 层面 | 本实验的表示 |
| --- | --- |
| 照明端相干传播 | 复矩阵 A 与相位输入 e |
| 目标至接收端的计数响应 | 非负瞬态强度 h_jk(q) |
| 参数反演与设计 | Poisson 似然、Jacobian、Fisher 信息 |

不同输入模式在同一目标区域发生照明干涉，与不同目标区域的返回强度在接收端相加，是两项独立的模型假设。例如接收端对足够多独立散斑/偏振模式的积分，可能降低区域间的返回交叉项。本程序直接采用这一强度接收近似，未从具体粗糙表面的统计模型推导它；真实实验需要检查该近似。

此外，假定局部形变范围内照明分布 A、接收效率和脉冲宽度近似固定，仅光程发生变化。真实目标运动可能改变 A、遮挡和方向响应，这些项尚未实现。程序中的毫米级参数和搜索区间是该数学模型的坐标，不能直接作为真实散斑场的稳定范围。

## 2. SLM 如何改变隐藏目标的照明

### 2.1 输入与相干传播

使用 M=8 个输入模式，归一化输入为

$$
e_l(\phi)=\frac{e^{i\phi_l}}{\sqrt M},\qquad \|e\|^2=1.
$$

每次曝光的实际输入光子预算 N_in 单独乘在计数模型上；所有图案的 N_in 相同。

传播矩阵取为

$$
A=F_M D,\qquad
(F_M)_{jl}=\frac{e^{-2\pi i jl/M}}{\sqrt M},\qquad
D=\operatorname{diag}(e^{i\psi_l}).
$$

其中 psi 是固定随机相位屏。该模型是“薄相位屏 + 理想离散传播”的示例，不是实际 NLOS 墙面标定。
它满足 A^dagger A=I，所以隐藏区域收到的总照明保持不变：

$$
u=Ae,\qquad w_j=|u_j|^2,\qquad \sum_jw_j=1.
$$

相位改变的是能量在隐藏区域之间的分配，而不是增加总输入能量。
Flat SLM 指 phi=0；由于存在相位屏 D，它在目标处通常产生非均匀照明，不能当作目标平面上的均匀光。

### 2.2 相位导数

$$
\frac{\partial u_j}{\partial\phi_l}=iA_{jl}e_l,
\qquad
\boxed{\frac{\partial w_j}{\partial\phi_l}
=2\operatorname{Re}[u_j^*iA_{jl}e_l].}
$$

这是程序 illumination() 的核心公式。它说明即使 SLM 不改变任何输入模式的强度，也能通过传播后的干涉改变目标照明。

### 2.3 相位平均对照

为了验证干涉是否真是作用来源，引入互相干矩阵

$$
C_{\rm in}=\gamma ee^\dagger+(1-\gamma)\operatorname{diag}(|e|^2),
\qquad 0\le\gamma\le1.
$$

这是一个明确的半正定混合模型，gamma 为假设的相干权重，不是从真实激光带宽估算的参数。目标照明为

$$
w_j(\gamma)
=\operatorname{diag}(A C_{\rm in}A^\dagger)_j
=\gamma|(Ae)_j|^2+(1-\gamma)\sum_l|A_{jl}|^2|e_l|^2.
$$

本例中 |e_l|²=|A_jl|²=1/M，因此

$$
w_j(0)=1/M,\qquad \frac{\partial w_j(0)}{\partial\phi_l}=0.
$$

当交叉项完全平均掉，同一 phase-only 控制的所有图案应得到相同结果。注意此时几何 Fisher 信息通常仍大于零；消失的是相位控制带来的差别，而非测量本身的全部信息。

## 3. 几何参数如何进入接收到的瞬态

### 3.1 q 是什么

本次不直接声称恢复气球半径。设 q 为一个预定义局部形变模式的幅度，单位 mm：

$$
z_j(q)=z_j^0+\beta_j q\times10^{-3}\ \mathrm{m}.
$$

z_j^0 从 0.65 m 到 0.85 m；beta_j 表示区域 j 随该模式运动的程度。采用共聚焦式等效往返光程，得到

$$
\mu_j(q)=\frac{2z_j(q)}c,\qquad
\boxed{\frac{\partial\mu_j}{\partial q}
=\frac{2\beta_j10^{-3}}c.}
$$

程序的时间单位为 ps，c=2.99792458e-4 m/ps，故导数单位是 ps/mm。
beta 最大为1，对应单点径向位移的光程导数界。这里只是受控几何模式，并非含完整照明墙/探测墙坐标的 NLOS 仿真。

### 3.2 时间 bin 积分

各区域时间核标准差固定为 sigma=60 ps，bin 宽度为16 ps。单位照明下的计数响应为

$$
h_{jk}(q)=\eta_j m_{jk}(q),
$$

$$
m_{jk}=\Phi\left(\frac{t_{k+1}-\mu_j}{\sigma}\right)
-\Phi\left(\frac{t_k-\mu_j}{\sigma}\right).
$$

eta_j 是从目标区域返回到探测器的固定效率。各时间 bin 的期望计数为

$$
\boxed{\lambda_k(q,\phi)=N_{\rm in}\sum_jw_j(\phi)h_{jk}(q).}
$$

这一步是强度叠加，接收端没有 e^{i optical_phase} 项。若完整时间窗覆盖信号，总回波计数近似 N_in*sum_j w_j*eta_j。

### 3.3 几何导数

令 l=(t_k-mu_j)/sigma、u=(t_{k+1}-mu_j)/sigma，用 varphi 表示标准正态密度：

$$
\frac{\partial m_{jk}}{\partial\mu_j}
=\frac{\varphi(l)-\varphi(u)}\sigma,
\qquad
\frac{\partial h_{jk}}{\partial q}
=\eta_j\frac{\varphi(l)-\varphi(u)}\sigma
\frac{2\beta_j10^{-3}}c.
$$

因此

$$
d_k:=\frac{\partial\lambda_k}{\partial q}
=N_{\rm in}\sum_jw_j\frac{\partial h_{jk}}{\partial q}.
$$

这对应 target_response() 和 moments()。代码用中心有限差分检查该解析导数。

## 4. 两种照明优化目标

### 4.1 最大接收光子数

照明优化可以最大化

$$
C(\phi)=\sum_k\lambda_k,\qquad
\partial_{\phi_l}C
=N_{\rm in}\sum_{j,k}\frac{\partial w_j}{\partial\phi_l}h_{jk}.
$$

其偏好效率 eta_j 大的区域。这个目标只通过发射端相位实现，称为 Max photons。

### 4.2 最大参数 Fisher 信息

独立 Poisson 计数给出

$$
F_{qq}(\phi)=\sum_k\frac{d_k^2}{\lambda_k}.
$$

先对相位求观测导数：

$$
\lambda_{k,l}=N_{\rm in}\sum_j
\frac{\partial w_j}{\partial\phi_l}h_{jk},\qquad
d_{k,l}=N_{\rm in}\sum_j
\frac{\partial w_j}{\partial\phi_l}\frac{\partial h_{jk}}{\partial q}.
$$

于是

$$
\boxed{\partial_{\phi_l}F_{qq}
=\sum_k\left[
\frac{2d_kd_{k,l}}{\lambda_k}
-\frac{d_k^2\lambda_{k,l}}{\lambda_k^2}
\right].}
$$

objective() 同时实现上述目标与解析梯度，再用有限差分独立核验。
optimize_phase() 使用8个随机初值运行 L-BFGS-B，固定最后一个相位为0以消除全局相位冗余。它不使用解析最佳聚焦相位初始化。

## 5. 未知强度与已知时间零点

计数模型允许一个未知全局增益 g，真实值为1：

$$
\lambda_k(q,g)=g\,b_k(q).
$$

令 nu=log(g)，则增益导数为 lambda，Fisher 信息块为

$$
F_{q\nu}=\sum_kd_k,\qquad F_{\nu\nu}=\sum_k\lambda_k,
$$

$$
F_{q|\nu}=F_{qq}-\frac{(\sum_kd_k)^2}{\sum_k\lambda_k}.
$$

本模型总计数近似不随q改变，交叉项近0，因此优化 F_qq 与优化有效信息近似等价。程序最终仍显式计算并报告 Schur complement。

本次主实验假设系统时间零点已知。如果把所有光照聚焦到一个区域，q引起的峰位变化会与未知 t0 完全共线：

$$
\partial_q\lambda_k=\mu_j'\,\partial_{t_0}\lambda_k.
$$

消去未知 gain 和 t0 后，该聚焦方案的有效q信息为0。程序将它作为独立的边界检查。
因此“聚焦到最敏感区域”的结果依赖系统时间标定；当 t0 也未知时，需要重新设计包含不同几何响应的多次照明。

## 6. 这个例子的全局上界为什么能算出来

定义隐藏区域 j 单位照明剂量的信息

$$
I_j=\sum_k\frac{(\partial_q h_{jk})^2}{h_{jk}}.
$$

接收时混合不同隐藏区域，沿用非负混合的信息不等式：

$$
F_{qq}(\phi)\le N_{\rm in}\sum_jw_j(\phi)I_j
\le N_{\rm in}\max_j I_j.
$$

该界并未被相干整形突破。相干照明允许改变隐藏区域的剂量分配w；固定输入端每个SLM模式的强度，并不等于固定每个隐藏区域的剂量。
完整理想参照是“能够直接把能量送到任意隐藏区域且保留来源”，它与之前的可见墙面非相干图案参照不同。

在当前 A 中，每个元素模长相同。相位共轭

$$
\phi_l=-\arg A_{j_*l},\qquad j_*=\arg\max_j I_j
$$

能使 w_j*=1，其他w为0，因而达到上界。本例中总计数与q无关，所以未知gain不破坏该可达性。
程序用这个解析解检查上界可达性，同时报告随机多初值优化与上界的间隙。全局最优的依据是这套特殊A的结构和上界，不是局部优化器返回success。

在近似连续时间、已知宽度条件下，I_j约为

$$
I_j\approx\eta_j\frac{(\mu_j')^2}{\sigma^2}.
$$

所以最亮区域由eta决定，而最敏感区域由eta与几何斜率平方共同决定。

## 7. 人为设置的可解释场景

| 隐藏区域 j | beta_j | eta_j |
| --- | ---: | ---: |
| 0 | 0.05 | 0.00080 |
| 1 | 0.08 | 0.00070 |
| 2 | 0.15 | 0.00050 |
| 3 | 0.30 | 0.00035 |
| 4 | 1.00 | 0.00010 |
| 5 | 0.60 | 0.00020 |
| 6 | 0.20 | 0.00040 |
| 7 | 0.10 | 0.00060 |

区域0最容易返回光子，但随q变化较慢；区域4返回效率低，却对q变化最明显。
这组参数特意用来隔离“亮度与参数信息”的区别，不是实验拟合，也不是证明所有场景必然出现这种反差。
若各区域beta相同，在该可聚焦模型中最大回波与最大信息的最佳区域可重合。

## 8. 数值估计如何检验理论

设计固定后，以q=0为真值生成Poisson计数。每种设计重复200次，联合估计q和未知gain。
对给定q，gain可解析profile：

$$
\hat g(q)=\frac{\sum_k y_k}{\sum_k b_k(q)}.
$$

将其代回Poisson负对数似然

$$
L(q)=\sum_k[\hat g(q)b_k(q)-y_k\log(\hat g(q)b_k(q))]
$$

后做一维有界最小化。将经验标准差、偏差和RMSE与

$$
\operatorname{std}(\hat q)\ge1/\sqrt{F_{q|\nu}}
$$

比较。CRLB适用于局部无偏与常规正则条件；有限次重复及估计偏差会产生波动。
这是已知参考形状附近的参数估计，不是从零恢复任意目标。

## 9. 首轮运行结果

seed=20261027；每种设计输入光子预算均为2e7；每种设计200次随机拟合：

| 照明设计 | 期望检测光子数 | 有效Fisher信息 (mm^-2) | CRLB标准差 (mm) | 经验标准差 (mm) |
| --- | ---: | ---: | ---: | ---: |
| 平相位SLM参考 | 11266 | 1.0842 | 0.9604 | 0.9042 |
| 最大检测光子数 | 16000 | 0.4916 | 1.4262 | 1.3674 |
| 最大参数信息 | 2000 | 24.5799 | 0.2017 | 0.1961 |

这里最大信息照明的总检测计数只有最大光子照明的1/8，但Fisher信息为其50倍，对应CRLB标准差约改善7.07倍。
每检测光子的信息比例约为400倍，来自人为选择的几何响应斜率平方比；不是对真实NLOS气球的倍率预测。

所有输入均保持相同强度及总能量。两种优化各8个随机初值均正常终止，信息优化值与可证全局上界的相对间隙约为1e-14。
几何Jacobian相对误差约1.9e-9，两种相位设计梯度误差均低于2e-10。600次计数拟合均成功。

![照明分布、检测光子数、每光子信息和相干性对照](results/illumination_wavefront/illumination_comparison.png)

图的右下角固定使用gamma=1时设计的图案，随后改变gamma评价，不在每个gamma处重新优化。
gamma=0时三条曲线相交，说明所有相位图案在相位平均模型下等价；gamma增加也不保证任意设计信息都单调提高。

## 10. 运行与阅读顺序

从仓库根目录运行：

```powershell
python math_tests/illumination_wavefront.py
```

只跑数值、不绘图：

```powershell
python math_tests/illumination_wavefront.py --no-plots
```

主要输出：

- results/illumination_wavefront/report.json：参数、检查、设计相位、照明权重、上界与Monte Carlo结果；
- results/illumination_wavefront/illumination_comparison.png：对比图。

支持 --photons、--trials、--seed、--output；默认覆盖同名输出。依赖沿用math_tests/requirements.txt。

推荐按此顺序阅读代码：

| 函数 | 对应数学 |
| --- | --- |
| propagation() | 已知相干照明传播A及能量守恒 |
| illumination() | 相位到隐藏照明的映射与导数 |
| target_response() | 几何光程到时间bin及导数 |
| moments() | 发射控制到实际接收计数的组合 |
| information() | Poisson信息与未知gain消元 |
| objective() | 两种设计目标及相位梯度 |
| optimize_phase() | phase-only数值设计 |
| run_checks() | 独立导数、上界、相位平均及未知t0检查 |
| monte_carlo() | 用实际随机计数检验估计误差 |

## 11. 结论与下一步边界

本验证支持以下限定结论：已知照明传播、固定输入能量与已知时间零点时，发射端phase-only控制可通过照明干涉选择对预定义几何模式更敏感的隐藏区域；接收端即使只计数瞬态强度，也可得到不同参数精度。优化接收光子数与优化几何信息可能给出不同照明。

这里不需要把RSD的虚拟相位解释为激光相位，也没有使用相位敏感接收的理想场信息公式作为SPAD的实际信息。

尚未覆盖的环节为：真实A的获取、A随形变的变化、激光频谱与相位屏色散、返回干涉的统计平均、完整气球几何、未知t0的多图案联合设计、标定与更新耗时。下一步应首先确认实际光路中可采用哪种A，以及能否在目标变化前完成标定与更新。
