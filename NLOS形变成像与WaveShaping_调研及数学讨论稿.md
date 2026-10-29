# 波前整形增强可形变目标非视域成像的可行性分析

**研究备忘录 · 2026 年 10 月 27 日**

## 摘要

本研究拟以三维高斯瞬态渲染（3D Gaussian Transient Rendering，3D-GTR）为基线，探索通过空间光调制提高非视域（non-line-of-sight，NLOS）测量对目标形变的灵敏度，并以气球膨胀作为首个验证场景。核心问题是：在给定照明能量、接收孔径和采集时间的条件下，能否通过优化入射光场，提高半径、膨胀速度等几何参数的估计精度？

分析采用 Poisson 光子计数模型及 Fisher 信息矩阵，区分墙面非相干强度编码与相干波前整形两类机制。对于前者，可以证明：在逐点照明剂量相同的条件下，编码混合后的 Fisher 信息不超过保留照明来源标签的完整观测；其信息损失由被混合通道响应得分的加权协方差精确刻画。对于后者，入射相位通过干涉改变观测的参数导数，其信息收益需由复数散射矩阵及实际探测方式共同确定。相干散射参数估计和主动 NLOS 聚焦的已有工作为该方向提供了依据。

目前已完成静态单高斯模型下的非相干编码数值验证。后续拟重点比较随机入射、最大回波入射和最大几何信息入射，并分别评估回波光子数、每光子信息及标定更新代价。本文给出上述分析的数学依据、已有验证及建议的研究步骤。

**关键词：** 非视域成像；波前整形；可形变目标；Fisher 信息；最优实验设计；瞬态渲染

## 1. 研究问题与基本判断

### 1.1 研究目标

膨胀目标在采集过程中持续改变自身几何，使 NLOS 测量同时受到弱回波、空间欠采样和时间混合的影响。本研究考虑在几何反演之外引入主动照明设计，使有限的光子和采集时间更多地用于区分目标形变。

首阶段关注低维参数，例如初始半径、膨胀速度和目标中心；在这些变量可辨识后，再扩展到非均匀局部形变。研究目标可表述为：

> 在明确的光学控制与接收条件下，设计使几何参数估计误差最小的照明场，并确定其相对于常规照明的增益及可达上限。

这一目标需要同时回答三个问题：

1. **物理机制：** 光场整形改变了回波数量、几何响应，还是两者兼有？
2. **统计收益：** 在排除未知反照率、系统增益和时间零点后，几何参数的信息是否增加？
3. **实验可实现性：** 标定、优化和采集能否在目标与散斑保持有效稳定的时间内完成？

### 1.2 基本判断

照明优化用于参数精密估计已有理论与实验先例；波前整形用于主动 NLOS 聚焦也已有验证。因此，值得研究的切入点是**面向非视域形变参数的光场设计及其信息效率**。

需要区分两类方案：

| 方案 | 受控量 | 计数模型 | 主要研究问题 |
| --- | --- | --- | --- |
| 墙面强度编码 | 不同墙面位置的非负照明剂量 | 各点响应的加权和 | 有限曝光内如何保留几何信息 |
| 相干波前整形 | 入射模式的相位与振幅 | 场叠加后取模平方 | 如何选择对几何变化最敏感的入射态 |

两类方案均可使用 SLM，但器件相同不代表物理机制相同。前者适合直接扩展现有强度瞬态模型；后者需要描述散射干涉，并明确哪些场信息可由探测器读出。

## 2. 相关工作与研究定位

### 2.1 几何反演基线：3D-GTR

3D-GTR 用三维高斯基元表示隐藏场景，通过可微瞬态渲染拟合测量，支持稀疏采样和任意中继面几何 [1]。其解析光程映射便于计算几何参数的 Jacobian，因此可作为照明设计的信息评价器。

现有实现使用静态高斯参数，并合成虚拟墙瞬态供后续 LCT/RSD 重建。对连续形变，需要增加状态时间、曝光积分及共享形变参数；相干波前整形还需要额外的场传播模型。

### 2.2 主动聚焦与参数信息优化

UNCOVER 通过波前整形在隐藏目标处主动聚焦，说明可利用中继面的散射建立主动 NLOS 照明 [2]。其结果支持光场控制的可行性，但聚焦能力与形变参数估计精度仍需分别评价。

Maximum information states 工作将参数 Fisher 信息与散射矩阵导数联系起来，识别对目标参数最敏感的入射态 [3]。其相位敏感读出包含参考光，移植到 SPAD 瞬态测量时需重新计算实际可获得的信息。

多参数衍射成像中的照明优化 [4]、时间波形信息优化 [6]，以及面向强度读出的张量波前整形 [7]，进一步说明照明设计应围绕具体参数、噪声与探测方式展开。正 Hadamard/Cosine 编码的光子噪声分析则表明，多路复用的信噪比优势具有明确适用条件 [5]。矢量光场主动 NLOS 工作提供了相位与偏振联合控制的参考 [8]。

### 2.3 本研究的定位

上述工作分别提供几何表示、主动聚焦和信息优化方法。拟研究的结合点是：

> 将可形变目标的几何参数显式连接到 NLOS 测量的参数导数，在实际光子计数和光学控制约束下优化入射场。

目前尚未验证该结合能在连续膨胀气球上获得净收益。研究贡献应由几何耦合模型、资源约束下的收益及真实实验共同支撑，而非仅采用 Fisher 信息作为优化目标。

## 3. 测量模型与几何灵敏度

### 3.1 光程约束与形变表示

扣除已标定的外部光程后，隐藏点 $\boldsymbol x$ 对墙面照明点 $\boldsymbol p_p$ 和探测点 $\boldsymbol q_d$ 的飞行时间为

$$
\tau_{pd}(\boldsymbol x)
=
\frac{\|\boldsymbol x-\boldsymbol p_p\|
+\|\boldsymbol x-\boldsymbol q_d\|}{c}.
$$

以 $\tau$ 表示光子飞行时间，以 $s$ 表示目标状态时间。两者分别对应皮秒至纳秒的到达时间，以及毫秒至秒的扫描与膨胀过程。

对第 $j$ 个高斯基元，记中心为 $\boldsymbol x_j$、协方差为 $\Sigma_j$，定义

$$
d_{pj}=\|\boldsymbol x_j-\boldsymbol p_p\|,\qquad
d_{dj}=\|\boldsymbol x_j-\boldsymbol q_d\|,
$$

$$
\boldsymbol u_{pj}
=\frac{\boldsymbol x_j-\boldsymbol p_p}{d_{pj}},\qquad
\boldsymbol u_{dj}
=\frac{\boldsymbol x_j-\boldsymbol q_d}{d_{dj}},\qquad
\boldsymbol v_{pdj}=\boldsymbol u_{pj}+\boldsymbol u_{dj}.
$$

在基元尺寸相对传播距离较小的一阶光程近似下，时间高斯满足

$$
\mu_{pdj}=\frac{d_{pj}+d_{dj}}{c},\qquad
\sigma_{pdj}^{2}=\frac{\boldsymbol v_{pdj}^{T}\Sigma_j\boldsymbol v_{pdj}}{c^{2}}.
$$

可采用共享形变映射

$$
\boldsymbol x_j(s)=\boldsymbol c(s)
+B(s)(\boldsymbol x_j^0-\boldsymbol c_0),\qquad
\Sigma_j(s)=B(s)\Sigma_j^0B(s)^T.
$$

球形膨胀对应 $B(s)=a(s)I$，椭球膨胀对应三个方向的伸缩。低维参数可选为

$$
\boldsymbol\theta_g=[R_0,v_R,c_x,c_y,c_z,\ldots]^T.
$$

散射强度、增益、时间零点、背景及系统脉冲响应参数记为 $\boldsymbol\theta_n$，完整参数为
$\boldsymbol\theta=(\boldsymbol\theta_g,\boldsymbol\theta_n)$。

### 3.2 从几何变化到瞬态变化

峰位对几何参数的导数为

$$
\boxed{
\frac{\partial\mu_{pdj}}{\partial\theta_a}
=
\frac{\boldsymbol v_{pdj}^{T}}{c}
\frac{\partial\boldsymbol x_j}{\partial\theta_a}.
}
$$

若球面基元为 $\boldsymbol x_j=\boldsymbol c_0+R\boldsymbol n_j$，则

$$
\frac{\partial\mu_{pdj}}{\partial R}
=
\frac{(\boldsymbol u_{pj}+\boldsymbol u_{dj})^T\boldsymbol n_j}{c}.
$$

因此，同一半径变化在不同光路上具有不同的投影，导数可为正、负或接近零。靠墙一侧的表面向外膨胀时可能接近墙面，光程随之缩短。

对一项非归一化时间高斯

$$
h(\tau)=A\exp\!\left[-\frac{(\tau-\mu)^2}{2\sigma^2}\right],
$$

有

$$
\partial_a h
=
h\left[
\partial_a\log A+
\frac{\tau-\mu}{\sigma^2}\partial_a\mu+
\frac{(\tau-\mu)^2}{\sigma^3}\partial_a\sigma
\right].
$$

三项分别对应幅度、峰位和宽度变化。使用归一化时间密度时，宽度导数还包含归一化系数的导数，详见附录 A。实际直方图应对每个时间 bin 积分。

### 3.3 Poisson 观测与有效几何信息

在线性、低通量光子计数近似下，

$$
y_i\overset{\mathrm{ind}}{\sim}
\operatorname{Poisson}(\lambda_i(\boldsymbol\theta)),
$$

对应 Fisher 信息矩阵

$$
\boxed{
F_{ab}
=
\sum_i
\frac{\partial_a\lambda_i\,\partial_b\lambda_i}{\lambda_i}.
}
$$

令 $J_{ia}=\partial_a\lambda_i$，则

$$
F=J^TD_\lambda^{-1}J
=\widetilde J^T\widetilde J,\qquad
\widetilde J=D_\lambda^{-1/2}J.
$$

这意味着灵敏度应结合噪声评价：更大的绝对导数未必带来更高精度，因为 Poisson 方差也随均值增加。

当完整信息矩阵正定、模型满足常规正则条件且估计器局部无偏时，

$$
\operatorname{Cov}(\hat{\boldsymbol\theta})\succeq F^{-1}.
$$

将几何与干扰参数分块，在 $F_{nn}$ 可逆时，几何的有效信息为

$$
\boxed{
F_{g|n}=F_{gg}-F_{gn}F_{nn}^{-1}F_{ng}.
}
$$

它扣除了可被未知强度、时间零点等变量解释的观测变化。例如，若半径变化仅使回波整体变亮，这一效应可能被未知增益吸收。因而照明设计不仅要增大几何响应，还要使它与干扰响应可区分。

## 4. 非相干强度编码的信息边界

### 4.1 编码模型与参照

设 $h_{pdk}(\boldsymbol\theta)$ 为单位照明剂量下的点响应，
$w_{rp}\ge0$ 为第 $r$ 次曝光分配到第 $p$ 个墙面位置的剂量。在无背景、图案固定且场景静态时，

$$
\lambda_{rdk}=\sum_p w_{rp}h_{pdk},\qquad
\nabla\lambda_{rdk}=\sum_p w_{rp}\nabla h_{pdk}.
$$

定义逐点总剂量 $e_p=\sum_r w_{rp}$。保留照明来源标签的完整参照具有

$$
F_{\mathrm{sep}}
=
\sum_{p,d,k}
e_p\frac{(\nabla h_{pdk})(\nabla h_{pdk})^T}{h_{pdk}}.
$$

静态无背景条件下，该参照可由相同逐点剂量的完整扫描实现。这里尚未计入额外曝光和切换的时间代价。

### 4.2 上限及信息损失

对上述模型，可以证明

$$
\boxed{F_{\mathrm{code}}(W)\preceq F_{\mathrm{sep}}(e).}
$$

更精确地，固定 $r,d,k$，令

$$
a_p=w_{rp}h_{pdk},\qquad
\boldsymbol s_p=\nabla\log h_{pdk},\qquad
\bar{\boldsymbol s}_{rdk}
=\frac{\sum_p a_p\boldsymbol s_p}{\sum_p a_p}.
$$

则

$$
\boxed{
F_{\mathrm{sep}}-F_{\mathrm{code}}
=
\sum_{r,d,k,p}
a_p(\boldsymbol s_p-\bar{\boldsymbol s}_{rdk})
(\boldsymbol s_p-\bar{\boldsymbol s}_{rdk})^T
\succeq0.
}
$$

完整证明见附录 B。该式给出三个直接结论：

- 固定逐点剂量时，上限与具体编码类型无关。
- 编码的作用体现在信息损失大小：混合通道的响应得分越相似，损失越小。
- 所有混合来源的响应得分相同时取等；导数方向相反时可能严重抵消。

直观上，完整观测知道光子来自哪个照明位置，编码测量只保留总计数。信息损失来自来源标签的丢弃。

例如 $h_1=1+\theta$、$h_2=1-\theta$，在 $\theta=0$ 时分别观测的 Fisher 信息为 $2$，相加后均值恒为 $2$，信息为 $0$。光子数量不变，参数响应却被抵消。

### 4.3 上限的适用范围

该结论针对同一组非相干通道在相同逐点剂量下的混合。它不直接比较不同剂量分配、不同采集时序，或具有相干交叉项的观测。

| 比较条件 | 可得结论 |
| --- | --- |
| 相同逐点剂量，编码与完整来源标签观测 | 编码信息不超过参照 |
| 相同总剂量，照明位置和分配不同 | 需分别计算信息矩阵 |
| 相同采集时间，曝光数与切换开销不同 | 需加入功率、背景及时间模型 |
| 相同输入能量，相干入射态不同 | 需使用复场模型 |
| 接收孔径、模式或读出方式改变 | 需重新定义完整参照 |

在无额外约束的单参数情形，最优强度设计可能是把剂量集中到最敏感的点；多参数及未知干扰场景则需要照明多样性维持可辨识性。因此，复杂图案与精选点扫描应在同一可行集合中比较。

## 5. 相干波前整形的信息增强机制

### 5.1 复场模型

记入射模式复振幅为 $\boldsymbol e$，归一化使
$\|\boldsymbol e\|^2=N_{\mathrm{in}}$ 表示输入光子预算。观测场与理想直接强度计数为

$$
\boldsymbol E(\theta)=H(\theta)\boldsymbol e,\qquad
\lambda_q=|(H(\theta)\boldsymbol e)_q|^2.
$$

传播损耗和探测效率纳入 $H$。强度展开为

$$
\lambda_q=
\sum_a|H_{qa}|^2|e_a|^2+
\sum_{a\ne b}H_{qa}H_{qb}^*e_ae_b^*.
$$

第二项是模式间交叉干涉。改变入射相位可改变观测及其几何导数，因此逐个入射模式的强度扫描通常不足以构成相干设计的完整信息参照。

静态粗糙墙可以产生稳定散斑，散射本身不等同于相位随机性在每次观测中重新抽样。是否可使用非相干模型，应由光源相干性、带宽、散斑稳定性和探测积分方式决定。相量场 NLOS 的包络相位也需与光学载频相位区分。

### 5.2 最大回波态与最大信息态

最大回波输入优化的是

$$
\max_{\|\boldsymbol e\|^2=N_{\mathrm{in}}}
\boldsymbol e^\dagger H^\dagger H\boldsymbol e.
$$

对标量参数 $\theta$，记 $H_\theta=\partial_\theta H$。实际直接强度 Poisson 读出的信息为

$$
\boxed{
F^{\mathrm{int}}_{\theta\theta}(\boldsymbol e)
=
\sum_q
\frac{
4\operatorname{Re}[(H\boldsymbol e)_q^*(H_\theta\boldsymbol e)_q]^2
}{
|(H\boldsymbol e)_q|^2
}.
}
$$

两个目标一般不同：亮度由 $H$ 决定，几何响应还取决于 $H_\theta$ 以及变化如何投影到可观测强度上。

在正均值模式及常规可微条件下，

$$
F^{\mathrm{int}}_{\theta\theta}(\boldsymbol e)
\le
4\boldsymbol e^\dagger H_\theta^\dagger H_\theta\boldsymbol e
\le
4N_{\mathrm{in}}\|H_\theta\|_2^2.
$$

这是给定场模型和输入预算下的上界，直接强度读出未必可达。若参数只改变输出全局相位，强度信息可以为零；引入相位参考后才可能读出该变化。

### 5.3 理想相位读出与有限模式增益

在相干态量子噪声限制、已知相位参考及恰当 homodyne 象限读出条件下，信息可达到

$$
F^{\mathrm{phase}}_{\theta\theta}(\boldsymbol e)
=
4\boldsymbol e^\dagger K_\theta\boldsymbol e,\qquad
K_\theta=H_\theta^\dagger H_\theta.
$$

允许任意复振幅控制时，$K_\theta$ 的最大特征向量给出局部最优输入。这是 maximum information states 框架的核心关系 [3]。其读出条件应与公式一起使用；普通 SPAD 直方图应按实际计数模型重新计算。

若有效可控输入模式数为 $M$，以各向同性随机单位输入的平均信息为基准，则

$$
\mathbb E[F_{\mathrm{rand}}]
=\frac{4N_{\mathrm{in}}}{M}\operatorname{tr}K_\theta,\qquad
F_{\mathrm{opt}}=4N_{\mathrm{in}}\lambda_{\max}(K_\theta),
$$

因此当 $K_\theta\ne0$ 时，

$$
\boxed{
G_{\mathrm{ideal}}
=
\frac{F_{\mathrm{opt}}}{\mathbb E[F_{\mathrm{rand}}]}
=
\frac{M\lambda_{\max}(K_\theta)}{\operatorname{tr}K_\theta},
\qquad 1\le G_{\mathrm{ideal}}\le M.
}
$$

信息在模式间分布均匀时增益接近 $1$；信息集中于少数模式时增益较大。这里的 $M$ 是独立可控光学模式数，且基准是指定随机输入的平均性能。该倍数界不自动适用于 phase-only 控制或直接强度读出。推导及取等条件见附录 C。

### 5.4 光子数量与每光子信息

对归一化分布
$\lambda_i(\theta)=N(\theta)p_i(\theta)$，
$\sum_i p_i=1$，有

$$
\boxed{
F_{\theta\theta}
=
\frac{[N'(\theta)]^2}{N(\theta)}
+
N(\theta)\sum_i
\frac{[\partial_\theta p_i(\theta)]^2}{p_i(\theta)}.
}
$$

第一项来自总计数变化，第二项来自光子在观测通道中的分布变化。由此可见，相同总检测计数并不决定相同信息。附录 D 给出一个输入和输出总光子数均相同、但直接强度信息不同的两模式例子。

实验评价应同时报告固定输入预算下的信息增益和回波计数，并在适当的归一化或重新配置预算后比较每光子信息。若对计数作条件化分析，应明确这会去除总计数携带的信息。

## 6. 连续膨胀与实际接收方式

### 6.1 曝光积分

第 $r$ 次曝光从 $s_r$ 开始、持续 $\Delta_r$，动态强度模型为

$$
\lambda_{rdk}(\boldsymbol\theta)
=
\int_{s_r}^{s_r+\Delta_r}
\sum_p I_{rp}(s)\,
h_{pdk}(\boldsymbol\theta;s)\,ds+b_{rdk},
$$

其中 $I_{rp}$ 是剂量率。静态模型中仅需记录其积分；动态模型中，照明发生的时刻也影响数据。

若单个时间核在曝光内线性漂移，
$\mu(s)=\mu_0+\dot\mu s$，且幅度、宽度固定，则平均时间核的均值与方差为

$$
\mathbb E[\tau]=\mu_0+\dot\mu\Delta/2,\qquad
\operatorname{Var}(\tau)=\sigma^2+(\dot\mu\Delta)^2/12.
$$

这是精确矩关系；用 Gaussian 表示该平均核则是矩匹配近似。对于固定墙面点，

$$
\dot\mu=\frac{(\boldsymbol u_p+\boldsymbol u_d)^T\dot{\boldsymbol x}}{c}.
$$

宽度变化、运动展宽及系统 IRF 可能互相混淆，应纳入联合参数分析。

### 6.2 脉冲与接收积分

短脉冲下，窄带复场矩阵可能需要扩展为频率相关模型：

$$
E_q(\tau;\theta)
=
\int [H(\omega;\theta)\boldsymbol e(\omega)]_q
e^{-i\omega\tau}\frac{d\omega}{2\pi},
$$

$$
\lambda_{qk}
=
\eta_q\int_{t_k}^{t_{k+1}}
|E_q(\tau;\theta)|^2\,d\tau+b_{qk}.
$$

SPAD 可测量由干涉转换成的强度变化，但空间模式混合、光谱平均及时间分箱可能降低可用信息。单 SPAD、阵列和带参考光的相位敏感接收需要分别建模。

### 6.3 标定与稳定时间

相干整形的有效收益还取决于标定和更新速度。需要测量散射矩阵或相关导数的稳定时间，并判断标定、优化与采集能否在其内完成。

若传播可分为固定墙面传输与随目标变化的响应，可考虑局部更新或分离标定；这一分解需要模型和实验验证。目标形变还可能引起散斑失相关，降低基于旧状态设计的波前性能。

## 7. 初步数值验证

### 7.1 实验设置

已建立独立 CPU 测试，采用二维截面中的单个静态散射高斯、9 个墙面照明点和固定探测点。未知参数为横向位置、深度及对数散射强度。时间核标准差为 $60\,\mathrm{ps}$，时间 bin 宽度为 $16\,\mathrm{ps}$。

测试对时间密度作 bin 积分，使用解析 Jacobian 计算 Poisson Fisher 信息，并用有限差分独立核对导数。比较对象为完整点扫描与两幅左右互补的非负图案，二者保持逐点总剂量相同。

验证内容包括：one-hot 取等、30 组随机非负编码的信息矩阵顺序、精确损失恒等式、导数抵消反例，以及未知强度下的几何 CRLB。两种设计各进行 $100$ 次 Poisson 随机观测和局部最大似然估计。

### 7.2 结果与解释

在固定随机种子及相同入射剂量下，两种设计均产生约 $6573$ 个期望信号光子：

| 设计 | 横向 CRLB 标准差 | 横向经验标准差 | 深度 CRLB 标准差 | 深度经验标准差 |
| --- | ---: | ---: | ---: | ---: |
| 9 点完整扫描 | 0.659 mm | 0.655 mm | 0.129 mm | 0.118 mm |
| 两幅互补编码，$\alpha=0.8$ | 1.446 mm | 1.489 mm | 0.235 mm | 0.229 mm |

解析导数与有限差分的相对误差小于 $4\times10^{-9}$；信息上限和损失恒等式检查均通过；$200$ 次拟合均成功。经验标准差接近对应理论下界，有限重复次数及估计偏差可导致局部数值波动。

这一结果验证了指定静态计数模型中的信息混合关系。当前实验尚未包含 Hadamard、最优图案搜索、连续形变或相干波前控制。两幅图案在较少曝光下保留了部分几何信息，但本测试未计入曝光时长、切换开销与动态畸变，因此尚不能评价实际信息获取速率。

图像与机器可读结果保存在配套实验目录，复现入口见附录 E。

## 8. 拟开展的研究

### 8.1 核心假设

> 针对半径或膨胀速度设计的入射场，相比随机入射和最大回波入射，可在给定资源约束下提高有效几何 Fisher 信息；其收益可能同时来自更多回波光子和更高的每光子信息。

该假设需要通过实际读出模型、几何耦合和标定更新代价共同检验。

### 8.2 理论与数值验证

建议两条建模工作并行推进：

| 模型 | 主要任务 | 预期产出 |
| --- | --- | --- |
| 非相干强度模型 | 加入半径与速度，比较精选扫描、随机、Hadamard 和优化图案 | 受限曝光与剂量下的信息保留比例 |
| 相干复场模型 | 比较随机态、最大回波态、信息优化态及 phase-only 约束 | 实际读出信息与理想场信息上界的差距 |

非相干模型用于量化资源受限采样的收益；相干模型直接检验波前整形的敏感模式选择。后一模型应从具有已知导数的小系统开始，再引入粗糙墙、波长、孔径和气球几何。随机散射矩阵可验证代数关系，但几何可行性需要物理传播模型或标定数据支持。

在强度编码模型下，针对膨胀速度可采用

$$
W^*
=
\arg\min_{W\in\mathcal C}
[F(\boldsymbol\theta_0;W)^{-1}]_{v_Rv_R},
$$

其中可行集合 $\mathcal C$ 包含图案数、总剂量、功率及时间限制。相干模型则以各次曝光的入射复振幅或 SLM 相位为设计变量，使用实际接收方式对应的 Fisher 信息。

多参数场景可采用 D-optimal、E-optimal 或 A-optimal 准则，并对未知状态范围评估稳健性。涉及最小特征值或方差求和时，应先按任务尺度归一化参数，避免长度、速度和增益的单位选择影响设计结果。

### 8.3 实验步骤

1. **静态受控扰动。** 选择两个可重复的相近位置或形状状态，估计几何导数及观测噪声，检查局部 Fisher 模型。
2. **分阶段膨胀。** 采用充气后保持静止的状态序列，利用外部轮廓或其他测量提供真值，验证半径信息及状态依赖性。
3. **连续膨胀。** 记录曝光起止时刻、同步信号与外部真值，加入曝光积分和波前更新。
4. **扩展形变与光学属性。** 在主导膨胀可辨识后，增加局部形变、壁厚及透射等因素。

### 8.4 比较基准与评价量

所有方案应说明共同的资源约束，并报告以下指标：

- 输入光子预算、期望回波计数、接收模式数与采集时间；
- 半径和速度的有效 Fisher 信息、CRLB、偏差与经验误差；
- 相同状态下对未知增益、时间零点和系统响应的稳健性；
- 标定与更新成本，以及波前性能随形变失相关的退化。

完整标签观测用于判断信息损失，同资源的可实现扫描用于评价实际收益；两者应分别列出。

### 8.5 与 3D-GTR 的结合

若标定得到的有效强度照明场足以预测计数，可将其接入 3D-GTR 的几何瞬态模型。若交叉干涉及其参数变化对测量至关重要，则需要增加复场或混合前向模型。

高斯表示可继续承担几何参数化，关键是照明到观测的映射能否准确表达并求导。相干效应的接入形式应由物理模型和实验数据确定。

## 9. 可行性判断所需的关键条件

| 条件 | 需要确认的内容 |
| --- | --- |
| 光场控制 | SLM 共轭平面、相位/振幅能力、有效独立模式数 |
| 光源 | 波长、脉宽、带宽、重复率及功率限制 |
| 接收 | 单 SPAD、阵列或相位敏感读出；空间与时间积分方式 |
| 标定 | 墙面或完整散射矩阵是否可测，参数导数如何获得 |
| 动态稳定性 | 场相关性、几何变化速度和局部设计有效范围 |
| 资源核算 | 输入预算是否包含标定；参考光与切换时间如何计入 |
| 几何真值 | 膨胀状态的可重复性、同步精度与外部测量误差 |

其中优先需要确定的是接收方式与散射矩阵导数的获取途径。它们决定 maximum information states 的场信息优势有多少能转化为实际 NLOS 计数信息。

## 10. 阶段性结论

波前整形用于增强非视域形变测量具有明确的理论切入点。对非相干编码，固定逐点剂量下的完整来源标签观测给出严格参照，编码设计主要解决有限资源内的信息保留问题。对相干波前整形，入射态可以改变干涉及几何导数，信息上限由散射矩阵导数、可控模式与实际读出共同决定。

当前验证已建立非相干模型的信息评价基础。下一步的关键是构造与气球半径或表面位移相联系的复场响应，并比较最大回波与最大几何信息设计。是否存在实际优势，应由固定资源下的估计精度以及标定、更新的总成本判断。

---

## 附录 A：瞬态核、几何导数与统计推导

### A.1 归一化时间核与计数单位

归一化 Gaussian 密度为

$$
f(\tau;\mu,\sigma)
=
\frac{1}{\sqrt{2\pi}\sigma}
\exp\!\left[-\frac{(\tau-\mu)^2}{2\sigma^2}\right].
$$

若 $N$ 为完整时间轴的期望计数，则 bin 计数为

$$
h_k=N\int_{t_k}^{t_{k+1}}f(\tau;\mu,\sigma)\,d\tau
=
N\left[
\Phi\!\left(\frac{t_{k+1}-\mu}{\sigma}\right)
-\Phi\!\left(\frac{t_k-\mu}{\sigma}\right)
\right].
$$

对归一化核，

$$
\partial_a\log(Nf)
=
\partial_a\log N+
\frac{\tau-\mu}{\sigma^2}\partial_a\mu+
\left[
\frac{(\tau-\mu)^2}{\sigma^3}-\frac1\sigma
\right]\partial_a\sigma.
$$

若用峰高 $A$ 表示非归一化 Gaussian，则连续时间下
$N=A\sqrt{2\pi}\sigma$。固定峰高与固定光子数属于不同约束，计数预算比较应保持一致。

宽度导数可从 $Q=\boldsymbol v^T\Sigma\boldsymbol v$ 得到：

$$
\partial_a\sigma
=\frac{\partial_aQ}{2c\sqrt Q},\qquad
\partial_aQ
=
2(\partial_a\boldsymbol v)^T\Sigma\boldsymbol v
+\boldsymbol v^T(\partial_a\Sigma)\boldsymbol v.
$$

固定墙面点时，

$$
\partial_a\boldsymbol u
=
\frac{I-\boldsymbol u\boldsymbol u^T}{d}
\partial_a\boldsymbol x.
$$

### A.2 Fisher 信息推导

对独立 Poisson 观测，

$$
\partial_a\ell
=
\sum_i
\frac{y_i-\lambda_i}{\lambda_i}\partial_a\lambda_i.
$$

由

$$
\mathbb E[(y_i-\lambda_i)(y_j-\lambda_j)]
=\delta_{ij}\lambda_i
$$

得到

$$
F_{ab}
=
\sum_{i,j}
\frac{\partial_a\lambda_i\partial_b\lambda_j}
{\lambda_i\lambda_j}
\mathbb E[(y_i-\lambda_i)(y_j-\lambda_j)]
=
\sum_i\frac{\partial_a\lambda_i\partial_b\lambda_i}{\lambda_i}.
$$

已知背景 $b_i$ 与参数无关时，

$$
F_{ab}
=
\sum_i
\frac{\partial_a\lambda_{\mathrm{sig},i}
\,\partial_b\lambda_{\mathrm{sig},i}}
{\lambda_{\mathrm{sig},i}+b_i}.
$$

显著 pile-up、dead time 或相关计数需要重新建立似然。

### A.3 干扰参数的投影解释

将白化 Jacobian 按列写成
$\widetilde J=[\widetilde J_g,\widetilde J_n]$，则

$$
F_{g|n}
=
\widetilde J_g^T(I-P_n)\widetilde J_g,\qquad
P_n=\widetilde J_n
(\widetilde J_n^T\widetilde J_n)^{-1}
\widetilde J_n^T.
$$

它保留无法被干扰导数解释的几何响应。若干扰信息块奇异，应在可辨识子空间中处理；数值正则或额外先验应与数据本身的信息分开。

## 附录 B：非相干编码上限的两种证明

### B.1 方向二次型证明

令 $\boldsymbol g_p=\nabla h_{pdk}$，
$z_p=\boldsymbol v^T\boldsymbol g_p$。对每个编码观测，Cauchy–Schwarz 不等式给出

$$
\begin{aligned}
\left(\sum_pw_{rp}z_p\right)^2
&=
\left(
\sum_p\sqrt{w_{rp}h_{pdk}}\,
\frac{\sqrt{w_{rp}}z_p}{\sqrt{h_{pdk}}}
\right)^2\\
&\le
\left(\sum_pw_{rp}h_{pdk}\right)
\left(\sum_pw_{rp}\frac{z_p^2}{h_{pdk}}\right).
\end{aligned}
$$

除以编码均值并对 $r,d,k$ 求和，得到

$$
\boldsymbol v^TF_{\mathrm{code}}\boldsymbol v
\le
\boldsymbol v^TF_{\mathrm{sep}}\boldsymbol v.
$$

这对任意方向成立，因此差矩阵为半正定。假设为正均值、可微响应、固定非负剂量、同一静态参数及独立 Poisson 计数。

### B.2 精确损失证明

对固定 $r,d,k$，分别观测与混合观测的信息贡献为

$$
F_{\mathrm{sep}}^{(rdk)}
=\sum_p a_p\boldsymbol s_p\boldsymbol s_p^T,\qquad
F_{\mathrm{code}}^{(rdk)}
=\Lambda\bar{\boldsymbol s}\bar{\boldsymbol s}^T,
\quad \Lambda=\sum_p a_p.
$$

展开中心化外积：

$$
\sum_p a_p(\boldsymbol s_p-\bar{\boldsymbol s})
(\boldsymbol s_p-\bar{\boldsymbol s})^T
=
\sum_p a_p\boldsymbol s_p\boldsymbol s_p^T
-\Lambda\bar{\boldsymbol s}\bar{\boldsymbol s}^T.
$$

累加各观测即可得到第 4 节的损失恒等式。每项均为非负权重乘外积，因而半正定。

也可构造潜在计数

$$
z_{rpdk}\sim\operatorname{Poisson}(w_{rp}h_{pdk}),\qquad
y_{rdk}=\sum_pz_{rpdk}.
$$

从完整来源计数 $z$ 到混合计数 $y$ 的处理与未知参数无关，因此信息不增加。若场景随时间变化，理想参照还需保留相同照明日程中的时间标签；另一个更慢的扫描日程不能直接视为同一参照。

### B.3 有效几何信息与零空间

在干扰信息块可逆时，

$$
\boldsymbol v^TF_{g|n}\boldsymbol v
=
\min_{\boldsymbol u}
\begin{bmatrix}\boldsymbol v\\\boldsymbol u\end{bmatrix}^T
F
\begin{bmatrix}\boldsymbol v\\\boldsymbol u\end{bmatrix}.
$$

完整二次型的顺序在取最小值后保持，因此

$$
F_{\mathrm{code},g|n}
\preceq F_{\mathrm{sep},g|n}.
$$

将静态响应展平，局部模型为

$$
\delta\boldsymbol\lambda
\approx
(W\otimes I_{DK})J_h\,\delta\boldsymbol\theta.
$$

故

$$
\operatorname{rank}((W\otimes I_{DK})J_h)
\le
\min\{\operatorname{rank}J_h,\operatorname{rank}(W)DK\}.
$$

原始完整响应的一阶零空间无法由这种线性混合消除。

## 附录 C：相干信息上界与理想增益

### C.1 直接强度读出的上界

固定入射态 $\boldsymbol e$ 求导：

$$
\partial_\theta\lambda_q
=
2\operatorname{Re}[(H\boldsymbol e)_q^*
(H_\theta\boldsymbol e)_q].
$$

利用 $\operatorname{Re}(z)^2\le |z|^2$，

$$
\begin{aligned}
F^{\mathrm{int}}_{\theta\theta}
&\le4\sum_q|(H_\theta\boldsymbol e)_q|^2\\
&=4\boldsymbol e^\dagger H_\theta^\dagger H_\theta\boldsymbol e\\
&\le4N_{\mathrm{in}}\lambda_{\max}(H_\theta^\dagger H_\theta).
\end{aligned}
$$

第一步取等需要各模式中的变化满足相位条件，最后一步取等需要输入匹配最大特征向量。两项条件同时成立时，直接强度读出才可能达到这一界。严格零均值模式涉及非正则性，需要单独分析。

### C.2 理想相位读出下的相对增益

令 $K_\theta=H_\theta^\dagger H_\theta$，
$\boldsymbol e=\sqrt{N_{\mathrm{in}}}\boldsymbol u$，
$\|\boldsymbol u\|=1$。理想相位读出给出

$$
F(\boldsymbol u)
=
4N_{\mathrm{in}}\boldsymbol u^\dagger K_\theta\boldsymbol u.
$$

各向同性随机单位输入满足
$\mathbb E[\boldsymbol u\boldsymbol u^\dagger]=I/M$，因此

$$
\mathbb E[F]
=\frac{4N_{\mathrm{in}}}{M}\operatorname{tr}K_\theta.
$$

最优值为 $4N_{\mathrm{in}}\lambda_{\max}(K_\theta)$。由

$$
\lambda_{\max}(K_\theta)
\le\operatorname{tr}K_\theta
\le M\lambda_{\max}(K_\theta)
$$

得到 $1\le G_{\mathrm{ideal}}\le M$。非零秩一矩阵给出最大增益 $M$；各特征值相等时增益为 $1$。

这是指定读出、完整复振幅控制及随机平均基准下的关系。phase-only 输入、有限接收模式、背景和未知干扰均需另外处理。相位参考也是实验资源，应明确光子预算是否包含本振及标定。

## 附录 D：两个辅助例子

### D.1 相同总光子数下的信息差异

构造无损两模式系统

$$
H(\theta)
=
\frac1{\sqrt2}
\begin{bmatrix}1&1\\1&-1\end{bmatrix}
\begin{bmatrix}e^{i\theta}&0\\0&1\end{bmatrix}.
$$

所有输入均满足
$\|H\boldsymbol e\|^2=\|\boldsymbol e\|^2=N$。

取
$\boldsymbol e_A=\sqrt N[1,0]^T$，
两个输出均值始终为 $N/2$，所以 $F_A=0$。

取
$\boldsymbol e_B=\sqrt{N/2}[1,i]^T$，则

$$
\lambda_{1,2}=\frac N2(1\pm\sin\theta).
$$

在 $\theta=0$，

$$
F_B
=
\frac{(N/2)^2}{N/2}
+
\frac{(-N/2)^2}{N/2}
=N.
$$

两种输入能量和总输出计数相同，在参考点的逐通道均值也相同，但导数不同。该例说明相干控制可以改变每光子信息；它是理想参数测量例子，真实几何收益仍需通过 $H(\theta)$ 与半径或位移的关系建立。

### D.2 固定宽度单峰测距的特例

设 $N$ 个光子的到达时间独立服从
$\mathcal N(\mu(q),\sigma_t^2)$，仅峰位依赖位移 $q$，则

$$
F_{qq}=\frac{N[\mu'(q)]^2}{\sigma_t^2}.
$$

对单位方向上的点位移，
$|\mu'|\le2/c$，故

$$
\operatorname{std}(\hat q)
\ge\frac{c\sigma_t}{2\sqrt N}.
$$

该界仅适用于固定计数、固定宽度和纯峰位模型。计数率、宽度或相干相位携带额外信息时，应重新计算完整 Fisher 信息。低维位置估计精度也应与任意三维结构的空间分辨率区分。

## 附录 E：数值验证与复现材料

- [数学测试程序](nlos-3d-gtr/math_tests/fisher_coding.py)
- [模型、推导及运行说明](nlos-3d-gtr/math_tests/README.md)
- [机器可读实验结果](nlos-3d-gtr/math_tests/results/report.json)
- [信息保留比例与 CRLB 图](nlos-3d-gtr/math_tests/results/information_bound.png)

结果对应随机种子 $20261026$、总入射剂量 $2\times10^6$ 个统一仿真单位，每种设计重复 $100$ 次。该剂量尚未与真实系统输入光子数标定。

程序使用解析 Jacobian 和独立有限差分检查，不依赖 GPU。局部估计从真值附近初始化，用于验证信息分析，而非评估全局重建能力。结果目录未纳入 Git；缺少输出文件时，可按运行说明重新生成。

## 参考文献

[1] Wang, Y., Zhan, Z., et al. Non-line-of-sight imaging with arbitrary relay surface geometries via 3D Gaussian Transient Rendering. *SIGGRAPH Conference Papers*, 2026. [DOI](https://doi.org/10.1145/3799902.3811137)；[预印本](https://arxiv.org/abs/2606.21270)；[代码](https://github.com/YWnlos/nlos-3d-gtr)。

[2] Cao, R., de Goumoens, F., Blochet, B., Xu, J., and Yang, C. High-resolution non-line-of-sight imaging employing active focusing. *Nature Photonics*, 16, 462–468, 2022. [DOI](https://doi.org/10.1038/s41566-022-01009-8)。

[3] Bouchet, D., Rotter, S., and Mosk, A. P. Maximum information states for coherent scattering measurements. *Nature Physics*, 17, 564–568, 2021. [DOI](https://doi.org/10.1038/s41567-020-01137-4)；[修订预印本](https://arxiv.org/abs/2002.10388v3)。期刊另附 2021 年作者更正，采用修订版本阅读。

[4] Bouchet, D., Seifert, J., and Mosk, A. P. Optimizing illumination for precise multi-parameter estimations in coherent diffractive imaging. *Optics Letters*, 46, 254–257, 2021. [DOI](https://doi.org/10.1364/OL.411339)；[预印本](https://arxiv.org/abs/2010.03306)。

[5] Scotté, C., Galland, F., and Rigneault, H. Photon-noise: Is a single-pixel camera better than point scanning? A signal-to-noise ratio analysis for Hadamard and Cosine positive modulation. *Journal of Physics: Photonics*, 2023. [DOI](https://doi.org/10.1088/2515-7647/acc70b)；[预印本](https://arxiv.org/abs/2210.01915)。

[6] Bouchet, D., and Bossy, E. Temporal shaping of wave fields for optimally precise measurements in scattering environments. *Physical Review Research*, 5, 013144, 2023. [DOI](https://doi.org/10.1103/PhysRevResearch.5.013144)。

[7] Gutiérrez-Cuevas, R., Bouchet, D., de Rosny, J., and Popoff, S. M. Reaching the precision limit with tensor-based wavefront shaping. *Nature Communications*, 2024. [DOI](https://doi.org/10.1038/s41467-024-50513-8)；[早期预印本](https://arxiv.org/abs/2310.17516)。

[8] Guo, Y., Lei, Y., Pu, M., et al. Vectorial Digitelligent Optics for High-Resolution Non-Line-of-Sight Imaging. *Engineering*, 45, 70–78, 2025. [DOI](https://doi.org/10.1016/j.eng.2024.11.013)。

本文为针对性文献调研与模型分析。理论推导采用标准估计理论和波动模型；部分相关文献目前仅核对摘要及公开方法信息，其实验条件仍需进一步全文审查。
