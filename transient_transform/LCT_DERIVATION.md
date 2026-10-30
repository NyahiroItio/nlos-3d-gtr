# 抽象前向模型、连续 LCT 与原版离散实现

日期：2026-10-30。分支：`research/transient-transform`。

## 1. A 的地位与记号

保留用户的抽象模型

\[
\tau(t)=A\rho(t).
\]

这里的 A 是固定线性前向映射的记号，不要求显式构造矩阵，也不预设矩阵元素、积分核、辐射衰减或体素基函数。抽象不等于未知：求逆问题仍以该映射已知或能够作用为前提。本文另用 \(\mathcal A_{\mathrm{conf}}\) 表示在明确假设下选定的共焦模型，避免把这种实例当作 A 的定义。

记号如下：

- 慢时间 \(t\)：一次完整扫描对应的场景时刻；扫描内假设静止。
- 快时间 \(s\)：墙点的光子飞行时间。
- \(\mathbf w=(x',y',0)\)：共焦照明/探测墙点。
- \(\mathbf x=(x,y,z)\)，\(z>0\)：隐藏空间位置。
- \(\rho(x,y,z;t)\)：固定空间坐标中的有效散射密度。
- \(r=\|\mathbf x-\mathbf w\|\)。

仅由抽象等式不能推出存在 LCT 逆；LCT 的适用性来自选定前向模型的结构。若采用其他模型，A 仍是合法记号，但下文的分解未必成立。

## 2. 用于推导的一类具体模型

采用平面共焦扫描、隐藏区域单次散射、忽略遮挡和角度余弦项的理想模型，定义

\[
(\mathcal A_{\mathrm{conf}}\rho)(x',y',s;t)
=\kappa\iiint_{z>0}r^{-p}\rho(x,y,z;t)
\delta(2r-cs)\,dx\,dy\,dz. \tag{1}
\]

\(\kappa\) 是已知、固定的整体尺度系数；\(p=4\) 对应论文的漫反射模型，\(p=2\) 对应其理想逆反射模型。其他已知、固定的幂次也能沿同一路径推导。若使用单位积分的时间核 \(\delta(s-2r/c)\)，有

\[
\delta(s-2r/c)=c\,\delta(2r-cs),
\]

因此需要一致调整整体尺度，不能混用这两种约定后再比较绝对幅值。

式 (1) 是本文为研究 LCT 而选定的模型实例，不是对抽象 A 的强制解释。论文将未知量称为 albedo；在本项目中把它解释为有效散射密度，需要一直维持相同的体积积分和离散系数约定。

## 3. Delta 变换与平方坐标

令

\[
u=z^2,\qquad v=(cs/2)^2.
\]

对 \(r>0,s>0\)，Delta 的变换恒等式为

\[
\delta(2r-cs)=r\,\delta(r^2-v). \tag{2}
\]

证明：令 \(a=cs/2>0\)，则

\[
\delta(2r-cs)=\tfrac12\delta(r-a),\qquad
r\delta(r^2-a^2)=\frac{r}{2a}\delta(r-a)=\tfrac12\delta(r-a).
\]

代入式 (1)：

\[
\tau(x',y',s;t)
=\kappa\iiint r^{1-p}\rho(x,y,z;t)
\delta\!\left((x'-x)^2+(y'-y)^2+z^2-v\right)dx\,dy\,dz.
\]

在 Delta 的支集上 \(r=\sqrt v\)，因而

\[
\tau(x',y',2\sqrt v/c;t)
=\kappa v^{(1-p)/2}\iiint\rho(x,y,z;t)
\delta\!\left((x'-x)^2+(y'-y)^2+z^2-v\right)dx\,dy\,dz.
\]

再用 \(dz=du/(2\sqrt u)\)，得到

\[
\frac{v^{(p-1)/2}}{\kappa}\tau(x',y',2\sqrt v/c;t)
=\iiint\frac{\rho(x,y,\sqrt u;t)}{2\sqrt u}
\delta\!\left(v-u-(x'-x)^2-(y'-y)^2\right)dx\,dy\,du. \tag{3}
\]

这里所有 Jacobian 都来自快时间或空间坐标变换，没有对慢时间求导。

## 4. 因式分解与连续未知量的含义

定义两个坐标变换

\[
(R_s\tau)(x',y',v;t)
=\frac{v^{(p-1)/2}}{\kappa}\tau(x',y',2\sqrt v/c;t), \tag{4}
\]

\[
(R_z\rho)(x,y,u;t)
=\frac{\rho(x,y,\sqrt u;t)}{2\sqrt u}. \tag{5}
\]

定义

\[
g=R_s\tau,\qquad q=R_z\rho,\qquad
h(\Delta x,\Delta y,w)=\delta(w-\Delta x^2-\Delta y^2).
\]

将 \(q\) 在 \(u<0\) 延拓为零，式 (3) 就是三维卷积

\[
g=h*q=Hq,\qquad \boxed{R_s\tau=HR_z\rho}. \tag{6}
\]

在理想连续模型及相应函数空间中，

\[
\boxed{\mathcal A_{\mathrm{conf}}=R_s^{-1}HR_z}. \tag{7}
\]

两个坐标变换的解析逆为

\[
(R_z^{-1}q)(x,y,z;t)=2z\,q(x,y,z^2;t), \tag{8}
\]

\[
(R_s^{-1}g)(x',y',s;t)
=\kappa(cs/2)^{1-p}g(x',y',(cs/2)^2;t). \tag{9}
\]

因此连续 LCT 中先求出的 q 是平方深度坐标下的加权密度；乘回 \(2z\) 并映射到 z 后，才恢复式 (1) 中的密度 \(\rho\)。此处没有换成材料点位置或速度作为未知量。

## 5. 非迭代逆与分辨率算子

记 \(\mathcal F\) 为三维 Fourier 变换。形式上

\[
\widehat g=\widehat h\,\widehat q.
\]

在 \(\widehat h\ne0\) 的频率上可作除法；若存在零响应或很小的响应，直接除法会丢失信息或放大扰动。对固定 \(\lambda>0\)，采用变换域 Tikhonov 目标

\[
\min_q\|Hq-g\|_2^2+\lambda\|q\|_2^2
\]

得到直接滤波解

\[
\widehat q_\lambda
=\frac{\overline{\widehat h}}{|\widehat h|^2+\lambda}\,\widehat g,
\]

以及

\[
\boxed{L_\lambda
=R_z^{-1}\mathcal F^{-1}
\left[\frac{\overline{\widehat h}}{|\widehat h|^2+\lambda}\right]
\mathcal F R_s}. \tag{10}
\]

这是固定线性、非迭代的重建算子。在满足对应随机过程假设时，该滤波形式也可解释为 Wiener 滤波；不能因此把原工程的 snr 参数直接当作实测光子计数 SNR。

对无噪声理想数据 \(\tau=\mathcal A_{\mathrm{conf}}\rho\)，

\[
\hat\rho=Q_\lambda\rho,
\]

\[
\boxed{Q_\lambda
=R_z^{-1}\mathcal F^{-1}
\left[\frac{|\widehat h|^2}{|\widehat h|^2+\lambda}\right]
\mathcal F R_z}. \tag{11}
\]

因此有限正则化会保留一个确定的分辨率算子，通常 \(Q_\lambda\ne I\)。只有在可观测频率、相容的边界条件和相应逆存在的条件下，才可以讨论 \(\lambda\to0\) 时的精确恢复。

式 (10)-(11) 描述理想完整域，或与 Fourier 卷积严格相容的离散模型。有限墙面、有限快时间窗口、IRF 和分箱采样都应另行建模。例如可写成

\[
A=\mathcal P\mathcal K\mathcal A_{\mathrm{conf}},
\]

其中 \(\mathcal K\) 为仪器响应，\(\mathcal P\) 为采样与有限域观测。此时一般不能直接把 \(L_\lambda A\) 化成式 (11)。

## 6. 原版代码的精确离散表达

下面只分析当前已经与 MATLAB 对齐的有符号核心，输入为准备后的原生测量数组，不包含最后的非负截断、显示翻转和裁剪。

设

- \(R\)：代码中的 mtx，沿快时间/深度维作用；
- \(W_p\)：原生归一化快时间的 p 次方对角权重；
- \(Z\)：将原生体积补零到双倍尺寸；\(C\)：裁回原生体积；
- \(\mathcal F_d\)：填充网格上的离散 Fourier 变换；
- \(\hat h_d\)：原版归一化 PSF 的频谱。

当前实现严格等价于

\[
\boxed{L_{\mathrm{code}}
=R^{\mathsf T}C\mathcal F_d^{-1}
\operatorname{diag}\!\left(
\frac{\overline{\hat h_d}}{|\hat h_d|^2+1/\mathrm{snr}}
\right)\mathcal F_d ZRW_p}. \tag{12}
\]

此式可以直接从 original_lct.py 的各阶段读出。R 的元素为

\[
R_{ij}=\frac1M\sum_{k=(i-1)M+1}^{iM}
\frac{\mathbf 1_{\{\lceil\sqrt{k}\rceil=j\}}}{\sqrt{k}},
\qquad 1\le i,j\le M. \tag{13}
\]

矩阵 mtxi 等于 \(R^{\mathsf T}\)，并非通过求逆获得。数值检查使用当前代码的 float64 矩阵及 NumPy 默认数值秩判据，得到：

| M | 数值秩 | \(R^{\mathsf T}R=I\) |
| ---: | ---: | --- |
| 8 | 6 | 否 |
| 16 | 12 | 否 |
| 64 | 48 | 否 |
| 512 | 384 | 否 |

所以在整个原生离散系数空间上，R 没有普通双侧逆，不能把代码变量 mtxi 的名字当作 \(R^{-1}\) 的数学证明。

进一步，由输出端的转置重采样可得

\[
\operatorname{rank}(L_{\mathrm{code}})
\le N^2\operatorname{rank}(R).
\]

按 M=512 的上述数值秩，整个有符号重建算子的数值秩至多是原生体素空间维数的 3/4。因此它不能在任意原生体素系数张量上构成严格左逆；对平滑场景或限定子空间的近似重建能力仍需另外评价。这是离散实现的性质，不否定理想连续模型的因式分解。


### 6.1 为什么代码乘 s 的四次方，而连续公式只有 v 的三次方根权重

在归一化坐标上，忽略有限分箱和端点误差，R 对平滑函数的作用近似为

\[
(Rf)(v)\approx\frac1M\frac{f(\sqrt v)}{\sqrt v}.
\]

因此

\[
RW_pf\approx\frac1M v^{(p-1)/2}f(\sqrt v).
\]

p=4 给出 \(v^{3/2}\)，p=2 给出 \(v^{1/2}\)，与连续推导相符。全局常数还受坐标归一化、PSF 归一化、网格步长和曝光标定影响。

### 6.2 为什么转置不能直接叫作连续逆

为了单独说明 Jacobian，令

\[
(Jf)(u)=\frac{f(\sqrt u)}{2\sqrt u},\qquad 0<u<1.
\]

对支集远离零点的函数，在普通 \(L^2(dz)\)、\(L^2(du)\) 内积下，换元可得

\[
\langle Jf,g\rangle_{du}
=\int_0^1 f(z)\overline{g(z^2)}\,dz,
\]

故

\[
(J^*g)(z)=g(z^2),\qquad
(J^{-1}g)(z)=2z\,g(z^2). \tag{14}
\]

两者不同。若在这种点值连续解释下，把理想 q 送进伴随而非逆，输出会是 \(q(z^2)=\rho(z)/(2z)\)。这是解释 Jacobian 差别的条件性示例，不能不加系数约定就把实际 vol 一律断言成精确的 \(\rho/(2z)\)。

实际离散数组可能代表点值、单元平均或单元积分，不同的系数定义和内积会改变离散伴随与尺度的解释。论文补充式 (17)-(18) 使用单元积分定义，因此未来实现匹配模拟器前，必须明确本项目采用的体素系数定义。

目前严格成立的结论是：连续推导中的 \(R_z^{-1}\) 恢复模型密度；原版代码严格输出 \(L_{\mathrm{code}}\tau\)。二者不能仅凭工程逐阶段对齐而直接等同为任意物体的精确、绝对标定密度。

如果希望把某个加权体积作为新的未知量，也可以进行合法的变量代换 \(\tilde\rho=D\rho\)，并相应写成 \(\tau=(AD^{-1})\tilde\rho\)。这改变了未知量的定义，需要明确声明。

## 7. 慢时间导数与参考状态

在 A、坐标变换、几何标定与正则化参数均不随慢时间变化的条件下，

\[
\partial_t\tau=A\partial_t\rho.
\]

对任意固定线性重建算子 L，

\[
\partial_t(L\tau)=L\partial_t\tau.
\]

令 \(Q=LA\)，则

\[
\boxed{\hat\rho=Q\rho,\qquad
\widehat{\partial_t\rho}=Q\partial_t\rho}. \tag{15}
\]

两者共用相同的空间分辨率算子。导数的正负值不会改变线性逆的适用性；非负截断、取模以及依赖当前数据重新调节滤波参数的操作不属于这个固定线性核心。

若进一步在参考时刻附近假设

\[
\rho(t)=\rho_0+(t-t_0)u,
\]

则

\[
\tau(t)=\tau_0+(t-t_0)d,\qquad
\tau_0=A\rho_0,\quad d=Au.
\]

对多次完整扫描的原始 transient 联合拟合截距与斜率，再分别使用同一个 L：

\[
\hat\rho_0=L\hat\tau_0,\qquad \hat u=L\hat d.
\]

这同时估计参考状态和密度变化率，不需要从纯导数中猜测积分常数。这里的 u 是固定坐标中的密度导数；材料点匀速运动一般不意味着 \(\partial_t^2\rho=0\)。

## 8. 结论与验证边界

1. A 可以始终是抽象前向算子的记号；LCT 是某类具有特殊结构的模型的直接逆方法。
2. 理想连续模型满足 \(\mathcal A_{\mathrm{conf}}=R_s^{-1}HR_z\)。正确恢复密度包含深度坐标的 Jacobian 补偿。
3. 固定正则化 LCT 恢复的是 \(Q\rho\)；同一算子用于慢时间导数时恢复 \(Q\partial_t\rho\)。
4. 原版 MATLAB/Python 对齐证明的是式 (12) 的数值复现，未证明 \(L_{\mathrm{code}}A=I\)。
5. 后续静态闭环应先固定体素系数、辐射模型、仪器响应、观测域和离散网格，再评价 \(L_{\mathrm{code}}A\)；不能只用矩阵转置代替逆的命名来跳过这一步。

## 来源

- O'Toole, Lindell, Wetzstein, *Confocal non-line-of-sight imaging based on the light-cone transform*, Nature 555, 338-341 (2018), DOI: [10.1038/nature25489](https://doi.org/10.1038/nature25489)。正文式 (1)-(3)。
- 同文 Supplementary Information，第 4-8 页，式 (6)、(8)、(12)-(22)。本地核对文本和完整公式页。
- [作者项目页与原版代码下载](https://davidlindell.com/publications/nlos-lct)。原版工程的固定校验值与实际 MATLAB 对齐记录见 [ORIGINAL_ALIGNMENT.md](ORIGINAL_ALIGNMENT.md)。
- 当前实现：[original_lct.py](original_lct.py)，尤其 resampling_matrices 和 OriginalLCT.__call__。

式 (1)-(11) 是按本项目 t/s 记号及一般衰减幂次整理的推导；式 (12)-(15)、转置/逆的区分和验证边界是对当前实现的分析，并非声称原论文作出了所有这些结论。
