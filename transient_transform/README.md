# 固定体素 transient 变换：非迭代数学 demo

本目录在 `research/transient-transform` 分支中独立实现，不导入 3D-GTR、Torch 或 Gaussian 场景组件。

## 当前推荐入口：已验证的原版工程

**原版 LCT 工程已在 MATLAB 和 Python 间逐阶段对齐。** 原始 S/U 与漫反射 S 的 64×64×512 输出相对差异约 1e-15，21 项测试通过。原版源文件未改动；MATLAB R2024b 兼容副本仅预声明隐式加载的 `width` 变量。

```powershell
python -m transient_transform.upstream
python -m transient_transform.demo
```

默认使用原版数据和原版工程约定。结果位于 `results/original_alignment/`，包括原生三维重建的三视图 PNG/SVG 和测量瞬态图。完整记录、原版来源、数值核验与运行方式见 **[原版工程对齐说明](ORIGINAL_ALIGNMENT.md)**。

下面保留早期 640 体素模拟实验，作为历史诊断。下文的 `lct` 指失败的连续公式原型，**不代表已对齐的原版 LCT**。两个旧后端必须显式选择 `--inverse spectral` 或 `--inverse lct`。

---

## 历史：固定体素合成实验

本版验证的链路是：**自行生成三维体素密度与完整共焦扫描 → 加入光子计数噪声 → 在慢时间上直接拟合参考 transient 和导数 → 用固定线性逆算子直接输出参考密度与密度导数 → 核对理论噪声传播。**

**早期实验后端：从连续公式构建的 FFT/LCT 线性原型，保留 spectral 作对照。** 原型包含平方光程重采样、抛物面卷积核、三维频域正则化反卷积和深度反变换；保留有符号输出。它不构造稠密逆矩阵，也不进行迭代，但未经原版工程对齐。

**首次替换实验发现：旧合成数据上的 LCT 定量误差很大，不能把替换解释为精度等效。** 默认无噪声参考密度相对误差约 87.68%，导数约 84.36%；原 spectral 对应约 0.85%、0.87%。两者处理完全相同的原始计数。有限孔径的零延拓、不同空间／时间离散以及未反卷积的 IRF 都会造成模型偏差。报告保留绝对单位，没有按真值拟合尺度或只展示归一化图像。

## 运行

从仓库根目录运行；只需 CPU：

```powershell
python -m pip install -r transient_transform/requirements.txt
python -m transient_transform.demo --inverse lct
python -m transient_transform.demo --inverse spectral
python -m transient_transform.compare
python -m unittest discover -s transient_transform/tests -v
```

显式选择旧后端后，默认 7 次完整扫描、80 次独立随机重复、2 个计算线程。原型 LCT 和 spectral 分别写入 `transient_transform/results/lct_demo/`、`spectral_demo/`，比较写入 `backend_comparison/`。换实验时指定新目录：

```powershell
python -m transient_transform.demo --inverse spectral --half-window 0.15 --peak-counts 150 --background 1.5 --trials 80 --output transient_transform/results/short_window
```

主要参数：`--frames` 为不小于 3 的奇数，`--half-window` 为慢时间半窗长，单位秒；`--ridge` 为 spectral 的相对正则化强度；`--lct-ridge` 默认 1e-4，为相对最大核频谱功率的正则化强度，`--lct-bins` 默认 256，为平方光程网格长度；`--background-repeats` 为背景标定的等曝光重复次数。演示期间假设每次完整扫描内物体静止，不模拟扫描耗时与墙点切换。

## 1. 固定空间栅格与已知算子

慢时间为 t，快时间为 s。未知量是固定 XYZ 网格上的有效体积散射密度 rho(t)，不是可移动点的位置。默认网格为 8×8×10，共 640 个未知量；中继墙为 12×12 个共焦位置，每个位置有 128 个均匀快时间箱。

对墙点 p_m 和体素中心 x_j：

\[
r_{mj}=\|x_j-p_m\|,\qquad s_{mj}=2r_{mj}/c,
\]

\[
A_{(m,k),j}
=g\,\frac{\Delta V}{r_{mj}^{4}}
\left[
\Phi\!\left(\frac{s_{k+1}-s_{mj}}{\sigma_{\rm IRF}}\right)
-\Phi\!\left(\frac{s_k-s_{mj}}{\sigma_{\rm IRF}}\right)
\right].
\]

这里空间积分采用体素中心求积；时间箱内的 Gaussian IRF 作解析积分。IRF 表示仪器时间响应，不是 Gaussian 空间表示。默认 sigma_IRF=60 ps，快时间覆盖 1–8 ns。极小时间核尾部按 1e-12 的概率阈值截断以构造稀疏矩阵。

g 是合成曝光增益：令线性密度场景参考时刻的最大期望信号约为 150 counts/bin，然后固定用于全部时刻、两个场景和全部方法。这是明确的合成标定，不是实际激光能量或硬件性能预测。

模型忽略未知表面法向、物体自遮挡和复杂多径，不把隐藏目标当成不透光气球。墙到激光器／探测器的外部固定光程已视为标定扣除。spectral 生成与求逆使用同一个离散 A；LCT 只使用几何、时间箱和曝光增益，不读取该 A，是独立离散的理想连续模型求逆。它未包含模拟器的 IRF 反卷积。尚没有高精度表面光传输或真实数据验证。

## 2. 直接拟合参考状态和变化率

在参考时刻 t_c 附近采用局部模型：

\[
\rho(t_n)\approx\rho_c+(t_n-t_c)u,
\qquad
\tau(t_n)\approx\tau_c+(t_n-t_c)d,
\]

其中 tau_c=A rho_c，d=A u。构造 X 的第 n 行为 [1,t_n-t_c]，直接计算：

\[
\begin{bmatrix}\widehat\tau_c\\\widehat d\end{bmatrix}
=X^+\,\tau.
\]

当前采用普通最小二乘，便于维持固定线性算子；未使用依赖当前观测的自适应权重。对称等间距时刻下，截距就是窗口均值，斜率权重与时间偏移成正比。

因此参考状态并不是从时间导数中猜出来的积分常数，而是由原始测量的绝对强度联合估计的截距。

## 3. 两个直接反演后端

### LCT：历史实验原型

当前模拟器的快时间核使用 delta(s-2r/c)，因此定义

\[
v=(cs/2)^2,\quad u=z^2,\quad
q(x,y,u)=\frac{\rho(x,y,\sqrt u)}{2\sqrt u},
\]

\[
g(x',y',v)=\frac{v^{3/2}}{\gamma c}\,
\tau\!\left(x',y',\frac{2\sqrt v}{c}\right),
\qquad g=h*q,
\]

其中 gamma 是前文的曝光增益，代码记作 `gain`；g 是变换后的瞬态。核为 h(dx,dy,v)=delta(v-dx²-dy²)。若采用论文的 delta(2r-cs) 约定，归一化常数相应不同。

实现先将计数除以快时间箱宽，变成每秒的瞬态密度，再作上述固定线性重采样与权重变换。卷积核在 v=dx²+dy² 上以两个相邻深度箱线性分配，并包含横向求积面积 dx*dy。零填充后计算

\[
\widehat q=\mathcal F^{-1}\!\left[
\frac{\overline{\widehat h}}{|\widehat h|^2+\alpha}
\mathcal F g\right],
\qquad \widehat\rho(x,y,z)=2z\widehat q(x,y,z^2).
\]

原生网格为墙面 12×12 和 256 个平方光程样本，FFT 网格为 24×24×512；最后用固定三线性插值取到原有 8×8×10 的 XYZ 输出网格。支持完整均匀矩形共焦墙网格，也处理墙点排列顺序。缺测、非均匀墙格和非 z=0 墙面会明确拒绝。

重建过程没有非负截断、取模、平方或逐帧最大值归一化。作者公开实现的 `max(real(vol),0)` 是输出的非负先验后处理，不是选择或消除算子零空间中的简并解。这里保留 `real(vol)` 的正负值。线性算子对参考 transient 和慢时间导数完全相同。

LCT 类还实现了精确离散伴随。理论噪声能量通过分块累加伴随基响应的平方求出，不生成稠密逆矩阵；这项诊断预计算的开销单独计时，不能混入普通 FFT 重建耗时。

有限测量域外的数据并不是真的为零。直接将测量补零并进行全局频域反卷积会引入边界偏差。`aperture_diagnostic` 使用同一个变换域源、同一个 H 和同一个滤波器，对比完整填充域数据与裁剪补零数据，隔离空间及时间测量域截断的效应；它不用于标定或缩放重建结果。默认变换域密度 q 的相对误差由完整域下约 34.01% 增至裁剪补零后的 90.21%；这是单独的匹配模型诊断，不等于对物理 demo 总误差的完整归因。

### Spectral：保留的小规模匹配模型对照

令 G=A^T A，作一次实对称谱分解 G=V diag(lambda_i) V^T。随后固定：

\[
\mathcal L_\alpha
=(A^T A+\alpha I)^{-1}A^T,
\qquad
\alpha=10^{-5}\lambda_{\max}(A^T A).
\]

再计算：

\[
\widehat\rho_c=\mathcal L_\alpha\widehat\tau_c,
\qquad
\widehat u=\mathcal L_\alpha\widehat d.
\]

没有训练、梯度下降或逐场景优化。正则化在看噪声实现之前固定，不按真值选择最优参数。rho 和 u 的重建都保留正负值，没有非负截断；原始真值 rho 非负，u 本来就可能有负值。

谱正则化造成的分辨率算子是 Q=L A，通常不等于 I。报告将 Q 的偏差和随机噪声分开，不把带正则的解称为精确逆。

两个后端采用相同的调用、时间拟合和噪声能量接口。spectral 包含模拟器的离散光程和 IRF；LCT 的正则化则作用在三维卷积核频谱上。二者正则化参数不具有逐值相同的含义，当前不是调参后的最优算法竞赛。

## 4. 噪声、背景标定与理论 SNR

每个扫描和时间箱采用独立 Poisson 计数：

\[
y_{ni}\sim\operatorname{Poisson}(\mu_{ni}+b_i),
\qquad \mu_n=A\rho(t_n).
\]

默认 b_i=1.5 counts/bin。另用 M=32 次等曝光背景测量形成一个共享的背景估计 b_hat；所有时刻都减去这同一个估计。它的标定误差因此在慢时间上相关。

对于某组时间权重 w_n，测量域方差为：

\[
v_i=\sum_n w_n^2(\mu_{ni}+b_i)
+\left(\sum_nw_n\right)^2\frac{b_i}{M}.
\]

截距权重和为 1，共享背景误差不会通过多帧平均完全消失；导数权重和为 0，共享背景的估计误差会抵消。但是每次实际背景光子的散粒噪声仍然保留在第一项。

重建噪声协方差和总能量为：

\[
\Sigma_{\rm out}=\mathcal L_\alpha\operatorname{diag}(v)\mathcal L_\alpha^T,
\qquad E\|e\|^2=\operatorname{tr}(\Sigma_{\rm out}).
\]

report.json 中分别记录：

- `output_signal_snr_db`：期望输出信号能量 / 随机噪声能量，取 10log10。
- `expected_reconstruction_snr_db_including_bias`：真实目标能量 /（确定性偏差能量＋随机噪声能量），取 10log10。这是包含误差偏差的重建指标。
- 理论与 Monte Carlo 的噪声 L2 RMS，以及时间近似偏差、反演偏差和单次重建误差。

未模拟热漂移、激光功率漂移、TDC 非线性、afterpulsing、死时间和首事件 pile-up。暗计数及稳定环境背景仅在这一版中合并为 b；不能据此评价真实 SPAD 的全部噪声。

## 5. 两个场景及比较方式

### linear_density

两个紧支撑多项式体素团之间的权重随慢时间线性变化：rho(t)=rho_c+t u。窗口内 rho 非负，局部模型严格成立。它不代表材料点匀速运动。

### translation

相同的两个多项式物体以固定速度 (0.08,0,-0.03) m/s 平移。真值直接在当前空间坐标求值，参考导数由解析空间梯度计算。尽管材料点速度恒定，固定栅格密度通常满足 rho_tt != 0。时间线性拟合存在可测的近似偏差。

对照方法：

- baseline：单个中心扫描重建 rho_c，相邻两侧扫描的中心差分重建 u。
- window：使用全部 7 次扫描，直接拟合截距与斜率后分别重建。

两者使用同一批已采集数据、同一 A、同一 L 和同一增益。但 window 使用更多观测，不能把收益宣传为相同两次扫描／相同有效光子预算下的提升。额外时间窗扫描也假设能完成足够快的整墙测量，真实硬件是否支持尚未验证。

## 6. 结果及当前限制

### LCT 与 spectral：相同数据、绝对单位的 window 重建

默认种子 20261029，7 次完整扫描、80 次噪声重复。下表为无噪声期望输出相对真值的 L2 偏差：

| 场景 | 参考密度 spectral / LCT | 密度导数 spectral / LCT |
| --- | ---: | ---: |
| 线性密度 | 0.85% / 87.68% | 0.87% / 84.36% |
| 匀速平移 | 9.03% / 87.84% | 12.19% / 93.32% |

线性密度的 LCT 参考输出沿真值方向的幅度投影约为 0.1654，导数约为 0.2283。绝对量值明显偏低，且还存在形状误差，不能只乘一个真值拟合增益就声称修复。3 个独立物理点源的峰值位置都能落在正确输出体素，但这不代表密度定量正确。

LCT 的线性拟合交换性、有符号响应、增益单位和协方差传播验证通过。80 次重复中，噪声能量的实测／理论比约 0.9686–1.0141。**噪声预测正确不等于重建准确。** 线性场景 window 的 LCT 输出信号 SNR 约 29.98 dB，而包含偏差的参考重建 SNR 只有约 1.14 dB；导数包含偏差的重建 SNR 约 1.47 dB。

LCT 的逆滤波频谱约 2.26 MiB；另有前向验证频谱、插值矩阵与 FFT 临时内存，不存储 90 MiB 的稠密逆映射。一次运行中 LCT 构建约 0.04 s，精确协方差诊断预计算约 15.3 s；这些数字是小规模实验记录，不是大场景速度保证。

### 历史 spectral 结果

默认种子 20261029，时间 t∈[-0.3,0.3] s，7 帧、80 次 Monte Carlo。下表采用包含偏差的期望重建 SNR，单位 dB：

| 场景 | 参考密度 baseline / window | 密度导数 baseline / window |
| --- | ---: | ---: |
| 线性密度 | 10.27 / 18.56 | -3.87 / 7.59 |
| 匀速平移 | 10.27 / 16.57 | -2.25 / 8.70 |

window 的参考密度噪声 RMS 约为 baseline 的 0.384，导数噪声 RMS 约为 0.267。默认 8 项方法／场景／分量组合中，Monte Carlo 噪声能量与理论预测之比约为 0.9915–1.0044。

线性密度的纯时间拟合误差在双精度舍入量级。平移场景的纯时间拟合相对误差为：参考状态约 8.60%，瞬时导数约 11.54%；加入固定谱反演偏差后分别约为 9.03%、12.19%。这些偏差不能通过增加光子数消除，需要缩短窗口或提高输运模型精度。

默认还对平移场景计算 4 种窗口的理论偏差／噪声／RMSE，见 `window_tradeoff.png`。各窗口有相同扫描数和每次曝光增益，未计入真实扫描时间限制。

历史本机一次 spectral 运行中，构造 A 约 3.02 s，预备谱逆约 3.48 s；两种方法的参考和导数求值合计约 0.08–0.09 s／场景。不含 Monte Carlo 和绘图。640×18432 的稠密逆映射占约 90 MiB，谱分解与存储开销随空间未知量迅速增加。这些数字仅描述 spectral 后端。

## 7. 输出及复用

| 文件 | 内容 |
| --- | --- |
| `report.json` | 模型条件、参数、时间、SNR、偏差、噪声核验及窗口扫描 |
| `geometry.npz` | XYZ 栅格、中继墙位置、快时间箱边界 |
| `operator.npz` | 稀疏前向算子 A，可用 scipy.sparse.load_npz 读取 |
| `reconstruction.png` | 真值、baseline、window 的参考密度与导数；显示沿 y 求和的 x-z 投影 |
| `temporal_and_noise.png` | 一个诊断时间箱的时间拟合，以及理论／实测噪声对照 |
| `window_tradeoff.png` | 平移场景的窗口偏差与方差权衡 |
| `backend_comparison/backends.png`、`comparison.json` | 相同原始计数上的 spectral / LCT 比较，使用公共色标和绝对单位 |
| `linear_density/experiment.npz`、`translation/experiment.npz` | 全部真值、原始计数、背景标定、时间权重、拟合结果与重建 |

结果目录已忽略，不提交模拟二进制数据。脚本可复现。数据文件均可通过 np.load(..., allow_pickle=False) 读取；`rho_frames` 是 [慢时间,体素]，`raw_counts` 是 [慢时间,墙点×快时间箱]，体素按 [x,y,z] 的 C 顺序展开。密度导数保存在 `rho_derivative`，重建的两行依次为参考密度和导数。

真实数据接入前需要提供完整扫描的慢时间、快时间标定、中继墙坐标、曝光归一化和独立背景测量。当前命令行只生成合成 demo，尚没有真实文件导入命令；各数学组件可以单独调用。

## 8. 验证与下一步

15 项单元测试覆盖原有 6 项数学检查，以及 LCT 的有符号线性与交换性、精确伴随、墙点重排、抛物面核光程／面积、完整卷积可观测子空间的尺度与相位、曝光增益、协方差、非法几何拒绝和 3 个独立物理点源的定位。完整卷积核本身也可能有零频谱模式，测试明确只要求恢复其可观测部分，不用非负截断掩盖这一事实。

demo 另检查整个实验链路的线性交换性、线性密度模型精确性、理论噪声与随机实验的一致性，以及导数正负值的保留。随机核验阈值是理论噪声能量比在 [0.7,1.3] 内，用于发现实现错误，不是物理精度保证。

下一步应先明确有限孔径边界处理、统一模拟与 LCT 的离散／体素积分，以及 IRF 的处理，降低当前巨大的模型偏差；随后再讨论输运生成元 B 的直接估计与真实数据。当前没有恢复速度场或声称解决不透光气球的完整形变。

模型参考：[O'Toole、Lindell、Wetzstein 的 LCT 推导说明](https://history.siggraph.org/wp-content/uploads/2021/06/2018-Talks-Otoole_Confocal-Non-line-of-sight-Imaging.pdf)；实现行为对照[作者公开的 LCT 重建分支](https://github.com/computational-imaging/nlos-fk/blob/master/cnlos_reconstruction.m#L82-L116)。本目录从积分恒等式独立实现，未复制其代码。
