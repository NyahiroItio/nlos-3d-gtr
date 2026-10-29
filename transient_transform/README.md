# 固定体素 transient 变换：非迭代数学 demo

本目录在 `research/transient-transform` 分支中独立实现，不导入 3D-GTR、Torch 或 Gaussian 场景组件。

本版验证的链路是：**自行生成三维体素密度与完整共焦扫描 → 加入光子计数噪声 → 在慢时间上直接拟合参考 transient 和导数 → 用固定线性逆算子直接输出参考密度与密度导数 → 核对理论噪声传播。**

**这是小规模谱分解参考实现，不是 FFT/LCT 快速卷积算法。** 它采用与简化 LCT 体积散射模型一致的球面光程积分，便于先核验数学关系。当前的速度、内存与可逆性不能外推到高分辨率或真实不透光表面。

## 运行

从仓库根目录运行；只需 CPU：

```powershell
python -m pip install -r transient_transform/requirements.txt
python -m transient_transform.demo
python -m unittest discover -s transient_transform/tests -v
```

默认 7 次完整扫描、80 次独立随机重复、2 个计算线程。结果在 `transient_transform/results/demo/`，重跑覆盖该输出目录中的同名文件。换实验时指定新目录：

```powershell
python -m transient_transform.demo --half-window 0.15 --peak-counts 150 --background 1.5 --trials 80 --output transient_transform/results/short_window
```

主要参数：`--frames` 为不小于 3 的奇数，`--half-window` 为慢时间半窗长，单位秒；`--ridge` 为相对谱正则化强度；`--background-repeats` 为背景标定的等曝光重复次数。演示期间假设每次完整扫描内物体静止，不模拟扫描耗时与墙点切换。

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

模型忽略未知表面法向、物体自遮挡和复杂多径，不把隐藏目标当成不透光气球。墙到激光器／探测器的外部固定光程已视为标定扣除。生成与求逆使用同一个 A，属于模型内验证；尚没有独立高精度光传输仿真或真实数据验证。

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

## 3. 谱分解直接反演

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

这是可替换的逆算子接口。后续可在适用几何下替换成真正的 LCT 坐标变换与频域滤波，而保留时间拟合和噪声传播组件。

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

## 6. 默认结果

默认种子 20261029，时间 t∈[-0.3,0.3] s，7 帧、80 次 Monte Carlo。下表采用包含偏差的期望重建 SNR，单位 dB：

| 场景 | 参考密度 baseline / window | 密度导数 baseline / window |
| --- | ---: | ---: |
| 线性密度 | 10.27 / 18.56 | -3.87 / 7.59 |
| 匀速平移 | 10.27 / 16.57 | -2.25 / 8.70 |

window 的参考密度噪声 RMS 约为 baseline 的 0.384，导数噪声 RMS 约为 0.267。默认 8 项方法／场景／分量组合中，Monte Carlo 噪声能量与理论预测之比约为 0.9915–1.0044。

线性密度的纯时间拟合误差在双精度舍入量级。平移场景的纯时间拟合相对误差为：参考状态约 8.60%，瞬时导数约 11.54%；加入固定谱反演偏差后分别约为 9.03%、12.19%。这些偏差不能通过增加光子数消除，需要缩短窗口或提高输运模型精度。

默认还对平移场景计算 4 种窗口的理论偏差／噪声／RMSE，见 `window_tradeoff.png`。各窗口有相同扫描数和每次曝光增益，未计入真实扫描时间限制。

本机一次运行中，构造 A 约 3.02 s，预备谱逆约 3.48 s；两种方法的参考和导数求值合计约 0.08–0.09 s／场景。不含 Monte Carlo 和绘图。640×18432 的稠密逆映射占约 90 MiB，谱分解与存储开销随空间未知量迅速增加。**当前实现是小体积数学参考，不具备 LCT 的高分辨率计算复杂度。**

## 7. 输出及复用

| 文件 | 内容 |
| --- | --- |
| `report.json` | 模型条件、参数、时间、SNR、偏差、噪声核验及窗口扫描 |
| `geometry.npz` | XYZ 栅格、中继墙位置、快时间箱边界 |
| `operator.npz` | 稀疏前向算子 A，可用 scipy.sparse.load_npz 读取 |
| `reconstruction.png` | 真值、baseline、window 的参考密度与导数；显示沿 y 求和的 x-z 投影 |
| `temporal_and_noise.png` | 一个诊断时间箱的时间拟合，以及理论／实测噪声对照 |
| `window_tradeoff.png` | 平移场景的窗口偏差与方差权衡 |
| `linear_density/experiment.npz`、`translation/experiment.npz` | 全部真值、原始计数、背景标定、时间权重、拟合结果与重建 |

结果目录已忽略，不提交模拟二进制数据。脚本可复现。数据文件均可通过 np.load(..., allow_pickle=False) 读取；`rho_frames` 是 [慢时间,体素]，`raw_counts` 是 [慢时间,墙点×快时间箱]，体素按 [x,y,z] 的 C 顺序展开。密度导数保存在 `rho_derivative`，重建的两行依次为参考密度和导数。

真实数据接入前需要提供完整扫描的慢时间、快时间标定、中继墙坐标、曝光归一化和独立背景测量。当前命令行只生成合成 demo，尚没有真实文件导入命令；各数学组件可以单独调用。

## 8. 验证与下一步

6 项单元测试覆盖：物理光程和时间箱积分；有负号的导数及参考截距；精确小算子求逆与时间拟合交换性；共享背景的相关噪声；匀速平移的解析密度导数；非线性密度的窗口偏差。

demo 另检查整个实验链路的线性交换性、线性密度模型精确性、理论噪声与随机实验的一致性，以及导数正负值的保留。随机核验阈值是理论噪声能量比在 [0.7,1.3] 内，用于发现实现错误，不是物理精度保证。

下一步可围绕真正的 LCT 快速逆算子、输运生成元 B 的直接估计、独立光传输模型与真实计数标定展开。当前没有恢复速度场或声称解决不透光气球的完整形变。

模型参考：[O'Toole、Lindell、Wetzstein 的 LCT 推导说明](https://history.siggraph.org/wp-content/uploads/2021/06/2018-Talks-Otoole_Confocal-Non-line-of-sight-Imaging.pdf)。本目录未复制其代码。
