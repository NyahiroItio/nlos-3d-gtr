# Transient 变换研究：原版 LCT 基线

本目录在 `research/transient-transform` 分支中独立实现，不依赖 3D-GTR、Torch 或 Gaussian 场景组件。

当前基线对应作者发布的 2018 LCT 工程。Python 与本机 MATLAB 的原始 S/U、漫反射 S 完整体积逐阶段差异约为 1e-15。6 项冻结参考测试验证 PSF、重采样、滤波、显示后处理和有符号线性；测试不需要 MATLAB。

## 运行

从仓库根目录运行，只需 CPU：

```powershell
python -m pip install -r transient_transform/requirements.txt
python -m transient_transform.upstream
python -m transient_transform.demo
```

默认重建场景 7（S/U）和 9（漫反射 S）。可选择其他原版场景或输出目录：

```powershell
python -m transient_transform.demo --scenes 7 --output transient_transform/results/scene_7
python -m transient_transform.demo --help
python -m unittest discover -s transient_transform/tests -v
```

`python -m transient_transform.original_demo` 提供相同的运行入口。原版数据下载后会保留在本地，后续重建不需要重复下载。

## 静态平均验证

```powershell
python -m transient_transform.static_mean
```

使用原版 S/U 与漫反射 S 测量作为固定的经验期望，模拟 7 次独立计数曝光，比较原版单帧与 LTAI 静态平均。另有相同总曝光量的原版对照。默认 16 组随机序列，CPU 两个线程；当前测试共 25 项。

结果在 `results/static_mean/`，包含重建三视图、噪声缩放图、观测与有符号体积、验证报告。这里验证相对于原版测量的平均估计，不声称消除了该测量自身的噪声或验证了真实密度。方法与判据见 [静态平均验证说明](STATIC_MEAN.md)。

## 动态参考状态与变化率

```powershell
python -m transient_transform.dynamic_demo
```

包含严格线性的密度淡入/淡出，以及从独立飞行时间模型生成的 S 平移。默认采用二阶慢时间拟合，与使用同一批计数的线性拟合、原版逐帧重建及相邻/首尾差分比较，输出 GIF、时间帧对比、有符号变化率图及偏差/噪声报告。默认 7 帧、12 组噪声序列、2 个 CPU 线程，原生体积仍为 64×64×512。

方法、数据假设、曝光预算和物理验证边界见 [动态验证说明](DYNAMIC_VALIDATION.md)。

固定 5 帧、半窗口 0.2 秒的**平面径向膨胀**实验可单独运行：

```powershell
python -m transient_transform.dynamic_demo --cases expansion --frames 5 --half-window 0.2 --trials 12 --output transient_transform/results/expansion_fixed_h02
```

最外侧散射点速度为 0.08 m/s，点权重守恒。中心状态与一阶导数总误差分别为约 3.08% 和 15.87%，具体定义及与平动的匹配对照见 [动态验证说明](DYNAMIC_VALIDATION.md)。

同样固定窗口的**深度方向膨胀**可单独运行：

```powershell
python -m transient_transform.dynamic_demo --cases depth_expansion --frames 5 --half-window 0.2 --trials 12 --output transient_transform/results/depth_expansion_fixed_h02
```

三层 S 形薄体在深度方向收缩/膨胀。结果另有 `depth_motion.png`、`depth_motion.gif` 与 `depth_derivative.png`，用 \(x\)-\(z\) 剖面显示正面图看不出的轴向运动。此场景的一阶导数总误差约 58.33%，主要由固定 0.2 秒窗口的时间近似偏差造成；详见 [动态验证说明](DYNAMIC_VALIDATION.md)。

基于该实验保存的瞬态，可以复查轴向伸缩先验和点响应位置导数匹配：

```powershell
python -m transient_transform.axial_prior_probe --trials 12
```

层间距能准确估计这组数据的全局膨胀率；直接对 LCT 体积做空间输运微分却明显劣于原时间拟合。原始瞬态上的点响应导数模板能在已知横向坐标处定位两个运动层，但尚不是通用速度重建。方法与完整误差见 [轴向先验试验](AXIAL_PRIOR_PROBE.md)。

在同一批五帧原始瞬态上进一步检验有限位移模板：

```powershell
python -m transient_transform.finite_shift_probe
```

匹配图直接扫描候选深度与轴向速度，不先估计瞬时导数。离网格单点及三层场景的 12 组泊松序列均给出接近真值的主峰；单点也会产生较弱的假次峰，因此这仍是已知横向位置的检测实验。见 [有限位移试验](FINITE_SHIFT_PROBE.md)。

对相同点响应字典进行几何相干性扫描：

```powershell
python -m transient_transform.coherence_probe
```

局部 7×7 墙点下，5 mm 横向移动后的模板相关度约 0.999；将同样数量的墙点分散到更宽墙面后约为 0.955。深度、速度与小位移强度的不同退化方式见 [相干性研究](COHERENCE_STUDY.md)。

## 二阶拟合窗口扫描

```powershell
python -m transient_transform.window_scan
```

在同一个物理 S 形场景中比较 3、5、7 帧窗口，并分别控制每帧曝光和总曝光。0.08 m/s 时，导数在已测范围内以 5 帧、半窗口 0.2 秒最低；把速度改为 0.04 或 0.16 m/s 后，最低点分别移至已测的 0.3 或 0.1 秒。模拟的 0.1 秒帧间隔不代表硬件可达到的扫描速度。设计、理论误差分解及速度敏感性见 [窗口扫描说明](WINDOW_SCAN.md)。

## 结果

默认输出位于 `results/original_alignment/`：

| 文件 | 内容 |
| --- | --- |
| `scene_*/reconstruction.png`、`.svg` | 原生三维重建的正面、俯视、侧面最大强度投影 |
| `scene_*/measured_transient.png` | 准备后的测量瞬态与墙面累计计数 |
| `scene_*/reconstruction.npz` | 有符号原生体积、显示体积及空间坐标 |
| `report.json` | 参数、计时、有符号线性及 MATLAB 对照结果 |

输入已由上游完成时间校正和裁剪，运行时再按原版降采样并屏蔽前段时间箱。输出采用作者工程的数值单位，尚未标定为物理散射密度。MATLAB 对照未生成时，报告会明确标记缺少参考。

## 代码与验证

| 文件 | 用途 |
| --- | --- |
| `original_lct.py` | 原版离散约定下的静态 LCT 线性核心与显示后处理 |
| `original_demo.py`、`demo.py` | 数据重建、绘图、保存及命令行入口 |
| `static_mean.py` | 流式静态均值估计、原版/新版本比较及曝光预算验证 |
| `temporal_fit.py`、`dynamic_demo.py` | 联合参考/斜率估计、动态对比与动画 |
| `motion_simulation.py` | 独立的空间点飞行时间与仪器响应模拟 |
| `axial_prior_probe.py` | 轴向层间距估计、输运导数对照和原始瞬态的单点导数匹配 |
| `finite_shift_probe.py` | 原始五帧瞬态的深度–轴向速度有限位移模板匹配 |
| `coherence_probe.py` | 理想中继面上点响应字典的深度、速度与横向相干性扫描 |
| `upstream.py` | 下载、缓存和校验固定版本的作者工程与数据 |
| `reference.py` | 生成 MATLAB 对照脚本及导出冻结参考 |
| `tests/test_original_lct.py` | 当前基线的 6 项数值验证 |
| `tests/test_static_mean.py` | 静态平均的统计与曝光归一化验证 |
| `tests/test_dynamic.py` | 时间协方差、窗口偏差、飞行时间与解析运动导数验证 |
| `tests/fixtures/` | 小型 MATLAB 数值参考及来源说明 |

`OriginalLCT.__call__()` 返回原生 `[depth, y, x]` 有符号线性结果；`display_volume(..., clip=True)` 用于复现作者的静态显示。当前实现提供静态平均、局部二阶参考/斜率/曲率拟合和独立平移模拟；尚未恢复材料速度场或验证真实动态实验数据。

完整来源、工程约定、MATLAB 兼容处理与重新生成参考的步骤见 [原版工程对齐说明](ORIGINAL_ALIGNMENT.md)。

数学建模、连续逆与实际离散实现的关系见 [LCT 推导笔记](LCT_DERIVATION.md)。
