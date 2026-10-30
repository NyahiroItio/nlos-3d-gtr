# 原版 LCT 工程对齐记录

更新：2026-10-30。工作分支：`research/transient-transform`。

当前推荐静态基线为 `original_lct.py`，对应作者项目页面下载的 2018 工程。此前的 `lct.py` 是从连续公式独立搭建的实验原型，尚未正确匹配旧模拟器，不再作为默认或原版基线。

## 1. 已完成的验证

- 下载并固定作者的代码／数据压缩包，校验 SHA-256；不覆盖上游原文件。
- 在本机 MATLAB R2024b Update 7 中实际运行作者工程。
- 处理一个现代 MATLAB 变量名兼容问题：只在单独的兼容副本中预声明 `width=[]`，随后原版 `load` 用真实墙面宽度覆盖它。原始源文件没有改动，原版数学运算和参数没有替换。
- 分别重建原始 S/U（逆反射）与漫反射 S 数据，原生体积为 64×64×512。
- 对照 MATLAB 的辐射加权、时间重采样、滤波、深度变换、正值截断、裁剪与翻转，Python 与 MATLAB 完整体积输出相差约 1e-15。
- MATLAB 兼容入口和只加入数组导出的参考副本，生成的 S/U PNG 像素完全相同。
- 21 项自动测试通过，其中 6 项使用实际 MATLAB 输出的冻结样例验证原版路径；运行这些测试不需要安装 MATLAB。

这证明工程实现对齐，不等于已经验证动态形变算法、绝对反射率标定或材料速度场。

## 2. 来源与可追溯性

作者页面：[Confocal non-line-of-sight imaging based on the light-cone transform](https://davidlindell.com/publications/nlos-lct)。使用其中的 **LCT MATLAB code and data** 链接，不使用后来的 nlos-fk 工程替代 2018 工程。

固定压缩包 `cnlos_code.zip`：54,057,937 bytes。

```text
archive SHA-256:
db3fef096e242baa642595cfd639da23f1f220e458d12c17832e4257a22024ec

original cnlos_reconstruction.m SHA-256:
824275cb1fbb5c33751781b7b7498d7c793a92dc07a456670ec169e38ea6b1f6
```

上游源代码和原始数据保存在已忽略的 `results/upstream/`，不作为本项目的 MIT 软件重新分发。压缩包没有单独的许可证文件；这里保留下载来源、校验值和原文件，仅提交本项目实现、验证工具及小型确定性数值参考。不要把后来的仓库许可证自动套到这个压缩包。

实际 MATLAB 导出参考位于 `results/original_reference/`；`provenance.json` 记录原始源文件、兼容变动及参考副本校验值。参考副本仅增加变量声明、数组导出和两个小型测试输入，保留原版重建运算。

## 3. 运行与查看图像

在仓库根目录运行：

```powershell
python -m pip install -r transient_transform/requirements.txt
python -m transient_transform.upstream
python -m transient_transform.demo
```

`upstream` 首次下载约 52 MiB；已有正确压缩包时无需联网。若下载内容不匹配固定校验值，会明确报错，不静默替换基线。后续 Python 重建不需要 MATLAB 或 GPU。

也可以直接运行新的独立入口：

```powershell
python -m transient_transform.original_demo --scenes 7 9
```

其他原版场景可通过 `--scenes` 选择。默认数据包还包含 Exit、分辨率板、点阵、人形、室外 S、Bunny。Bunny 的 256×256 横向网格需要显著更多内存，不作为默认测试。

默认输出：

```text
transient_transform/results/original_alignment/
  report.json
  scene_7_s_u/
    reconstruction.png
    reconstruction.svg
    measured_transient.png
    reconstruction.npz
  scene_9_diffuse_s/
    reconstruction.png
    reconstruction.svg
    measured_transient.png
    reconstruction.npz
```

`reconstruction.png` 是按原版方式生成的正面／俯视／侧面最大强度投影。采用真实 64×64×512 体积，不是将旧版 8×8×10 插值放大。`measured_transient.png` 展示进入重建前的测量：数据已经按上游说明做过直达光时间校正／裁剪，我们再按原版求和降采样及屏蔽前段时间箱，因此不能称为完全未经处理的 SPAD 原始事件。

`reconstruction.npz` 保存：

- `signed_volume`：[depth,y,x] 原生未截断体积。
- `display_volume`：按原版正值截断、x 翻转和深度裁剪后的体积。
- `lateral_m`、`depth_m`：显示坐标。

输出单位是作者工程的数值单位，尚不是经过系统增益标定的物理散射密度。原始数据包没有可见光相机的目标原图。

## 4. 精确对齐的工程约定

| 项目 | 对齐方式 |
| --- | --- |
| 光速 | 3e8 m/s，与作者代码一致 |
| `width` | 扫描墙面的半宽，S/U 与 diffuse S 均为 0.35 m |
| 输入 | `rect_data[x,y,fast_time]`，上游已校正和裁剪 |
| 降采样 | K=2，每次将相邻时间箱计数相加，不求平均 |
| 原始时间箱 | 默认 4 ps，K=2 后为 16 ps；Bunny 原始为 8 ps |
| 早期屏蔽 | 默认原始前 600 箱，经两次降采样后为 150 箱 |
| 体积布局 | MATLAB `permute(rect_data,[3 2 1])`，即 [depth,y,x] |
| 辐射权重 | 逆反射用归一化深度平方；漫反射用四次方 |
| PSF | 相同归一化网格、最近深度层选择、双重归一化与循环移位 |
| 重采样 | 对作者 M² 个细样本加权、逐对平均的操作作等价聚合；反变换与作者一样使用配套转置权重，不另行替换成点插值 |
| 滤波 | conj(FFT(PSF))/(abs(FFT(PSF))²+1/snr) |
| 默认 snr | 逆反射 0.8；漫反射 0.08，与原版分支一致 |
| 后处理 | 保留原版正值截断、x 翻转、场景相关深度裁剪 |

这里的 `snr` 是原工程的滤波参数，不能直接等同于先前从 Poisson 计数协方差推导的实测 SNR。当前没有改成相对最大频谱功率的正则化，也没有按真值拟合增益。

PSF、重采样和各线性阶段采用 float64。Python 用实数 FFT 减少内存；与 MATLAB 完整复数 FFT 的差异在双精度数值误差量级。

## 5. 完整体积数值核验

下表为 Python 相对 MATLAB 的 L2 差异，不是重建相对真实物体的误差：

| 阶段 | S/U | 漫反射 S |
| --- | ---: | ---: |
| 辐射加权 | 1.75e-16 | 2.47e-16 |
| 时间重采样 | 5.33e-16 | 5.75e-16 |
| 滤波后体积 | 1.12e-15 | 1.10e-15 |
| 未截断重建 | 1.06e-15 | 1.06e-15 |
| 截断并裁剪后的显示体积 | 8.86e-16 | 8.95e-16 |

一次本机 CPU 运行中，两个场景各需约 1.4–1.7 s 准备、1.3 s 线性重建，不包括文件读取、绘图、保存及数值核验。MATLAB 原版的计时范围不同，不将这两个数字用作严格的速度竞赛。

通过标准：全部被核验阶段的相对误差 <1e-9，参数、尺寸与模式一致。`report.json` 会显式标记 MATLAB 参考是否存在；缺少参考时只生成图像，不谎报已经在当前机器完成完整场景数值核验。

## 6. 重新生成 MATLAB 参考与小型测试样例

```powershell
python -m transient_transform.reference
```

该命令打印验证脚本的绝对路径。在本项目位置，可运行：

```powershell
matlab -batch "run('F:/deformable NLOS/nlos-3d-gtr/transient_transform/results/original_reference/run_reference.m')"
python -m transient_transform.reference --export-fixtures
python -m unittest discover -s transient_transform/tests -v
```

MATLAB 脚本先尝试未改动的作者入口并记录状态；在本机 R2024b 上，它因 `width` 的隐式加载／同名函数解析失败。接着运行仅增加 `width=[]` 的兼容副本，重建 S/U；再导出两个小型参考输入及两个完整原始场景。兼容声明不改变真实加载的 `width` 值。

被冻结的小型参考位于 `tests/fixtures/author_matlab_reference.npz`，约 50 KiB。它由 MATLAB 产生，不由待测 Python 算法生成，包含 PSF、滤波器、两套重采样权重、逐阶段数组及后处理结果，因此后续测试可以脱离 MATLAB 和大数据运行。

## 7. 与后续时序研究的接口

`OriginalLCT.__call__()` 返回未施加非负截断的线性结果；静态显示才通过 `display_volume(..., clip=True)` 复现原版后处理。已在小型参考和完整数据上验证有符号线性，以及时间拟合与固定线性逆的交换性。

这为之后重新接入 rho_0 / dot(rho) 提供已验证的后端。当前默认入口只做原版静态基线，尚未重新实现动态合成、逐扫描时间拟合或速度场恢复。旧的物理中心点模拟器也未因这次对齐而自动变成原版 LCT 的匹配前向模型。

历史入口仍保留：

```powershell
python -m transient_transform.demo --inverse spectral
python -m transient_transform.demo --inverse lct
```

其中 `lct` 明确指早期失败的连续公式原型，不能将它的约 88% 误差当作原版 LCT 的性能。历史说明见 `README.md` 后半部分。
