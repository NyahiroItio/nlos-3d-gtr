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
| `upstream.py` | 下载、缓存和校验固定版本的作者工程与数据 |
| `reference.py` | 生成 MATLAB 对照脚本及导出冻结参考 |
| `tests/test_original_lct.py` | 当前基线的 6 项数值验证 |
| `tests/fixtures/` | 小型 MATLAB 数值参考及来源说明 |

`OriginalLCT.__call__()` 返回原生 `[depth, y, x]` 有符号线性结果；`display_volume(..., clip=True)` 用于复现作者的静态显示。当前实现提供静态基线，动态合成、逐扫描时间拟合和速度场恢复尚未接入。

完整来源、工程约定、MATLAB 兼容处理与重新生成参考的步骤见 [原版工程对齐说明](ORIGINAL_ALIGNMENT.md)。

数学建模、连续逆与实际离散实现的关系见 [LCT 推导笔记](LCT_DERIVATION.md)。
