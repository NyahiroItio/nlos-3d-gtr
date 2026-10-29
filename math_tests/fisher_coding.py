"""CPU 数学实验：非相干照明编码的 Poisson Fisher 信息上限。

运行：python math_tests/fisher_coding.py
依赖：numpy, scipy；生成图片另外需要 matplotlib。无需 torch/CUDA/Triton。

这里刻意使用可解析的简化模型，而不是导入完整 3D-GTR：
一个隐藏散射 Gaussian，固定时间宽度，未知中心位置与散射强度。
不包含形变、遮挡、IRF 参数估计或相干光学；先把统计上限验证清楚。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr


# theta = [x_cm, z_cm, log_gain]。固定单位后，矩阵特征值才可以比较。
# x,z 用 cm 是为了让数值尺度较接近。CRLB 的位置标准差也因此为 cm。
C_M_PER_PS = 2.99792458e-4
THETA_TRUE = np.array([8.0, 70.0, 0.0])
WALL_X = np.linspace(-0.45, 0.45, 9)
BIN_EDGES_PS = np.arange(3000.0, 6500.1, 16.0)
SIGMA_PS = 60.0


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def transient_and_jacobian(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回每单位入射剂量的期望计数 h[p,k] 和 J[p,k,a]。

    墙位于 z=0，照明点 p=(wall_x,0)，固定探测点 d=(0,0)。
    隐藏点 x=(x,z)，d1=|x-p|，d2=|x-d|。

    时间中心 mu=(d1+d2)/c，单位 ps。
    toy 振幅 A = 1e-3 * exp(log_gain) * z^2/(d1^3*d2^3)。
    它包含墙面余弦 z/d1,z/d2 和逆平方衰减；1e-3 是固定效率。
    这是简化散射模型，不是完整 3D-GTR 的投影面积振幅。

    与在 bin 中心采样相比，对 bin 积分能保留正确的光子计数单位：
      q_k = Phi((edge_hi-mu)/sigma) - Phi((edge_lo-mu)/sigma)
      h_k = A*q_k
      dq_k/dmu = [phi(lo)-phi(hi)]/sigma。
    """
    x, z = theta[:2] * 0.01  # cm -> m
    log_gain = theta[2]
    dx1 = x - WALL_X
    d1 = np.sqrt(dx1**2 + z**2)
    d2 = np.sqrt(x**2 + z**2)
    mu = (d1 + d2) / C_M_PER_PS
    amplitude = 1e-3 * np.exp(log_gain) * z**2 / (d1**3 * d2**3)

    lo = (BIN_EDGES_PS[:-1][None, :] - mu[:, None]) / SIGMA_PS
    hi = (BIN_EDGES_PS[1:][None, :] - mu[:, None]) / SIGMA_PS
    # 用右尾的生存函数表达式，避免两个非常接近 1 的 CDF 相减丢精度。
    q = np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
    dq_dmu = (normal_pdf(lo) - normal_pdf(hi)) / SIGMA_PS
    h = amplitude[:, None] * q

    # 链式法则：dh/dtheta = h*dlog(A)/dtheta + A*dq/dmu*dmu/dtheta。
    # 第二项来自光程改变导致的峰移动，第一项来自几何辐射衰减。
    dmu = np.column_stack([
        (dx1 / d1 + x / d2) * 0.01 / C_M_PER_PS,
        (z / d1 + z / d2) * 0.01 / C_M_PER_PS,
        np.zeros_like(d1),
    ])
    dlog_a = np.column_stack([
        -3 * (dx1 / d1**2 + x / d2**2) * 0.01,
        (2 / z - 3 * z * (1 / d1**2 + 1 / d2**2)) * 0.01,
        np.ones_like(d1),
    ])
    jac = h[..., None] * dlog_a[:, None, :]
    jac += amplitude[:, None, None] * dq_dmu[..., None] * dmu[:, None, :]
    return h, jac


def encode(w: np.ndarray, h: np.ndarray, jac: np.ndarray):
    """w[r,p] 是非负剂量，而非允许任意放大的抽象矩阵系数。

    每个编码曝光 r 同时照亮多个墙面点，探测器丢失来源 p 的标签：
      lambda[r,k] = sum_p w[r,p]*h[p,k]
      J_code[r,k,a] = sum_p w[r,p]*J[p,k,a]。
    假设没有相干交叉项、背景、pile-up，所有期望数服从线性叠加。
    """
    return np.einsum('rp,pk->rk', w, h), np.einsum('rp,pka->rka', w, jac)


def fisher(mean: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """从独立 Poisson 似然得到 F = sum_i (d lambda_i)(d lambda_i)^T/lambda_i。

    log p(y|theta)=sum_i [y_i log(lambda_i)-lambda_i-log(y_i!)]。
    score=sum_i (y_i/lambda_i-1)*d lambda_i。
    用 E[y_i]=Var[y_i]=lambda_i 和通道独立性，得到上述公式。
    等价于 F=J_white.T@J_white，J_white=J/sqrt(lambda)。

    极小尾部 bin 在 float64 中可能下溢为 0，这些 bin 的导数也为 0。
    仅跳过不可表示的尾部；不加人为 epsilon 以免制造/削弱信息。
    """
    flat_mean = mean.ravel()
    flat_jac = jac.reshape(-1, jac.shape[-1])
    keep = flat_mean > 1e-280
    white = flat_jac[keep] / np.sqrt(flat_mean[keep, None])
    return white.T @ white


def separated_fisher(w: np.ndarray, h: np.ndarray, jac: np.ndarray):
    """公平参照：同样的逐点总剂量 e[p]=sum_r w[r,p]，但保留来源标签。

    在静态、无额外背景和切换开销时，可用逐点扫描实现。
    编码和参照的总期望信号光子数也相同。
    """
    e = w.sum(axis=0)
    return fisher(e[:, None] * h, e[:, None, None] * jac)


def information_loss(w: np.ndarray, h: np.ndarray, jac: np.ndarray):
    """上限的精确损失恒等式，比只检查特征值更能解释原因。

    每个 r,k 中令 a_p=w_rp*h_pk，s_p=grad log(h_pk)=J_pk/h_pk。
    Lambda=sum_p a_p，s_bar=sum_p a_p*s_p/Lambda。
    则：
      F_separated - F_code
        = sum_{r,k,p} a_p (s_p-s_bar)(s_p-s_bar)^T >= 0。

    它是被混合通道 score 的加权协方差。每项都是 vv^T 乘非负数，
    所以必为半正定。混合的 score 相同时损失为 0；相反时损失很大。
    """
    loss = np.zeros((jac.shape[-1], jac.shape[-1]))
    for row in w:
        for k in range(h.shape[1]):
            active = (row > 0) & (h[:, k] > 1e-280)
            weights = row[active] * h[active, k]
            if weights.size == 0:
                continue
            scores = jac[active, k] / h[active, k, None]
            mean_score = np.average(scores, axis=0, weights=weights)
            centered = scores - mean_score
            loss += (centered * weights[:, None]).T @ centered
    return loss


def effective_geometry(f: np.ndarray) -> np.ndarray:
    """最后一个参数是未知 log_gain；前两个为几何量。

    nuisance 已知时几何信息为 F_gg；未知时应使用 Schur complement：
      F_eff = F_gg - F_gn F_nn^{-1} F_ng。
    等价于从几何的白化导数中投影掉可由 nuisance 导数解释的部分。
    本例 nuisance 块是一个严格为正的标量；不添加 ridge 充当信息。
    """
    if f[-1, -1] <= 0:
        raise ValueError('nuisance 参数没有信息，不能直接求 Schur complement')
    eff = f[:2, :2] - np.outer(f[:2, -1], f[-1, :2]) / f[-1, -1]
    return (eff + eff.T) / 2


def crlb_std_cm(f: np.ndarray):
    """仅在局部可辨识时求逆；奇异矩阵不能用伪逆伪装有限误差。"""
    eff = effective_geometry(f)
    vals = np.linalg.eigvalsh(eff)
    if vals[0] <= max(vals[-1], 1.0) * 1e-12:
        return None
    return np.sqrt(np.diag(np.linalg.inv(eff)))


def complementary_codes(alpha: float, total_dose: float):
    """两幅非负互补图案；每个墙面点总剂量始终为 total_dose/P。

    alpha=0.5 时两个图案相同；alpha=1 时两个墙面分组完全分离。
    alpha=1 仍然混合每组内部通道，不等于完整逐点扫描。
    """
    signs = np.where(WALL_X < 0, -1.0, 1.0)
    first = 0.5 + (alpha - 0.5) * signs
    return np.stack([first, 1 - first]) * (total_dose / len(WALL_X))


def run_checks(seed: int, total_dose: float):
    rng = np.random.default_rng(seed)
    h, jac = transient_and_jacobian(THETA_TRUE)
    errors = []
    for a, step in enumerate([1e-4, 1e-4, 1e-5]):
        delta = np.zeros(3)
        delta[a] = step
        fd = (transient_and_jacobian(THETA_TRUE + delta)[0]
              - transient_and_jacobian(THETA_TRUE - delta)[0]) / (2 * step)
        error = np.linalg.norm(fd - jac[..., a]) / np.linalg.norm(jac[..., a])
        errors.append(float(error))
        assert error < 2e-6, ('Jacobian finite difference', a, error)

    e = np.full(len(WALL_X), total_dose / len(WALL_X))
    onehot = np.diag(e)
    coded = fisher(*encode(onehot, h, jac))
    separated = separated_fisher(onehot, h, jac)
    assert np.allclose(coded, separated, rtol=1e-12, atol=1e-10)

    worst_relative_min_eigenvalue = 0.0
    for _ in range(30):
        w = rng.uniform(0.01, 1.0, (4, len(WALL_X)))
        w *= e[None, :] / w.sum(axis=0, keepdims=True)
        fc = fisher(*encode(w, h, jac))
        fs = separated_fisher(w, h, jac)
        loss = (fs - fc + (fs - fc).T) / 2
        scale = np.linalg.norm(fs, 2)
        min_eig = np.linalg.eigvalsh(loss)[0] / scale
        worst_relative_min_eigenvalue = min(worst_relative_min_eigenvalue, min_eig)
        assert min_eig > -1e-11, ('PSD bound', min_eig)
        explicit = information_loss(w, h, jac)
        assert np.allclose(loss, explicit, rtol=2e-9, atol=scale * 1e-10)
        # 处理未知 gain 后的有效几何信息也不能超出相同 nuisance 的参照。
        assert np.linalg.eigvalsh(effective_geometry(fs) - effective_geometry(fc))[0] > -scale * 1e-10
        assert np.isclose(encode(w, h, jac)[0].sum(), (e[:, None] * h).sum())

    # 极简反例：h1=1+theta,h2=1-theta，在 theta=0 时导数为 +1,-1。
    # 分开看 F=1^2/1+(-1)^2/1=2；只看总数时 lambda=2 不随 theta 变化，F=0。
    toy_h = np.ones((2, 1))
    toy_j = np.array([[[1.0]], [[-1.0]]])
    f_before = separated_fisher(np.ones((1, 2)), toy_h, toy_j)[0, 0]
    f_after = fisher(*encode(np.ones((1, 2)), toy_h, toy_j))[0, 0]
    assert np.isclose(f_before, 2) and np.isclose(f_after, 0)

    # 取等例子：h=[1,2], J=[1,2] -> score 都是 1，混合不损失信息。
    equal_h = np.array([[1.0], [2.0]])
    equal_j = equal_h[..., None]
    assert np.allclose(separated_fisher(np.ones((1, 2)), equal_h, equal_j),
                       fisher(*encode(np.ones((1, 2)), equal_h, equal_j)))
    return {'jacobian_relative_errors': errors, 'random_codes_checked': 30,
            'worst_relative_loss_eigenvalue': worst_relative_min_eigenvalue,
            'cancellation_information_before': float(f_before),
            'cancellation_information_after': float(f_after)}


def fit_counts(w: np.ndarray, counts: np.ndarray, initial: np.ndarray):
    """最小化 Poisson NLL（省略与 theta 无关的 log(y!)）。

    NLL=sum(lambda-y*log(lambda))；梯度=sum((1-y/lambda)*J)。
    仅此优化器在极小 lambda 处做机器级截断，避免 log(0)。
    Fisher 计算没有用这个截断来人为稳定矩阵。
    """
    def objective(theta):
        mean, jac = encode(w, *transient_and_jacobian(theta))
        safe = np.maximum(mean, 1e-280)
        nll = np.sum(mean - counts * np.log(safe))
        gradient = np.einsum('rk,rka->a', 1 - counts / safe, jac)
        return nll, gradient

    return minimize(objective, initial, jac=True, method='L-BFGS-B',
                    bounds=[(-30, 30), (40, 100), (-2, 2)],
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-6})


def run_monte_carlo(seed: int, dose: float, trials: int):
    rng = np.random.default_rng(seed + 1)
    h, jac = transient_and_jacobian(THETA_TRUE)
    designs = {'raster': np.eye(len(WALL_X)) * dose / len(WALL_X),
               'two_codes_alpha_0.8': complementary_codes(0.8, dose)}
    records = []
    for name, w in designs.items():
        mean, j = encode(w, h, jac)
        f = fisher(mean, j)
        estimates, failures = [], 0
        for _ in range(trials):
            counts = rng.poisson(mean)
            # 给各次重复独立的小扰动初值。这里只研究真值附近的局部估计性能。
            initial = THETA_TRUE + rng.normal(size=3) * [0.15, 0.15, 0.02]
            result = fit_counts(w, counts, initial)
            if not result.success:
                failures += 1
            else:
                estimates.append(result.x)
        if len(estimates) < 2:
            raise RuntimeError(f'{name}: 有效拟合次数不足')
        est = np.asarray(estimates)
        error = est[:, :2] - THETA_TRUE[:2]
        records.append({'design': name, 'trials': trials, 'fit_failures': failures,
                        'expected_signal_photons': float(mean.sum()),
                        'crlb_std_cm': crlb_std_cm(f).tolist(),
                        'empirical_std_cm': np.std(est[:, :2], axis=0, ddof=1).tolist(),
                        'bias_cm': error.mean(axis=0).tolist(),
                        'rmse_cm': np.sqrt(np.mean(error**2, axis=0)).tolist()})
    return records


def make_plot(output: Path, alphas, retention, crlb, raster_crlb):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(alphas, retention, color='tab:blue', label='Two-code effective geometry FI')
    axes[0].axhline(1, color='0.4', linestyle='--', label='Labeled/raster reference')
    axes[0].set(xlabel='Pattern separation alpha', ylabel='Minimum generalized FI ratio',
                title='Information retained under equal pointwise dose', ylim=(-0.03, 1.05))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].plot(alphas, np.asarray(crlb)[:, 0], label='Two codes: x')
    axes[1].plot(alphas, np.asarray(crlb)[:, 1], label='Two codes: z')
    axes[1].axhline(raster_crlb[0], color='tab:blue', linestyle='--', label='Raster: x')
    axes[1].axhline(raster_crlb[1], color='tab:orange', linestyle='--', label='Raster: z')
    axes[1].set(xlabel='Pattern separation alpha', ylabel='CRLB standard deviation (cm)',
                title='Unknown scattering gain is marginalized', yscale='log')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.savefig(output / 'information_bound.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--dose', type=float, default=2e6, help='总入射剂量，任意统一单位')
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.dose <= 0 or args.trials < 2:
        parser.error('--dose 必须为正，--trials 至少为 2')
    args.output.mkdir(parents=True, exist_ok=True)
    checks = run_checks(args.seed, args.dose)
    h, jac = transient_and_jacobian(THETA_TRUE)
    raster_w = np.eye(len(WALL_X)) * args.dose / len(WALL_X)
    fs = separated_fisher(raster_w, h, jac)
    eff_ref = effective_geometry(fs)
    # 白化参照矩阵：特征值范围 [0,1]，不受同一线性参数单位变换影响。
    val, vec = np.linalg.eigh(eff_ref)
    inv_sqrt = (vec * (1 / np.sqrt(val))) @ vec.T
    alphas = np.linspace(0.5, 1.0, 51)
    retention, crlb = [], []
    for alpha in alphas:
        fc = fisher(*encode(complementary_codes(alpha, args.dose), h, jac))
        normalized = inv_sqrt @ effective_geometry(fc) @ inv_sqrt
        retention.append(float(np.linalg.eigvalsh(normalized)[0]))
        std = crlb_std_cm(fc)
        crlb.append([None, None] if std is None else std.tolist())
    mc = run_monte_carlo(args.seed, args.dose, args.trials)
    report = {'model': 'static Gaussian pulse, x_cm/z_cm/log_gain; no background',
              'seed': args.seed, 'total_incident_dose': args.dose,
              'theta_true': THETA_TRUE.tolist(), 'checks': checks,
              'raster_fisher': fs.tolist(), 'raster_effective_geometry': eff_ref.tolist(),
              'raster_crlb_std_cm': crlb_std_cm(fs).tolist(),
              'alpha': alphas.tolist(), 'minimum_generalized_fi_ratio': retention,
              'coded_crlb_std_cm': crlb, 'monte_carlo': mc}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if not args.no_plots:
        make_plot(args.output, alphas, retention, crlb, crlb_std_cm(fs))
    print('PASS: analytic Jacobian vs finite differences; one-hot equality;')
    print('      30 nonnegative codes PSD bound + exact loss identity;')
    print('      nuisance-adjusted bound; photon budget; cancellation; equal-score equality.')
    for row in mc:
        print(row['design'], 'photons=', round(row['expected_signal_photons']),
              'CRLB std(cm)=', np.round(row['crlb_std_cm'], 5),
              'empirical std(cm)=', np.round(row['empirical_std_cm'], 5),
              'failures=', row['fit_failures'])
    print('Results:', args.output.resolve())


if __name__ == '__main__':
    main()
