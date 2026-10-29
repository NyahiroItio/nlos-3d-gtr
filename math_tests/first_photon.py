"""首光子计时与理想 Poisson 计数：独立 CPU 数学验证。

运行 python math_tests/first_photon.py；完整推导见 FIRST_PHOTON.md。
不导入 3D-GTR，不需要 GPU。输入是一个随局部形变 q 移动的时间峰。
这里的“首光子”是每个脉冲周期最多保留一次事件，然后累积许多周期；
不是整个墙面位置收到一个光子就停止的 First-Photon Imaging 协议。

所有比较固定脉冲数和潜在事件过程，显式保留无事件周期。
假定每周期开始时探测器已恢复，不含跨周期死时间或 afterpulsing。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr


EDGES_PS = np.arange(0.0, 8192.1, 16.0)
SIGMA_PS = 60.0
TARGET_MU_PS = 5000.0
# q 的单位 mm。等效两段光程的轴向变化给出 dmu/dq=2/c。
# 它是单一预定义形变模式的时间峰代理，不是完整气球表面模型。
KAPPA_PS_PER_MM = 2e-3 / 2.99792458e-4
PARAMETERS = ['q_mm', 'log_signal_gain']
THETA_TRUE = np.zeros(2)
Q_BOUNDS = (-25.0, 25.0)
GAIN_BOUNDS = (-8.0, 5.0)
SCENARIOS = [
    {'name': 'low_flux', 'signal': 0.02, 'early': 0.0, 'background': 0.002},
    {'name': 'moderate_flux', 'signal': 0.5, 'early': 0.0, 'background': 0.002},
    {'name': 'high_flux', 'signal': 3.0, 'early': 0.0, 'background': 0.002},
    {'name': 'early_wall', 'signal': 0.02, 'early': 3.0, 'background': 0.002},
]


def gaussian_bins(mu_ps, sigma_ps):
    """精确积分时间箱，返回概率质量及对 mu 的导数。

    h_k=Phi((hi-mu)/sigma)-Phi((lo-mu)/sigma)。
    dh_k/dmu=[normal_pdf(lo)-normal_pdf(hi)]/sigma。
    正尾部使用生存函数差，避免两个接近 1 的 CDF 相减。
    """
    lo = (EDGES_PS[:-1] - mu_ps) / sigma_ps
    hi = (EDGES_PS[1:] - mu_ps) / sigma_ps
    mass = np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
    pdf = lambda z: np.exp(-0.5*z*z) / np.sqrt(2*np.pi)
    derivative = (pdf(lo) - pdf(hi)) / sigma_ps
    return mass, derivative


def latent_response(theta, scenario):
    """返回每周期潜在可检测事件均值 nu[k] 和 J[k,a]。

    nu=signal*exp(g)*h_target(q)+early*h_early+background/K。
    q 只移动目标峰；g 只缩放目标，不缩放已知早期墙面返回或背景。
    激光/探测效率已包含在这些“潜在事件”均值中，首光子筛选尚未作用。
    时间零点、两个峰的宽度、早期峰和背景强度均视为已标定。
    """
    q_mm, log_gain = theta
    h, dh = gaussian_bins(TARGET_MU_PS + KAPPA_PS_PER_MM*q_mm, SIGMA_PS)
    early, _ = gaussian_bins(1000.0, 80.0)
    target = scenario['signal'] * np.exp(log_gain) * h
    nu = target + scenario['early']*early + scenario['background']/h.size
    jac = np.column_stack([
        scenario['signal'] * np.exp(log_gain) * dh * KAPPA_PS_PER_MM,
        target,
    ])
    return nu, jac


def first_distribution(nu, jac):
    """首事件分类概率、Jacobian；最后一个类别是“本周期无事件”。

    L_k=sum_{j<k}nu_j，S_k=exp(-L_k)，a_k=1-exp(-nu_k)。
    p_k=S_k*a_k；p_empty=exp(-sum nu)。
    dp_k=S_k*[exp(-nu_k)*dnu_k-a_k*sum_{j<k}dnu_j]。
    dp_empty=-p_empty*sum dnu。
    这些式子由独立 Poisson 增量推导，不假设输出计数也是 Poisson。
    """
    prefix = np.r_[0.0, np.cumsum(nu[:-1])]
    prefix_jac = np.vstack([np.zeros(jac.shape[1]), np.cumsum(jac[:-1], axis=0)])
    survival = np.exp(-prefix)
    hit = -np.expm1(-nu)
    bins = survival * hit
    d_bins = survival[:, None] * (
        np.exp(-nu)[:, None]*jac - hit[:, None]*prefix_jac
    )
    empty = np.exp(-nu.sum())
    probability = np.r_[bins, empty]
    derivative = np.vstack([d_bins, -empty*jac.sum(axis=0)])
    return probability, derivative


def weighted_fisher(mean, jac, repetitions=1.0):
    """sum J_i J_i^T / mean_i。

    mean=nu 时是每周期理想 Poisson FI；mean=p（含空周期）时是分类 FI。
    重复 n 个独立周期，两者分别乘 n。只跳过 float64 无法表示的尾部。
    """
    keep = mean > 1e-280
    white = jac[keep] / np.sqrt(mean[keep, None])
    return repetitions * white.T @ white


def first_fisher_hazard(nu, jac, repetitions=1.0):
    """独立于 dp 公式的第二种首光子 FI 推导，用来交叉核验。

    在仍未触发的条件下，第 k 箱是 Bernoulli(a_k=1-exp(-nu_k))。
    各条件 score 是鞅差，交叉期望为 0，所以
      F_first=n*sum_k exp(-L_k)/(exp(nu_k)-1) * J_k J_k^T。
    这是离散时间箱的精确式，不需要用连续时间近似。
    系数比理想 1/nu_k 小，因而 F_first <= F_ideal（PSD 意义）。
    """
    prefix = np.r_[0.0, np.cumsum(nu[:-1])]
    keep = nu > 1e-280
    coefficient = np.exp(-prefix[keep]-nu[keep]) / (-np.expm1(-nu[keep]))
    white = jac[keep] * np.sqrt(coefficient[:, None])
    return repetitions * white.T @ white


def information_summary(fisher):
    """消去未知 log_signal_gain；不加 ridge 人为制造可辨识性。

    Feff=Fqq-Fqg^2/Fgg；CRLB std(q)=1/sqrt(Feff)，单位 mm。
    本例 target 总潜在光子数不随 q 变，但首光子竞争会引入 q/g 耦合。
    """
    if fisher[1, 1] <= 0:
        raise ValueError('Signal gain is not locally identifiable.')
    effective = float(fisher[0, 0] - fisher[0, 1]**2/fisher[1, 1])
    if effective <= 0:
        raise ValueError('Deformation is not locally identifiable in this configuration.')
    return {
        'fisher_matrix': fisher.tolist(),
        'effective_fisher_per_mm2': effective,
        'crlb_std_mm': float(1/np.sqrt(effective)),
    }


def expected_summary(scenario, pulses):
    nu, jac = latent_response(THETA_TRUE, scenario)
    p, dp = first_distribution(nu, jac)
    ideal = information_summary(weighted_fisher(nu, jac, pulses))
    first = information_summary(weighted_fisher(p, dp, pulses))
    centers = (EDGES_PS[:-1]+EDGES_PS[1:])/2
    window = (centers >= 4000) & (centers < 6200)
    return {
        'scenario': scenario,
        'pulses': pulses,
        'latent_events_per_pulse': float(nu.sum()),
        'probability_no_event': float(p[-1]),
        'expected_ideal_total_events': float(pulses*nu.sum()),
        'expected_first_total_events': float(pulses*p[:-1].sum()),
        'expected_ideal_target_window_events': float(pulses*nu[window].sum()),
        'expected_first_target_window_events': float(pulses*p[:-1][window].sum()),
        'ideal': ideal,
        'first': first,
        'effective_information_fraction': (
            first['effective_fisher_per_mm2']/ideal['effective_fisher_per_mm2']
        ),
    }


def fit_counts(counts, scenario, pulses, model):
    """正确 Poisson / 正确首光子 / 错误 Poisson 拟合共用同一个几何模型。

    model='poisson': lambda=n*nu，NLL=sum(lambda-H*log(lambda))。
    model='first': H 含空类别，NLL=-sum(H*log(p))。
      log p_k=-L_k+log(1-exp(-nu_k))；log p_empty=-sum(nu)。
      dlog p_k=dnu_k/(exp(nu_k)-1)-sum_{j<k}dnu_j。
    在首光子数据上选 'poisson' 就是有意制造观测模型失配。
    拟合 q 和 log_gain，允许错误模型把丢失计数解释成增益下降。
    初值从数据计算，不把真实 q 当作初值；解析梯度用于 L-BFGS-B。
    """
    bins = counts[:-1] if model == 'first' else counts
    centers = (EDGES_PS[:-1]+EDGES_PS[1:])/2
    window = (centers >= 4000) & (centers < 6200)
    weight = bins[window]
    if weight.sum() > 0:
        q_init = (np.dot(weight, centers[window])/weight.sum()-TARGET_MU_PS)/KAPPA_PS_PER_MM
    else:
        q_init = 0.0
    gain_init = np.log(max(float(weight.sum()), 0.01)/(pulses*scenario['signal']))
    initial = [np.clip(q_init, -20, 20), np.clip(gain_init, -7, 4)]
    scale = max(pulses*scenario['signal'], 1.0)

    def objective(theta):
        nu, jac = latent_response(theta, scenario)
        if model == 'poisson':
            mean = pulses*nu
            # 减去只依赖数据的饱和模型 NLL，避免常数过大导致线搜索失精。
            # 半 Poisson deviance=sum[lambda-H+H*log(H/lambda)]。
            positive = counts > 0
            value = mean[~positive].sum()
            ratio = (mean[positive]-counts[positive])/counts[positive]
            value += np.dot(counts[positive], ratio-np.log1p(ratio))
            gradient = ((pulses-counts/nu)[:, None]*jac).sum(axis=0)
        elif model == 'first':
            prefix = np.r_[0.0, np.cumsum(nu[:-1])]
            prefix_jac = np.vstack([np.zeros(2), np.cumsum(jac[:-1], axis=0)])
            hit = -np.expm1(-nu)
            log_probability = np.r_[-prefix+np.log(hit), -nu.sum()]
            log_jac = np.vstack([
                (np.exp(-nu)/hit)[:, None]*jac-prefix_jac,
                -jac.sum(axis=0),
            ])
            positive = counts > 0
            value = np.dot(counts[positive],
                           np.log(counts[positive]/pulses)-log_probability[positive])
            gradient = -(counts[:, None]*log_jac).sum(axis=0)
        else:
            raise ValueError(model)
        return float(value/scale), gradient/scale

    result = minimize(objective, initial, method='L-BFGS-B', jac=True,
                      bounds=[Q_BOUNDS, GAIN_BOUNDS],
                      options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': 250})
    retried = not result.success
    if retried:
        # 极少数近极值点的线搜索可能因 float64 精度退出。
        # 使用另一种有界算法重新求解，真实报告重试次数；不修改 success 标志。
        fallback = minimize(lambda x: objective(x)[0], result.x,
                            method='Nelder-Mead', bounds=[Q_BOUNDS, GAIN_BOUNDS],
                            options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-10})
        if fallback.success and fallback.fun <= result.fun+1e-9:
            result = fallback
    result['retried'] = retried
    at_bound = bool(
        min(result.x[0]-Q_BOUNDS[0], Q_BOUNDS[1]-result.x[0]) < 1e-4
        or min(result.x[1]-GAIN_BOUNDS[0], GAIN_BOUNDS[1]-result.x[1]) < 1e-4
    )
    return result, at_bound


def run_checks(seed):
    """核验概率、导数、两种 FI 推导和独立可解析边界情况。"""
    rng = np.random.default_rng(seed+17)
    max_derivative_error = 0.0
    max_hazard_error = 0.0
    min_scaled_psd_eigenvalue = 1.0
    # 从潜在 Poisson 增量逐周期生成事件，再执行首个非空箱筛选。
    # 独立于 multinomial 模拟器，检查探测协议确实产生所推导的 p。
    simulated_cycles = 200000
    physical_nu = np.array([0.2, 0.35, 0.05, 0.1])
    arrivals = rng.poisson(physical_nu, size=(simulated_cycles, physical_nu.size))
    hit = arrivals > 0
    selected = np.where(hit.any(axis=1), hit.argmax(axis=1), physical_nu.size)
    observed = np.bincount(selected, minlength=physical_nu.size+1)
    physical_p, _ = first_distribution(physical_nu, np.ones((physical_nu.size, 1)))
    residual = (observed-simulated_cycles*physical_p) / np.sqrt(
        simulated_cycles*physical_p*(1-physical_p)
    )
    max_physical_zscore = float(np.max(np.abs(residual)))
    assert max_physical_zscore < 6.0
    for scenario in SCENARIOS:
        theta = np.array([1.7, -0.2])
        nu, jac = latent_response(theta, scenario)
        p, dp = first_distribution(nu, jac)
        assert np.isclose(p.sum(), 1.0, atol=2e-14)
        assert np.allclose(dp.sum(axis=0), 0.0, atol=2e-14)
        for a, step in enumerate([1e-3, 1e-5]):
            delta = np.zeros(2)
            delta[a] = step
            plus, _ = first_distribution(*latent_response(theta+delta, scenario))
            minus, _ = first_distribution(*latent_response(theta-delta, scenario))
            fd = (plus-minus)/(2*step)
            error = np.linalg.norm(fd-dp[:, a])/np.linalg.norm(dp[:, a])
            max_derivative_error = max(max_derivative_error, float(error))
            assert error < 2e-6
        f = weighted_fisher(p, dp)
        hazard = first_fisher_hazard(nu, jac)
        error = np.linalg.norm(f-hazard)/np.linalg.norm(f)
        max_hazard_error = max(max_hazard_error, float(error))
        assert error < 1e-11

    # 任意非负时间响应与多参数导数：矩阵不等式，而不只比较标量 CRLB。
    for _ in range(80):
        nu = np.exp(rng.uniform(-8, 1, 15))
        jac = nu[:, None]*rng.normal(size=(15, 3))
        ideal = weighted_fisher(nu, jac)
        first = weighted_fisher(*first_distribution(nu, jac))
        scaled_min = np.linalg.eigvalsh(ideal-first).min()/np.linalg.norm(ideal)
        min_scaled_psd_eigenvalue = min(min_scaled_psd_eigenvalue, float(scaled_min))
        assert scaled_min > -1e-12
        # 离散箱也有 e^-mu F_ideal <= F_first <= F_ideal。
        assert np.linalg.eigvalsh(first-np.exp(-nu.sum())*ideal).min() > -1e-11

    # 一箱模型：观测只剩是否发生事件，精确 F=1/(exp(mu)-1)，dmu/dtheta=1。
    for mu in [1e-4, 0.2, 3.0, 10.0]:
        got = weighted_fisher(*first_distribution(np.array([mu]), np.ones((1, 1))))[0, 0]
        assert np.isclose(got, 1/np.expm1(mu), rtol=1e-12)

    # 稀有事件极限，不是“目标弱就足够”，而是整窗口 mu 足够小。
    nu = np.array([0.7, 0.2, 0.1])*1e-7
    jac = nu[:, None]*rng.normal(size=(3, 2))
    ideal = weighted_fisher(nu, jac)
    first = weighted_fisher(*first_distribution(nu, jac))
    rare_error = float(np.linalg.norm(first-ideal)/np.linalg.norm(ideal))
    assert rare_error < 1e-7

    # 完全不重叠的早期事件，且其强度已知：对后期参数的所有 FI 恰乘 e^-a。
    late_nu = np.array([0.01, 0.03, 0.02])
    late_jac = np.array([[0.001, 0.01], [-0.002, 0.03], [0.001, 0.02]])
    reference = weighted_fisher(*first_distribution(late_nu, late_jac))
    early_scale_error = 0.0
    for early in [0.1, 1.0, 3.0, 6.0]:
        nu = np.r_[early, late_nu]
        jac = np.vstack([np.zeros(2), late_jac])
        got = weighted_fisher(*first_distribution(nu, jac))
        error = np.linalg.norm(got-np.exp(-early)*reference)/np.linalg.norm(reference)
        early_scale_error = max(early_scale_error, float(error))
        assert error < 1e-12

    # 用期望直方图而非随机抽样独立检验拟合器和模型失配。
    expected_fit_error = 0.0
    asymptotic_naive_bias = {}
    for scenario in SCENARIOS:
        pulses = 50000
        nu, jac = latent_response(THETA_TRUE, scenario)
        p, _ = first_distribution(nu, jac)
        for data, model in [(pulses*nu, 'poisson'), (pulses*p, 'first')]:
            fit, at_bound = fit_counts(data, scenario, pulses, model)
            expected_fit_error = max(expected_fit_error, float(np.max(np.abs(fit.x))))
            assert fit.success and not at_bound
            assert np.max(np.abs(fit.x)) < 1e-3
        naive, at_bound = fit_counts(pulses*p[:-1], scenario, pulses, 'poisson')
        assert naive.success and not at_bound
        asymptotic_naive_bias[scenario['name']] = {
            'q_bias_mm': float(naive.x[0]), 'log_signal_gain': float(naive.x[1]),
        }
    assert asymptotic_naive_bias['high_flux']['q_bias_mm'] < -1.0
    return {
        'direct_poisson_to_first_event_simulation_cycles': simulated_cycles,
        'max_direct_simulation_probability_zscore': max_physical_zscore,
        'max_first_probability_jacobian_relative_error': max_derivative_error,
        'max_probability_vs_hazard_fisher_relative_error': max_hazard_error,
        'random_psd_checks': 80,
        'min_scaled_eigenvalue_ideal_minus_first': min_scaled_psd_eigenvalue,
        'rare_event_fisher_relative_error': rare_error,
        'early_survival_scaling_error': early_scale_error,
        'max_expected_histogram_fit_parameter_error': expected_fit_error,
        'asymptotic_naive_poisson_fit': asymptotic_naive_bias,
    }


def monte_carlo(scenario, pulses, trials, rng):
    """独立模拟两个探测器；同 n、同 nu，不强行匹配记录计数。

    理想：H_k~Poisson(n*nu_k)。首事件：(H_k,H_empty)~Multinomial(n,p)。
    首事件数据分别用正确多项式似然和错误 Poisson 似然拟合。
    统计全部重复的 bias/std/RMSE，同时报告优化失败和边界命中。
    """
    nu, jac = latent_response(THETA_TRUE, scenario)
    p, _ = first_distribution(nu, jac)
    # numpy 要求概率和不超过 1；最后空类别由模拟器取剩余概率，
    # 浮点误差仅在 1e-15 量级，解析概率本身不作裁剪或重新归一化。
    records = {name: [] for name in ['ideal_poisson', 'first_correct', 'first_naive_poisson']}
    failures = dict.fromkeys(records, 0)
    retries = dict.fromkeys(records, 0)
    boundaries = dict.fromkeys(records, 0)
    for _ in range(trials):
        ideal_data = rng.poisson(pulses*nu)
        first_data = rng.multinomial(pulses, p)
        tasks = [
            ('ideal_poisson', ideal_data, 'poisson'),
            ('first_correct', first_data, 'first'),
            ('first_naive_poisson', first_data[:-1], 'poisson'),
        ]
        for name, data, model in tasks:
            fit, at_bound = fit_counts(data, scenario, pulses, model)
            records[name].append(fit.x.tolist())
            failures[name] += int(not fit.success)
            retries[name] += int(fit.retried)
            boundaries[name] += int(at_bound)
    output = {}
    for name, estimates in records.items():
        values = np.asarray(estimates)
        q = values[:, 0]
        output[name] = {
            'trials': trials,
            'q_bias_mm': float(q.mean()),
            'q_std_mm': float(q.std(ddof=1)),
            'q_rmse_mm': float(np.sqrt(np.mean(q*q))),
            'bias_standard_error_mm': float(q.std(ddof=1)/np.sqrt(trials)),
            'mean_log_signal_gain': float(values[:, 1].mean()),
            'optimizer_failures': failures[name],
            'optimizer_retries': retries[name],
            'boundary_hits': boundaries[name],
        }
    return output


def fixed_dose_sweep():
    """同总照明剂量，分摊到更多周期：理想 FI 不变，首光子损失下降。

    固定潜在目标事件总均值 1000；早期墙面事件与照明一起缩放。
    本对照不加环境光（否则增加时间也会增加背景），不计切换开销。
    n*nu_target 与 n*nu_early 均固定，不是偷偷增加输入光子预算。
    """
    pulse_counts = np.unique(np.rint(np.logspace(3, 6, 70)).astype(int))
    rows = []
    for ratio in [0.0, 20.0]:
        values = []
        for pulses in pulse_counts:
            scenario = {'signal': 1000/pulses, 'early': ratio*1000/pulses, 'background': 0.0}
            nu, jac = latent_response(THETA_TRUE, scenario)
            ideal = information_summary(weighted_fisher(nu, jac, pulses))
            first = information_summary(first_fisher_hazard(nu, jac, pulses))
            values.append(first['effective_fisher_per_mm2']/ideal['effective_fisher_per_mm2'])
        assert np.all(np.diff(values) > -1e-10)
        rows.append({'early_to_target_ratio': ratio, 'information_fraction': values})
    return {'pulses': pulse_counts.tolist(), 'total_latent_target_events': 1000.0, 'curves': rows}


def plot_results(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    centers = (EDGES_PS[:-1]+EDGES_PS[1:])/2
    ax = axes[0, 0]
    for scenario, color in [(SCENARIOS[0], 'C0'), (SCENARIOS[2], 'C3')]:
        nu, jac = latent_response(THETA_TRUE, scenario)
        p, _ = first_distribution(nu, jac)
        ax.plot(centers, nu/nu.sum(), '--', color=color, alpha=0.7,
                label=f"{scenario['name']}: ideal")
        ax.plot(centers, p[:-1]/p[:-1].sum(), color=color,
                label=f"{scenario['name']}: first event")
    ax.set(xlim=(4700, 5300), xlabel='Flight time (ps)', ylabel='Normalized recorded histogram',
           title='Shape distortion: same latent process')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    rows = report['scenarios']
    x = np.arange(len(rows))
    labels = ['Low flux', 'Moderate', 'High flux', 'Early wall']
    fractions = [r['effective_information_fraction'] for r in rows]
    bars = ax.bar(x, fractions, color=['C0', 'C1', 'C3', 'C4'])
    for bar, value in zip(bars, fractions):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.025, f'{value:.3f}', ha='center')
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.15), ylabel='Effective Fisher / ideal Fisher',
           title='Unknown target gain; calibrated time zero')

    ax = axes[1, 0]
    names = ['ideal_poisson', 'first_correct', 'first_naive_poisson']
    for i, (name, color, label) in enumerate(zip(
            names, ['C0', 'C2', 'C3'], ['Ideal + Poisson fit', 'First + correct fit', 'First + Poisson fit'])):
        ax.bar(x+(i-1)*0.24, [r['monte_carlo'][name]['q_rmse_mm'] for r in rows],
               width=0.24, color=color, label=label)
    ax.plot(x-0.24, [r['ideal']['crlb_std_mm'] for r in rows], 'k_', markersize=14,
            label='Correct-model CRLB')
    ax.plot(x, [r['first']['crlb_std_mm'] for r in rows], 'k_', markersize=14)
    ax.set(xticks=x, xticklabels=labels, ylabel='Deformation RMSE (mm)', yscale='log',
           title=f"Monte Carlo: {report['trials']} repeats per scenario")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    sweep = report['fixed_dose_sweep']
    for curve in sweep['curves']:
        ax.loglog(sweep['pulses'], curve['information_fraction'],
                  label=f"Early / target = {curve['early_to_target_ratio']:g}")
    ax.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax.set(xlabel='Number of laser cycles', ylabel='Effective Fisher / ideal Fisher',
           title='Fixed total dose: spread photons over more cycles')
    ax.legend(fontsize=8)
    fig.suptitle('First-event SPAD model vs ideal Poisson counting', fontsize=14)
    fig.savefig(output/'first_photon_comparison.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--pulses', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent/'results'/'first_photon')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.trials < 2 or args.pulses < 1:
        parser.error('--trials must be >=2 and --pulses must be >=1')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        'seed': args.seed, 'trials': args.trials, 'pulses': args.pulses,
        'parameters': PARAMETERS,
        'model': {
            'target_center_ps': TARGET_MU_PS, 'target_sigma_ps': SIGMA_PS,
            'deformation_slope_ps_per_mm': KAPPA_PS_PER_MM,
            'time_bin_ps': float(EDGES_PS[1]-EDGES_PS[0]),
            'period_window_ps': float(EDGES_PS[-1]-EDGES_PS[0]),
            'time_zero_known': True, 'target_gain_unknown': True,
            'cross_cycle_dead_time': False, 'actual_balloon_geometry': False,
        },
        'checks': run_checks(args.seed),
        'fixed_dose_sweep': fixed_dose_sweep(),
        'scenarios': [],
    }
    rng = np.random.default_rng(args.seed)
    for scenario in SCENARIOS:
        row = expected_summary(scenario, args.pulses)
        row['monte_carlo'] = monte_carlo(scenario, args.pulses, args.trials, rng)
        report['scenarios'].append(row)
        print(f"{scenario['name']}: info fraction={row['effective_information_fraction']:.6f}, "
              f"CRLB ideal/first={row['ideal']['crlb_std_mm']:.4f}/{row['first']['crlb_std_mm']:.4f} mm, "
              f"naive bias={row['monte_carlo']['first_naive_poisson']['q_bias_mm']:.4f} mm", flush=True)
    with (args.output/'report.json').open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    if not args.no_plots:
        plot_results(report, args.output)
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
