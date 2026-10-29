"""发射端 phase-only SLM 的 CPU 验证；接收端只有 Poisson 瞬态计数。

运行 python math_tests/illumination_wavefront.py
完整推导见 ILLUMINATION_WAVEFRONT.md。

主模型：SLM e(phi) -> 固定相干传播 A -> 隐藏区域照明 |A e|^2
       -> 强度瞬态响应 h_jk(q) -> Poisson SPAD counts。
只在照明端保留干涉，不测量返回光的复相位，也不使用 RSD。
A 是可解析的理想散射混合示例，不是实际墙面标定数据。
q 是预定义表面形变模式的幅度(mm)，不是完整气球的半径。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import ndtr


M = 8
C_M_PER_PS = 2.99792458e-4
SIGMA_PS = 60.0
EDGES_PS = np.arange(3000.0, 6500.1, 16.0)
# 隐藏区域的基准等效往返距离；只是一个局部瞬态 toy 场景。
DEPTH_M = np.linspace(0.65, 0.85, M)
# q 每增加 1mm，区域 j 的等效径向距离增加 beta_j mm。
# beta 范围 [0,1]，dmu/dq=2*beta*1e-3/c，满足几何光程斜率界。
BETA = np.array([0.05, 0.08, 0.15, 0.30, 1.00, 0.60, 0.20, 0.10])
# 各区域的接收效率，特意让最亮区域与最敏感区域不同。
# beta/eta 是人为控制条件，不是任何真实目标的测量结果。
ETA = 1e-3 * np.array([0.8, 0.7, 0.5, 0.35, 0.1, 0.2, 0.4, 0.6])


def propagation(seed: int):
    """A = DFT * diag(exp(i*screen))：薄相位屏 + 理想离散传播。

    A^H A=I，所以总隐藏照明能量恒为1；SLM仅重新分配能量。
    |A_jl|=1/sqrt(M)，任一输出区域均能被 phase-only 输入精确聚焦。
    这种可解析性质使全局信息上界可以独立核验；不代表任意粗糙墙。
    """
    rng = np.random.default_rng(seed)
    j = np.arange(M)
    dft = np.exp(-2j * np.pi * np.outer(j, j) / M) / np.sqrt(M)
    screen = rng.uniform(-np.pi, np.pi, M)
    return dft * np.exp(1j * screen)[None, :]


def illumination(a, phase, visibility=1.0):
    """返回隐藏区域能量比例 w[j] 及 dw[j]/dphi[l]。

    e_l=exp(i phi_l)/sqrt(M)，u=Ae，w_coh=|u|^2。
    du_j/dphi_l=i*A_jl*e_l，dw_j/dphi_l=2Re(conj(u_j)*i*A_jl*e_l)。

    visibility=gamma 是一个明确的相位平均对照模型：
      C_in=gamma*e e^H+(1-gamma)*diag(|e|^2)
      w=diag(A C_in A^H)
       =gamma*|Ae|^2+(1-gamma)*sum_l |A_jl|^2/M。
    gamma=0 时交叉项消失，同一 phase-only SLM 不再改变照明。
    gamma 是假设的互相干权重，不是由激光脉宽直接算出的数。
    """
    e = np.exp(1j * phase) / np.sqrt(M)
    u = a @ e
    incoherent = (abs(a)**2).sum(axis=1) / M
    w = visibility * abs(u)**2 + (1 - visibility) * incoherent
    dw = visibility * 2 * np.real(np.conj(u)[:, None] * (1j * a * e[None, :]))
    return w, dw


def target_response(q_mm):
    """隐藏区域单位照明 -> 接收到的时间 bin 概率及 q 导数。

    mu_j(q)=2*(z_j+beta_j*q*1e-3)/c (ps)。
    h_jk=eta_j * integral_bin Normal(t;mu_j,sigma) dt。
    假定所选局部形变范围内照明场、接收效率与宽度不变，仅峰位变化。
    这让本验证可单独检查照明整形，真实场景应扩展 A(q),eta(q),sigma(q)。
    多个隐藏区域的返回强度相加：假定区域间返回干涉已被平均。
    该假定不否定同一区域上不同入射模式之间的照明干涉。
    """
    mu = 2 * (DEPTH_M + BETA * q_mm * 1e-3) / C_M_PER_PS
    slope = 2 * BETA * 1e-3 / C_M_PER_PS
    lo = (EDGES_PS[:-1][None, :] - mu[:, None]) / SIGMA_PS
    hi = (EDGES_PS[1:][None, :] - mu[:, None]) / SIGMA_PS
    mass = np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
    pdf = lambda x: np.exp(-x*x/2) / np.sqrt(2*np.pi)
    dmass = (pdf(lo) - pdf(hi)) / SIGMA_PS * slope[:, None]
    return ETA[:, None] * mass, ETA[:, None] * dmass


def moments(a, phase, n_in, q_mm=0.0, visibility=1.0):
    """lambda_k=N_in sum_j w_j h_jk；d_k=d lambda_k/dq。

    接收端没有任何相位控制变量，只观察一个探测通道的时间直方图。
    d lambda/dphi=N_in*(dw/dphi)^T h，用于设计发射端相位。
    """
    w, dw = illumination(a, phase, visibility)
    h, dh = target_response(q_mm)
    mean = n_in * w @ h
    derivative = n_in * w @ dh
    grad_mean = n_in * dw.T @ h
    grad_derivative = n_in * dw.T @ dh
    return mean, derivative, grad_mean, grad_derivative


def information(mean, derivative):
    """Poisson F_qq=sum_k d_k^2/lambda_k；同时消去未知 log_gain。

    log_gain 导数等于 lambda；所以 F_qg=sum_k d_k，F_gg=sum_k lambda_k。
    F_eff=F_qq-F_qg^2/F_gg。
    本例完整时间窗的总计数几乎与q无关，因此交叉项接近0。
    系统时间零点在本实验中已知；若也未知，单区域峰移可与之完全混淆。
    """
    keep = mean > 1e-250
    m, d = mean[keep], derivative[keep]
    f = float(np.sum(d*d/m))
    cross = float(d.sum())
    return f, max(0.0, f - cross*cross / m.sum())


def objective(a, phase, n_in, task):
    """任务一最大计数；任务二最大 F_qq；返回解析设计梯度。

    dF/dphi=sum_k [2*d*d_phi/lambda-d^2*lambda_phi/lambda^2]。
    当前场景 F_qg 近0，优化 F_qq 与优化有效几何信息数值等价，
    最终评价仍显式报告 Schur complement。
    """
    mean, d, gm, gd = moments(a, phase, n_in)
    if task == 'photons':
        return float(mean.sum()), gm.sum(axis=1)
    keep = mean > 1e-250
    m, d = mean[keep], d[keep]
    gm, gd = gm[:, keep], gd[:, keep]
    value = np.sum(d*d/m)
    grad = np.sum(2*d[None, :]*gd/m[None, :]
                  - (d/m)[None, :]**2 * gm, axis=1)
    return float(value), grad


def optimize_phase(a, n_in, task, seed):
    """8个随机初值的 phase-only 局部优化，不使用解析最优相位作初值。

    固定最后一个相位为0以消除全局相位冗余。
    用独立可证的全局上界评价gap，不把多初值优化本身当全局最优证明。
    """
    rng = np.random.default_rng(seed)
    flat = np.zeros(M)
    scale = max(objective(a, flat, n_in, task)[0], 1e-10)

    def fun(x):
        phase = np.r_[x, 0.0]
        value, grad = objective(a, phase, n_in, task)
        return -value/scale, -grad[:-1]/scale

    results = []
    for _ in range(8):
        res = minimize(fun, rng.uniform(-np.pi, np.pi, M-1), jac=True,
                       method='L-BFGS-B', options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9})
        results.append(res)
    best = min(results, key=lambda r: r.fun)
    return np.r_[best.x, 0.0], {'starts': len(results),
                              'successful_starts': sum(bool(r.success) for r in results)}


def focused_phase(a, j):
    """phi_l=-arg(A_jl) 给定行的 phase conjugation，可把全部照明投入j。
    只作为解析参考/上界可达性检查，不用于初始化局部优化器。
    """
    phase = -np.angle(a[j])
    return phase - phase[-1]


def run_checks(a, n_in, seed):
    rng = np.random.default_rng(seed + 99)
    assert np.allclose(a.conj().T @ a, np.eye(M), atol=1e-13)
    phase = rng.uniform(-np.pi, np.pi, M)
    w, dw = illumination(a, phase)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(dw.sum(axis=0), 0, atol=1e-13)
    h, dh = target_response(0.0)
    step = 1e-4
    fd = (target_response(step)[0] - target_response(-step)[0]) / (2*step)
    jac_error = float(np.linalg.norm(fd-dh)/np.linalg.norm(dh))
    assert jac_error < 1e-7
    phase_errors = {}
    for task in ['photons', 'information']:
        _, analytic = objective(a, phase, n_in, task)
        finite = []
        for l in range(M):
            delta = np.zeros(M)
            delta[l] = 1e-5
            finite.append((objective(a, phase+delta, n_in, task)[0]
                           - objective(a, phase-delta, n_in, task)[0]) / 2e-5)
        err = float(np.linalg.norm(np.array(finite)-analytic)/np.linalg.norm(analytic))
        assert err < 1e-6, (task, err)
        phase_errors[task] = err

    # 信息上界来自旧的非相干混合定理，但用在隐藏区域的返回混合上：
    # F <= N sum_j w_j I_j <= N max_j I_j；这里 sum_j w_j=1。
    per_patch = np.array([information(hj, dj)[0] for hj, dj in zip(h, dh)])
    for _ in range(50):
        p = rng.uniform(-np.pi, np.pi, M)
        f = information(*moments(a, p, n_in)[:2])[0]
        assert f <= n_in*per_patch.max()*(1+1e-10)
        w0, dw0 = illumination(a, p, 0.0)
        assert np.allclose(w0, np.ones(M)/M)
        assert np.allclose(dw0, 0)
    for j in range(M):
        wf, _ = illumination(a, focused_phase(a, j))
        assert np.isclose(wf[j], 1, atol=1e-12)

    # 边界检查：若只照亮一个patch，q引起的峰移动与未知时间零点完全共线。
    # d lambda/dt0 = (d lambda/dq)/slope_j。消去gain,t0后，q信息应为0。
    j = int(np.argmax(per_patch))
    wf, _ = illumination(a, focused_phase(a, j))
    mean = n_in * wf @ h
    dq = n_in * wf @ dh
    slopes = 2 * BETA * 1e-3 / C_M_PER_PS
    dt0 = n_in * wf @ (dh / slopes[:, None])
    keep = mean > 1e-250
    white = np.column_stack([dq[keep], mean[keep], dt0[keep]]) / np.sqrt(mean[keep, None])
    f = white.T @ white
    eff_with_t0 = float(f[0,0] - f[0,1:] @ np.linalg.solve(f[1:,1:],f[1:,0]))
    assert abs(eff_with_t0) < n_in*per_patch.max()*1e-10

    # 独立的理想两输出解析例子：输入/输出计数均匹配，却有不同参数信息。
    b = np.array([[1,1],[1,-1]]) / np.sqrt(2)
    hp = b @ np.diag([1j, 0])
    example = []
    for e in [np.array([1,0]), np.array([1,1j])/np.sqrt(2)]:
        field = b @ e
        dmean = 2*np.real(np.conj(field)*(hp@e))
        example.append(float(np.sum(dmean**2/abs(field)**2)))
    assert np.allclose(example, [0,1])
    return {'unitary_energy_conservation': True, 'q_jacobian_relative_error': jac_error,
            'phase_objective_gradient_errors': phase_errors,
            'random_phases_bound_checked': 50, 'incoherent_phase_independence': True,
            'focused_patch_effective_fisher_with_unknown_time_zero': eff_with_t0,
            'analytic_two_mode_information_per_input_photon': example}, per_patch


def monte_carlo(a, phase, n_in, trials, rng):
    """固定预先设计的照明，重复生成计数并估计q，profile掉未知gain。

    lambda_k(q,g)=g*b_k(q)，极大似然 g_hat(q)=sum(y)/sum(b(q))。
    把该值代回Poisson NLL，再做1D局部模型内的有界估计。
    q真值0mm，搜索区间[-20,20]mm属于toy模型，不是物理标定范围。
    """
    mean, derivative = moments(a, phase, n_in)[:2]
    estimates, failures = [], 0
    for _ in range(trials):
        counts = rng.poisson(mean)
        total = counts.sum()
        if total == 0:
            failures += 1
            continue

        def nll(q):
            base = moments(a, phase, n_in, q)[0]
            expected = base * total / base.sum()
            return float(np.sum(expected-counts*np.log(np.maximum(expected, 1e-250))))

        fit = minimize_scalar(nll, bounds=(-20,20), method='bounded',
                              options={'xatol': 1e-7, 'maxiter': 300})
        if fit.success and abs(fit.x) < 19.99:
            estimates.append(float(fit.x))
        else:
            failures += 1
    eff = information(mean, derivative)[1]
    values = np.asarray(estimates)
    if len(values) < 2:
        raise RuntimeError('Too few successful estimates')
    return {'trials': trials, 'failures': failures, 'crlb_std_mm': float(1/np.sqrt(eff)),
            'empirical_std_mm': float(values.std(ddof=1)), 'bias_mm': float(values.mean()),
            'rmse_mm': float(np.sqrt(np.mean(values**2)))}


def plot_results(output, a, designs, n_in, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(designs)
    fig, axes = plt.subplots(2,2,figsize=(11,7.3),constrained_layout=True)
    weights = np.stack([illumination(a,p)[0] for p in designs.values()])
    im = axes[0,0].imshow(weights, aspect='auto', vmin=0, vmax=1, cmap='Blues')
    axes[0,0].set(xticks=np.arange(M), yticks=np.arange(len(names)), yticklabels=names,
                  xlabel='Hidden patch index j', title='SLM changes illumination at the target')
    fig.colorbar(im, ax=axes[0,0], label='Fraction of input illumination')
    axes[0,1].bar(names, [r['expected_photons'] for r in records], color='tab:blue')
    axes[0,1].set(ylabel='Expected detected photons', title='Same input photon budget')
    axes[1,0].bar(names, [r['effective_fisher_per_detected_photon'] for r in records],color='tab:orange')
    axes[1,0].set(ylabel='Effective FI / photon (1/mm^2)', title='Brightness and information differ')
    gammas = np.linspace(0,1,51)
    for name,p in designs.items():
        vals=[information(*moments(a,p,n_in,visibility=g)[:2])[1] for g in gammas]
        axes[1,1].plot(gammas, vals, label=name)
    axes[1,1].set(xlabel='Coherence visibility gamma (model parameter)',
                  ylabel='Effective Fisher information (1/mm^2)',
                  title='Phase control disappears at gamma = 0')
    axes[1,1].legend(fontsize=8)
    for ax in [axes[0,1],axes[1,0]]:
        ax.tick_params(axis='x',labelrotation=15)
    for ax in axes.flat:
        if ax is not axes[0,0]:
            ax.grid(axis='y',alpha=0.18)
    fig.savefig(output/'illumination_comparison.png',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261027)
    parser.add_argument('--photons',type=float,default=2e7,help='Incident photon budget per acquisition')
    parser.add_argument('--trials',type=int,default=200)
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results'/'illumination_wavefront')
    args=parser.parse_args()
    if args.photons<=0 or not np.isfinite(args.photons) or args.trials<2:
        parser.error('photons must be finite and positive; trials >= 2')
    args.output.mkdir(parents=True,exist_ok=True)
    a=propagation(args.seed)
    checks, patch_info=run_checks(a,args.photons,args.seed)
    p_count, count_diag=optimize_phase(a,args.photons,'photons',args.seed+1)
    p_info, info_diag=optimize_phase(a,args.photons,'information',args.seed+2)
    designs={'Flat SLM':np.zeros(M),'Max photons':p_count,'Max information':p_info}
    rng=np.random.default_rng(args.seed+3)
    records=[]
    for name,phase in designs.items():
        mean,d=moments(a,phase,args.photons)[:2]
        raw,eff=information(mean,d)
        records.append({'design':name,'incident_photons':args.photons,
                        'expected_photons':float(mean.sum()),'fisher_q_known_gain':raw,
                        'effective_fisher_q_unknown_gain':eff,
                        'effective_fisher_per_detected_photon':float(eff/mean.sum()),
                        'hidden_illumination_weights':illumination(a,phase)[0].tolist(),
                        'phase_rad':phase.tolist(),
                        'monte_carlo':monte_carlo(a,phase,args.photons,args.trials,rng)})
    oracle=args.photons*patch_info.max()
    gap=1-records[-1]['fisher_q_known_gain']/oracle
    # 在这个特意可解的A中，聚焦到最佳patch可达到全局上界。
    # 算法若未找到它则报告gap，不隐瞒多初值优化的局限。
    check_phase=focused_phase(a,int(np.argmax(patch_info)))
    assert np.isclose(information(*moments(a,check_phase,args.photons)[:2])[0],oracle,rtol=1e-10)
    report={'scope':'illumination-side coherent shaping + incoherent transient reception',
            'seed':args.seed,'theta':'q_mm, local prescribed deformation amplitude',
            'known_parameters':['time zero','IRF width','A','ETA','BETA','reference depths'],
            'nuisance':'unknown global gain (profile likelihood)',
            'sigma_ps':SIGMA_PS,'beta':BETA.tolist(),'eta':ETA.tolist(),
            'checks':checks,'per_patch_fisher_per_incident_photon':patch_info.tolist(),
            'global_information_upper_bound':float(oracle),
            'optimized_information_relative_gap':float(gap),
            'optimization':{'photons':count_diag,'information':info_diag},'designs':records}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    if not args.no_plots:
        plot_results(args.output,a,designs,args.photons,records)
    print('PASS: energy conservation, q Jacobian, phase gradients, incoherent control, FI bound.')
    for row in records:
        mc=row['monte_carlo']
        print(row['design'], 'photons=',round(row['expected_photons']),
              'FI=',round(row['effective_fisher_q_unknown_gain'],6),
              'CRLB(mm)=',round(mc['crlb_std_mm'],6),
              'MC std(mm)=',round(mc['empirical_std_mm'],6),'failures=',mc['failures'])
    print('Relative gap to proved global FI bound:',gap)
    print('Output:',args.output.resolve())


if __name__=='__main__':
    main()
