"""Numerical checks for the research proposals; no diffusion results are claimed.

Run: python /home/shx/code/ce-new/methods/rethink_math_checks.py
Dependencies: numpy, scipy. Fixed seed, CPU only.
"""

from pathlib import Path
import json

import numpy as np
from scipy.linalg import null_space


def constrained_minimum(h, a, b):
    ha = np.linalg.solve(h, a.T)
    return ha @ np.linalg.solve(a @ ha, b)


def fid_covariance(x, y):
    """FID with unbiased covariances, using independent PSD eigensolvers."""
    mx, my = x.mean(0), y.mean(0)
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    v, u = np.linalg.eigh(cx)
    sx = (u * np.sqrt(np.maximum(v, 0))) @ u.T
    v2 = np.linalg.eigvalsh(sx @ cy @ sx)
    return float(np.sum((mx - my) ** 2) + np.trace(cx + cy)
                 - 2 * np.sqrt(np.maximum(v2, 0)).sum())


def main():
    rng = np.random.default_rng(20261021)
    report = {
        "scope": "Synthetic finite-dimensional algebra only; no diffusion model evaluation.",
        "seed": 20261021,
    }

    # Compare the proposed formula with an independently assembled KKT system.
    p, k, nr = 24, 5, 48
    jr = rng.normal(size=(nr, p))
    a = rng.normal(size=(k, p))
    b = rng.normal(size=k)
    h = jr.T @ jr / nr + 0.2 * np.eye(p)
    u = constrained_minimum(h, a, b)
    kkt = np.block([[h, a.T], [a, np.zeros((k, k))]])
    ukkt = np.linalg.solve(kkt, np.r_[np.zeros(p), b])[:p]
    n = null_space(a)
    comparisons = [u + n @ rng.normal(size=n.shape[1]) for _ in range(100)]
    gaps = [float(v @ h @ v - u @ h @ u) for v in comparisons]
    assert np.linalg.norm(a @ u - b) < 1e-10
    assert np.linalg.norm(u - ukkt) < 1e-10
    assert min(gaps) > -1e-10
    report["minimum_retention_cost"] = {
        "constraint_error": float(np.linalg.norm(a @ u - b)),
        "difference_from_kkt": float(np.linalg.norm(u - ukkt)),
        "minimum_cost_gap_of_100_feasible_perturbations": min(gaps),
    }

    # A real tradeoff: parameter norm and output distortion need not agree.
    j2 = np.diag([100.0, 1.0])
    a2 = np.array([[1.0, 1.0]])
    b2 = np.array([1.0])
    ordinary = constrained_minimum(np.eye(2), a2, b2)
    functional = constrained_minimum(j2.T @ j2, a2, b2)
    assert np.linalg.norm(j2 @ functional) < np.linalg.norm(j2 @ ordinary)
    report["parameter_vs_function_metric_example"] = {
        "euclidean_update": ordinary.tolist(),
        "functional_update": functional.tolist(),
        "euclidean_retained_squared_error": float(np.linalg.norm(j2 @ ordinary) ** 2),
        "functional_retained_squared_error": float(np.linalg.norm(j2 @ functional) ** 2),
        "warning": "Constructed linear example, not evidence of real-model performance.",
    }

    # Infeasible constraints must not be silently described as exact erasure.
    abad = np.array([[1.0, 0.0], [1.0, 0.0]])
    bbad = np.array([1.0, -1.0])
    ubad = abad.T @ np.linalg.pinv(abad @ abad.T) @ bbad
    incompatible_residual = float(np.linalg.norm(abad @ ubad - bbad))
    assert incompatible_residual > 1.0
    report["infeasible_constraint_example"] = {
        "residual": incompatible_residual,
        "interpretation": "A pseudoinverse does not make inconsistent constraints feasible.",
    }

    # Fixed-basis repair at a new operating point, keeping erasure scores fixed.
    anew = rng.normal(size=(k, p))
    nnew = null_space(anew)
    jnew = rng.normal(size=(nr, p))
    measured_drift = rng.normal(size=nr)
    lam = 0.1
    jn = jnew @ nnew
    v = -np.linalg.solve(jn.T @ jn + lam * np.eye(jn.shape[1]),
                         jn.T @ measured_drift)
    correction = nnew @ v
    before = float(np.linalg.norm(measured_drift) ** 2)
    after = float(np.linalg.norm(measured_drift + jnew @ correction) ** 2)
    assert np.linalg.norm(anew @ correction) < 1e-10
    assert after <= before + 1e-10
    report["forget_protected_repair"] = {
        "linearized_erasure_change": float(np.linalg.norm(anew @ correction)),
        "retained_squared_drift_before": before,
        "retained_squared_drift_after": after,
        "regularized_objective_after": after + lam * float(correction @ correction),
    }

    # Exact sampler derivative in a finite linear system, sharing one patch u
    # across all steps. Compare to finite differences of the full rollout.
    dim, nsteps, psmall = 7, 5, 4
    ms = [np.eye(dim) + 0.07 * rng.normal(size=(dim, dim)) for _ in range(nsteps)]
    bs = [rng.normal(size=(dim, psmall)) for _ in range(nsteps)]
    x0 = rng.normal(size=dim)

    def rollout(value):
        state = x0.copy()
        for mk, bk in zip(ms, bs):
            state = mk @ state + bk @ value
        return state

    sensitivity = np.zeros((dim, psmall))
    for mk, bk in zip(ms, bs):
        sensitivity = mk @ sensitivity + bk
    delta = 1e-5
    finite_difference = np.stack([
        (rollout(delta * np.eye(psmall)[j]) - rollout(-delta * np.eye(psmall)[j])) / (2 * delta)
        for j in range(psmall)
    ], axis=1)
    assert np.linalg.norm(sensitivity - finite_difference) < 1e-8
    report["shared_weight_sampler_response"] = {
        "full_rollout_derivative_error": float(np.linalg.norm(sensitivity - finite_difference)),
        "error_when_ignoring_state_propagation": float(np.linalg.norm(sum(bs) - sensitivity)),
    }

    # Nonlinear compact-support features induce a valid PSD kernel in any
    # input dimension. Compare null-space regression to the Schur kernel.
    centers = np.linspace(-2, 2, 10)[:, None]
    radius = 0.85

    def phi(x):
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        return np.maximum(1 - distances / radius ** 2, 0) ** 2

    xr = np.array([[-1.7], [0.1], [1.8]])
    xf = np.array([[-1.1], [-0.6], [0.65], [1.25]])
    pr, pf = phi(xr), phi(xf)
    nrk = null_space(pr)
    proj = nrk @ nrk.T
    krr = pr @ pr.T
    kfr = pf @ pr.T
    conditional_kernel = pf @ pf.T - kfr @ np.linalg.pinv(krr) @ kfr.T
    assert np.linalg.norm(conditional_kernel - pf @ proj @ pf.T) < 1e-10
    y = rng.normal(size=(len(xf), 3))
    ridge = 0.03
    coef = proj @ pf.T @ np.linalg.solve(conditional_kernel + ridge * np.eye(len(xf)), y)
    design = pf @ nrk
    direct = nrk @ np.linalg.solve(design.T @ design + ridge * np.eye(nrk.shape[1]), design.T @ y)
    outside = phi(np.array([[20.0], [-20.0]])) @ coef
    assert np.linalg.norm(pr @ coef) < 1e-10
    assert np.linalg.norm(coef - direct) < 1e-9
    assert np.count_nonzero(outside) == 0
    report["conditional_local_kernel"] = {
        "retain_anchor_correction_norm": float(np.linalg.norm(pr @ coef)),
        "difference_from_direct_constrained_regression": float(np.linalg.norm(coef - direct)),
        "outside_support_correction_norm": float(np.linalg.norm(outside)),
        "minimum_conditional_kernel_eigenvalue": float(np.linalg.eigvalsh(conditional_kernel).min()),
    }

    # Verify the finite-sample FID bound, with the same n-1 covariance
    # convention as the existing project's evaluator.
    fid_trials = []
    for count, width in [(8, 13), (30, 5), (50, 10)]:
        x = rng.normal(size=(count, width))
        y = x @ (np.eye(width) + 0.15 * rng.normal(size=(width, width)))
        y += 0.3 * rng.normal(size=(count, width)) + 0.2
        d = y - x
        mean = d.mean(0)
        paired_mse = float(np.sum(d ** 2) / count)
        tight_bound = float(np.sum(mean ** 2) + np.sum((d - mean) ** 2) / (count - 1))
        loose_bound = count / (count - 1) * paired_mse
        fid = fid_covariance(x, y)
        assert fid <= tight_bound + 1e-6
        assert tight_bound <= loose_bound + 1e-10
        fid_trials.append({
            "n": count, "dimension": width, "fid": fid,
            "paired_displacement_bound": tight_bound,
            "scaled_mse_bound": loose_bound,
        })
    report["finite_sample_fid_bound"] = fid_trials
    report["all_checks_passed"] = True
    destination = Path(__file__).with_name("rethink_math_checks_20261021.json")
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
