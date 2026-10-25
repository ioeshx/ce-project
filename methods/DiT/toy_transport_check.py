#!/usr/bin/env python3
"""Analytic sanity checks for the CFG-PTE proposal; no image model is loaded.

The risk and retained statistic are linear terminal coordinates, NOT semantic
probabilities.  Exact Gaussian tilts let us check the signs, moment constraints,
and guidance dependence of the proposed continuum construction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cfg_pte_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cfg_pte_cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import expm


OUT = Path(__file__).resolve().parent / "toy_results"
BETA_MAX = 1.2
REFERENCE_GAMMA = 3.0
E_X = np.array([1.0, 0.0])
E_Y = np.array([0.0, 1.0])


def base_parameters(gamma: float) -> tuple[np.ndarray, np.ndarray]:
    # v0(z) = -0.12*z; vc(z) = (-0.12*I + B1)*z + a1.
    # Therefore B_gamma*z+a_gamma is exactly v0+gamma*(vc-v0).
    b1 = np.array([[0.0, 0.08], [0.08, 0.0]])
    a1 = np.array([0.5, 0.05])
    return -0.12 * np.eye(2) + gamma * b1, gamma * a1


def affine_flow(gamma: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    b, a = base_parameters(gamma)
    augmented = np.zeros((3, 3))
    augmented[:2, :2] = b
    augmented[:2, 2] = a
    flow = expm(duration * augmented)
    return flow[:2, :2], flow[:2, 2]


def base_moments(gamma: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    flow, mean = affine_flow(gamma, t)
    return mean, flow @ flow.T


def beta(t: float) -> float:
    return BETA_MAX * t * t


def beta_dot(t: float) -> float:
    return 2.0 * BETA_MAX * t


def residual_direction(gamma: float, retain: bool = True) -> np.ndarray:
    _, terminal_cov = base_moments(gamma, 1.0)
    ratio = terminal_cov[0, 1] / terminal_cov[1, 1] if retain else 0.0
    return E_X - ratio * E_Y


def exact_shift(gamma: float, t: float, retain: bool = True) -> np.ndarray:
    _, cov = base_moments(gamma, t)
    continuation, _ = affine_flow(gamma, 1.0 - t)
    return -beta(t) * cov @ continuation.T @ residual_direction(gamma, retain)


def exact_control(gamma: float, t: float, retain: bool = True) -> np.ndarray:
    _, cov = base_moments(gamma, t)
    continuation, _ = affine_flow(gamma, 1.0 - t)
    return -beta_dot(t) * cov @ continuation.T @ residual_direction(gamma, retain)


def integrate_mean(gamma: float, mode: str, times: np.ndarray) -> np.ndarray:
    b, a = base_parameters(gamma)

    def rhs(t: float, mean: np.ndarray) -> np.ndarray:
        if mode == "base":
            correction = np.zeros(2)
        elif mode == "constrained":
            correction = exact_control(gamma, t)
        elif mode == "unconstrained":
            correction = exact_control(gamma, t, retain=False)
        elif mode == "x_only":
            correction = exact_control(gamma, t).copy()
            correction[1] = 0.0
        elif mode == "fixed_gamma":
            correction = exact_control(REFERENCE_GAMMA, t)
        else:
            raise ValueError(mode)
        return b @ mean + a + correction

    result = solve_ivp(
        rhs, (0.0, 1.0), np.zeros(2), t_eval=times,
        method="DOP853", rtol=2e-11, atol=2e-12,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y.T


def gaussian_density(points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    delta = points - mean
    exponent = np.einsum("...i,ij,...j->...", delta, np.linalg.inv(cov), delta)
    return np.exp(-0.5 * exponent) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))


def quadrature_tilt(gamma: float, strength: float) -> tuple[np.ndarray, float]:
    """Independent quadrature + dual Newton solve for the terminal tilt.

    Unlike exact_shift(), this evaluates exponential weights directly and solves
    the retention constraint numerically, without using the analytic lambda.
    """
    mean, cov = base_moments(gamma, 1.0)
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    gx, gy = np.meshgrid(nodes, nodes)
    standard = np.sqrt(2.0) * np.stack([gx.ravel(), gy.ravel()], axis=-1)
    points = mean + standard @ np.linalg.cholesky(cov).T
    prior_weights = np.outer(weights, weights).ravel() / np.pi
    lam = 0.0
    for _ in range(40):
        logits = -strength * points[:, 0] - lam * points[:, 1]
        tilted = prior_weights * np.exp(logits - np.max(logits))
        tilted /= tilted.sum()
        tilted_mean = tilted @ points
        residual = tilted_mean[1] - mean[1]
        if abs(residual) < 2e-14:
            return tilted_mean, lam
        var_y = tilted @ (points[:, 1] - tilted_mean[1]) ** 2
        lam += residual / var_y
    raise RuntimeError("Quadrature dual Newton solve failed")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0.0, 1.0, 101)
    gammas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    modes = ["base", "constrained", "unconstrained", "x_only", "fixed_gamma"]
    rng = np.random.default_rng(20261024)
    checks = {
        "integrated_mean_max_abs_error": 0.0,
        "retained_terminal_mean_max_abs_error": 0.0,
        "retained_continuation_mean_max_abs_error": 0.0,
        "continuity_divergence_plus_source_max_abs_error": 0.0,
        "risk_derivative_max_abs_error": 0.0,
        "linear_galerkin_control_max_abs_error": 0.0,
        "quadrature_tilt_mean_max_abs_error": 0.0,
        "dual_lambda_max_abs_error": 0.0,
    }
    rows = []
    curves = {}
    for gamma in gammas:
        original_mean, terminal_cov = base_moments(gamma, 1.0)
        k = residual_direction(gamma)
        residual_variance = float(k @ terminal_cov @ k)
        by_mode = {mode: integrate_mean(gamma, mode, times) for mode in modes}
        curves[gamma] = by_mode
        analytic = np.array([
            base_moments(gamma, float(t))[0] + exact_shift(gamma, float(t))
            for t in times
        ])
        checks["integrated_mean_max_abs_error"] = max(
            checks["integrated_mean_max_abs_error"],
            float(np.max(np.abs(by_mode["constrained"] - analytic))),
        )
        checks["retained_terminal_mean_max_abs_error"] = max(
            checks["retained_terminal_mean_max_abs_error"],
            abs(float(by_mode["constrained"][-1, 1] - original_mean[1])),
        )
        quad_mean, quad_lambda = quadrature_tilt(gamma, BETA_MAX)
        checks["quadrature_tilt_mean_max_abs_error"] = max(
            checks["quadrature_tilt_mean_max_abs_error"],
            float(np.max(np.abs(quad_mean - analytic[-1]))),
        )
        exact_lambda = -BETA_MAX * terminal_cov[1, 0] / terminal_cov[1, 1]
        checks["dual_lambda_max_abs_error"] = max(
            checks["dual_lambda_max_abs_error"], abs(float(quad_lambda - exact_lambda)),
        )
        derivative_eps = 1e-4
        risk_plus = quadrature_tilt(gamma, BETA_MAX + derivative_eps)[0][0]
        risk_minus = quadrature_tilt(gamma, BETA_MAX - derivative_eps)[0][0]
        checks["risk_derivative_max_abs_error"] = max(
            checks["risk_derivative_max_abs_error"],
            abs(float((risk_plus - risk_minus) / (2 * derivative_eps) + residual_variance)),
        )
        # Test the exact continuity source and the weak linear-basis solution.
        for t in [0.0, 0.15, 0.4, 0.75, 1.0]:
            mean, cov = base_moments(gamma, t)
            mean_q = mean + exact_shift(gamma, t)
            future, offset = affine_flow(gamma, 1.0 - t)
            continuation_mean = future @ mean_q + offset
            checks["retained_continuation_mean_max_abs_error"] = max(
                checks["retained_continuation_mean_max_abs_error"],
                abs(float(continuation_mean[1] - original_mean[1])),
            )
            z = rng.multivariate_normal(mean_q, cov, size=64)
            centered = z - mean_q
            h = -beta_dot(t) * ((centered @ future.T) @ k)
            u = exact_control(gamma, t)
            div_qu_over_q = -(centered @ np.linalg.inv(cov)) @ u
            checks["continuity_divergence_plus_source_max_abs_error"] = max(
                checks["continuity_divergence_plus_source_max_abs_error"],
                float(np.max(np.abs(div_qu_over_q + h))),
            )
            # phi_i(z)=z_i: A=I and E_q[z*h] is the following covariance.
            weak_rhs = -beta_dot(t) * cov @ future.T @ k
            checks["linear_galerkin_control_max_abs_error"] = max(
                checks["linear_galerkin_control_max_abs_error"],
                float(np.max(np.abs(weak_rhs - u))),
            )
        rows.append({
            "gamma": gamma,
            "baseline_terminal_mean": original_mean.tolist(),
            "baseline_terminal_covariance": terminal_cov.tolist(),
            "risk_residual_variance": residual_variance,
            "methods": {
                mode: {
                    "terminal_mean": path[-1].tolist(),
                    "risk_mean_change": float(path[-1, 0] - original_mean[0]),
                    "retained_mean_change": float(path[-1, 1] - original_mean[1]),
                } for mode, path in by_mode.items()
            },
        })

    # These tolerances are numerical checks, not statistical confidence claims.
    tolerance = 1e-8
    if any(value >= tolerance for value in checks.values()):
        raise AssertionError(checks)

    metrics = {
        "experiment_type": "analytic_2d_gaussian_sanity_check",
        "semantic_erasure_evaluated": False,
        "image_models_loaded": [],
        "base_model_or_controller_trained": False,
        "risk": "unbounded first terminal coordinate; not a probability",
        "retained_statistic": "second terminal coordinate mean",
        "continuation_observables": "exact original-CFG affine continuation",
        "time_convention": "noise t=0 to data t=1",
        "initial_distribution": "N(0,I), unchanged for all methods",
        "beta_schedule": {"form": "beta_max*t^2", "beta_max": BETA_MAX},
        "fixed_correction_reference_gamma": REFERENCE_GAMMA,
        "numeric_dtype": "float64",
        "rng_seed": 20261024,
        "solver": {"method": "DOP853", "rtol": 2e-11, "atol": 2e-12},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__},
        "checks": checks,
        "check_tolerance": tolerance,
        "rows": rows,
        "limitations": [
            "No DiT, decoder, semantic evaluator, learned probes or controller.",
            "Linear Gaussian tilting changes means but not covariances.",
            "Does not validate high-dimensional particle/Galerkin approximation.",
            "The linear-basis check uses an analytic Gaussian covariance identity.",
            "The dual and derivative checks use independent 24x24 Gaussian quadrature.",
        ],
    }
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    colors = {"base": "#657082", "constrained": "#147d92", "unconstrained": "#cd6840",
              "x_only": "#9e7533", "fixed_gamma": "#9a5bab"}
    labels = {"base": "Original CFG", "constrained": "Constrained transport",
              "unconstrained": "Tilt without retention", "x_only": "Keep x correction only",
              "fixed_gamma": "Reuse correction from CFG=3"}
    mean_ref, cov_ref = base_moments(REFERENCE_GAMMA, 1.0)
    xx, yy = np.meshgrid(
        np.linspace(mean_ref[0] - 3.5, mean_ref[0] + 2.7, 180),
        np.linspace(mean_ref[1] - 3.1, mean_ref[1] + 3.1, 180),
    )
    grid = np.stack([xx, yy], axis=-1)
    ax = axes[0, 0]
    for mode in ["base", "unconstrained", "constrained"]:
        mean = curves[REFERENCE_GAMMA][mode][-1]
        ax.contour(xx, yy, gaussian_density(grid, mean, cov_ref), levels=[0.04, 0.10],
                   colors=[colors[mode]], linewidths=1.4)
        ax.plot(mean[0], mean[1], "o", color=colors[mode], label=labels[mode])
    ax.axhline(mean_ref[1], color="#adb3bd", linewidth=0.9, linestyle="--")
    ax.set(title="(a) Exact terminal distributions, CFG=3",
           xlabel="Terminal risk coordinate x", ylabel="Retained coordinate y")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[0, 1]
    for mode in ["constrained", "unconstrained", "x_only", "fixed_gamma"]:
        ax.plot(gammas, [row["methods"][mode]["retained_mean_change"] for row in rows],
                "o-", color=colors[mode], label=labels[mode], markersize=4)
    ax.axhline(0.0, color="#adb3bd", linewidth=0.9, linestyle="--")
    ax.set(title="(b) Guidance-dependent retention",
           xlabel="External true CFG coefficient", ylabel="Change in terminal E[y]")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for mode in ["unconstrained", "constrained", "x_only", "fixed_gamma"]:
        ax.plot(gammas, [row["methods"][mode]["risk_mean_change"] for row in rows],
                "o-", color=colors[mode], label=labels[mode], markersize=4)
    ax.set(title="(c) Risk reduction and the retention tradeoff",
           xlabel="External true CFG coefficient", ylabel="Change in terminal E[x]")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for mode in ["constrained", "unconstrained", "x_only"]:
        continuation_shift = []
        for t, mu in zip(times, curves[REFERENCE_GAMMA][mode]):
            future, offset = affine_flow(REFERENCE_GAMMA, 1.0 - float(t))
            continuation_shift.append((future @ mu + offset)[1] - mean_ref[1])
        ax.plot(times, continuation_shift, color=colors[mode], label=labels[mode])
    ax.axhline(0.0, color="#adb3bd", linewidth=0.9, linestyle="--")
    ax.set(title="(d) Retain the future semantic, not a local axis",
           xlabel="Flow time t (noise to data)", ylabel="Change in E[f_t(z)], CFG=3")
    ax.legend(fontsize=8)
    fig.suptitle("CFG-PTE: analytic mathematical check only; no image model or semantic erasure test",
                 fontsize=13)
    fig.savefig(OUT / "transport_check.png", dpi=180)
    fig.savefig(OUT / "transport_check.pdf")
    plt.close(fig)
    print(json.dumps({"checks": checks, "output_directory": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
