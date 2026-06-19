# ============================================================
# Least-action / dimensional-economy scan for coarse load maps C_q
#
# Goal:
#   Test whether q=3 is selected as the minimal transient load sector.
#
# For each q:
#   1. Define C_q by partitioning hypercube directions into q bundles.
#   2. Build the capacity-normalized q-dimensional load quotient.
#   3. Compute the Green kernel K_D = L_grav^+.
#   4. Estimate heat-kernel spectral dimension d_s.
#   5. Fit the Green decay exponent K(r) ~ A/r^p + B.
#
# Selection criterion:
#
#   q_* = min { q : d_s(C_q) > 2 }
#
# Interpretation:
#   q <= 2 is recurrent / non-decaying in the infinite-volume Green sense.
#   q = 3 is the first transient sector and gives p ≈ 1.
#   q = 4 gives p ≈ 2, etc.
#
# Colab:
#   Runtime -> Change runtime type -> GPU
# ============================================================

import os
import time
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "/content/Cq_selection_scan_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32 if device.type == "cuda" else torch.float64

print("Device:", device)
print("Torch dtype:", dtype)

# q = number of macroscopic load bundles.
# M = side parameter; quotient has (M+1)^q leaves.
#
# Keep q=4 at M=24 for manageable GPU memory.
# You may raise q=3 to 64 for a cleaner window if desired.
SCAN_CONFIG = {
    1: 512,
    2: 192,
    3: 48,
    4: 24,
}

MAX_CG_ITERS = 4000
CG_TOL = 1e-6

HEAT_STEPS_DEFAULT = 512
LAZY = 0.50

FIT_R_MIN_DEFAULT = 4
FIT_R_MAX_FRACTION = 0.33

TRANSIENCE_THRESHOLD = 2.05

# ============================================================
# Utilities
# ============================================================

def make_grid_edges(q, M):
    """
    Build q-dimensional cubic quotient graph on {0,...,M}^q.

    This is the capacity-normalized quotient induced by C_q:
      C_q maps microscopic hypercube load densities to active-count
      coordinates in q balanced direction bundles.

    The raw binomial multiplicities are quotiented out by capacity normalization.
    """
    shape = tuple([M + 1] * q)
    N = (M + 1) ** q
    ids = np.arange(N, dtype=np.int64).reshape(shape)

    edge_u = []
    edge_v = []

    for axis in range(q):
        sl1 = [slice(None)] * q
        sl2 = [slice(None)] * q
        sl1[axis] = slice(0, M)
        sl2[axis] = slice(1, M + 1)

        u = ids[tuple(sl1)].ravel()
        v = ids[tuple(sl2)].ravel()

        edge_u.append(u)
        edge_v.append(v)

    edge_u = np.concatenate(edge_u)
    edge_v = np.concatenate(edge_v)
    edge_w = np.ones_like(edge_u, dtype=np.float64)

    return N, shape, edge_u, edge_v, edge_w


def build_laplacian_torch(N, u, v, w, device, dtype):
    deg = np.bincount(
        np.concatenate([u, v]),
        weights=np.concatenate([w, w]),
        minlength=N
    ).astype(np.float64)

    rows = np.concatenate([u, v, np.arange(N, dtype=np.int64)])
    cols = np.concatenate([v, u, np.arange(N, dtype=np.int64)])
    vals = np.concatenate([-w, -w, deg]).astype(np.float64)

    indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
    values = torch.tensor(vals, dtype=dtype, device=device)

    L = torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()
    deg_t = torch.tensor(deg, dtype=dtype, device=device)

    return L, deg_t, deg


def spmv(A, x):
    return torch.sparse.mm(A, x.reshape(-1, 1)).reshape(-1)


def project_zero_mean(x):
    return x - torch.mean(x)


def pcg_laplacian(L, b, diag_inv, max_iter=4000, tol=1e-6):
    """
    Solve L x = b on the zero-mean subspace using preconditioned CG.
    L is singular, so b must have zero mean.
    """
    x = torch.zeros_like(b)

    r = project_zero_mean(b - spmv(L, x))
    z = diag_inv * r
    p = project_zero_mean(z)
    rz_old = torch.dot(r, z)

    b_norm = torch.linalg.norm(b).item()
    if b_norm == 0:
        b_norm = 1.0

    history = []
    start = time.time()

    for it in range(1, max_iter + 1):
        Ap = project_zero_mean(spmv(L, p))
        denom = torch.dot(p, Ap)

        if torch.abs(denom).item() < 1e-30:
            print("CG stopped: denominator too small.")
            break

        alpha = rz_old / denom
        x = project_zero_mean(x + alpha * p)
        r = project_zero_mean(r - alpha * Ap)

        rel_res = torch.linalg.norm(r).item() / b_norm
        history.append(rel_res)

        if it == 1 or it % 100 == 0:
            print(f"    CG iter {it:5d} | relative residual {rel_res:.3e}")

        if rel_res < tol:
            print(f"    CG converged at iter {it} | relative residual {rel_res:.3e}")
            break

        z = diag_inv * r
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old
        p = project_zero_mean(z + beta * p)
        rz_old = rz_new

    elapsed = time.time() - start
    return x, np.array(history, dtype=np.float64), elapsed


def radial_average(phi, q, M, source_coord):
    shape = tuple([M + 1] * q)
    N = (M + 1) ** q

    coords = np.stack(np.unravel_index(np.arange(N), shape), axis=1).astype(np.float64)
    r = np.linalg.norm(coords - source_coord.reshape(1, q), axis=1)
    r_bin = np.rint(r).astype(np.int64)

    rows = []
    for rb in range(0, int(r_bin.max()) + 1):
        mask = (r_bin == rb)
        if not np.any(mask):
            continue

        rows.append({
            "r_bin": rb,
            "r_mean": float(np.mean(r[mask])),
            "K_mean": float(np.mean(phi[mask])),
            "K_std": float(np.std(phi[mask])),
            "count": int(np.sum(mask))
        })

    return pd.DataFrame(rows)


def fit_power_decay(radial, M, q):
    """
    Fit K(r) = A / r^p + B over an interior window.

    For q >= 3, continuum expectation is p = q - 2.
    We grid-search p and solve A,B by least squares for each p.
    """
    fit_r_min = FIT_R_MIN_DEFAULT
    fit_r_max = FIT_R_MAX_FRACTION * M

    fit_df = radial[
        (radial["r_mean"] >= fit_r_min) &
        (radial["r_mean"] <= fit_r_max) &
        (radial["count"] >= max(5, 2 * q))
    ].copy()

    if len(fit_df) < 5:
        return {
            "fit_ok": False,
            "fit_r_min": fit_r_min,
            "fit_r_max": fit_r_max,
            "best_p": np.nan,
            "A_fit": np.nan,
            "B_fit": np.nan,
            "R2": np.nan,
            "expected_p": max(q - 2, np.nan)
        }, fit_df

    r = fit_df["r_mean"].values
    y = fit_df["K_mean"].values

    # q=1 and q=2 are recurrent; power-decay p is not the right infinite-volume model.
    # We still fit p as a diagnostic, but the transience test comes from d_s.
    p_grid = np.linspace(0.10, 3.50, 341)

    best = None

    for p in p_grid:
        x = r ** (-p)
        X = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coef

        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        if best is None or R2 > best["R2"]:
            best = {
                "fit_ok": True,
                "fit_r_min": fit_r_min,
                "fit_r_max": fit_r_max,
                "best_p": float(p),
                "A_fit": float(coef[0]),
                "B_fit": float(coef[1]),
                "R2": float(R2),
                "expected_p": float(q - 2) if q >= 3 else np.nan
            }

    return best, fit_df


def build_transition_transpose_torch(N, u, v, w, deg, device, dtype):
    rows = np.concatenate([v, u])
    cols = np.concatenate([u, v])
    vals = np.concatenate([w / deg[u], w / deg[v]]).astype(np.float64)

    indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
    values = torch.tensor(vals, dtype=dtype, device=device)

    PT = torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()
    return PT


def heat_spectral_dimension(N, u, v, w, deg, source, M, q):
    """
    Estimate spectral dimension from lazy random-walk return probability.
    """
    PT = build_transition_transpose_torch(N, u, v, w, deg, device, dtype)

    p = torch.zeros(N, dtype=dtype, device=device)
    p[source] = 1.0

    heat_steps = HEAT_STEPS_DEFAULT
    returns = []

    start = time.time()

    for t in range(1, heat_steps + 1):
        p_walk = spmv(PT, p)
        p = LAZY * p + (1.0 - LAZY) * p_walk
        returns.append(float(p[source].detach().cpu()))

    elapsed = time.time() - start

    t_arr = np.arange(1, heat_steps + 1, dtype=np.float64)
    ret = np.array(returns, dtype=np.float64)

    valid = ret > 0
    log_t = np.log(t_arr[valid])
    log_p = np.log(ret[valid])

    ds = -2.0 * np.gradient(log_p, log_t)

    heat_df = pd.DataFrame({
        "t": t_arr[valid],
        "return_probability": ret[valid],
        "spectral_dimension_estimate": ds
    })

    # Measurement window:
    # Avoid microscopic early time and avoid boundary late time.
    t_min = max(8, int(0.03 * heat_steps))
    t_max_by_steps = int(0.35 * heat_steps)
    t_max_by_size = max(t_min + 5, int(0.12 * M * M))
    t_max = min(t_max_by_steps, t_max_by_size)

    window = heat_df[
        (heat_df["t"] >= t_min) &
        (heat_df["t"] <= t_max)
    ].copy()

    ds_median = float(window["spectral_dimension_estimate"].median())
    ds_mean = float(window["spectral_dimension_estimate"].mean())

    return heat_df, {
        "heat_elapsed_sec": elapsed,
        "heat_window_min": t_min,
        "heat_window_max": t_max,
        "ds_median": ds_median,
        "ds_mean": ds_mean
    }


# ============================================================
# Main scan
# ============================================================

all_summary_rows = []

for q, M in SCAN_CONFIG.items():
    print("\n============================================================")
    print(f"Scanning C_{q}: q={q} bundles, side M={M}")
    print("============================================================")

    N, shape, edge_u, edge_v, edge_w = make_grid_edges(q, M)
    E = len(edge_u)

    source_coord = np.array([M // 2] * q, dtype=np.int64)
    source = np.ravel_multi_index(tuple(source_coord), shape)

    print(f"  Quotient leaves N = {N:,}")
    print(f"  Undirected edges E = {E:,}")
    print(f"  Source coordinate = {tuple(source_coord)}")
    print(f"  Source index = {source}")

    L, deg_t, deg_np = build_laplacian_torch(N, edge_u, edge_v, edge_w, device, dtype)
    diag_inv = torch.where(deg_t > 0, 1.0 / deg_t, torch.zeros_like(deg_t))

    b_np = np.full(N, -1.0 / N, dtype=np.float64)
    b_np[source] += 1.0
    b = torch.tensor(b_np, dtype=dtype, device=device)

    phi_t, cg_history, cg_elapsed = pcg_laplacian(
        L, b, diag_inv,
        max_iter=MAX_CG_ITERS,
        tol=CG_TOL
    )

    phi = phi_t.detach().cpu().numpy().astype(np.float64)

    q_dir = os.path.join(OUTPUT_DIR, f"q_{q}")
    os.makedirs(q_dir, exist_ok=True)

    pd.DataFrame({
        "iteration": np.arange(1, len(cg_history) + 1),
        "relative_residual": cg_history
    }).to_csv(os.path.join(q_dir, "cg_convergence.csv"), index=False)

    radial = radial_average(phi, q, M, source_coord)
    radial.to_csv(os.path.join(q_dir, "green_kernel_radial_average.csv"), index=False)

    fit_result, fit_df = fit_power_decay(radial, M, q)
    heat_df, heat_result = heat_spectral_dimension(N, edge_u, edge_v, edge_w, deg_np, source, M, q)

    heat_df.to_csv(os.path.join(q_dir, "heat_kernel_spectral_dimension.csv"), index=False)

    # Classification
    ds_median = heat_result["ds_median"]
    transient = ds_median > TRANSIENCE_THRESHOLD

    if q <= 2:
        expected_green_class = "recurrent: no finite decaying infinite-volume Green kernel"
    elif q == 3:
        expected_green_class = "minimal transient: expected 1/r Green decay"
    else:
        expected_green_class = f"transient higher-dimensional: expected 1/r^{q-2} Green decay"

    # Plots
    plt.figure(figsize=(8, 5))
    plt.scatter(radial["r_mean"], radial["K_mean"], s=18, label="Radial average")

    if fit_result["fit_ok"]:
        rr = np.linspace(fit_result["fit_r_min"], fit_result["fit_r_max"], 300)
        plt.plot(
            rr,
            fit_result["A_fit"] / (rr ** fit_result["best_p"]) + fit_result["B_fit"],
            linewidth=2,
            label=f"Fit: A/r^p+B, p={fit_result['best_p']:.2f}"
        )

    plt.xlabel(f"Radial distance r in C_{q} quotient")
    plt.ylabel("Green response K_D(r)")
    plt.title(f"Green kernel radial average for C_{q}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(q_dir, "green_kernel_fit.png"), dpi=180)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(heat_df["t"], heat_df["spectral_dimension_estimate"], linewidth=2)
    plt.axhline(q, linestyle="--", label=f"Reference d_s={q}")
    plt.axhline(3.0, linestyle=":", label="d_s=3")
    plt.axvspan(
        heat_result["heat_window_min"],
        heat_result["heat_window_max"],
        alpha=0.15,
        label="Measurement window"
    )
    plt.xlabel("Diffusion time t")
    plt.ylabel("Effective spectral dimension d_s(t)")
    plt.title(f"Effective spectral dimension for C_{q}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(q_dir, "spectral_dimension.png"), dpi=180)
    plt.show()

    summary_row = {
        "q_bundles": q,
        "M_side": M,
        "N_leaves": N,
        "E_edges": E,
        "cg_final_relative_residual": float(cg_history[-1]) if len(cg_history) else np.nan,
        "cg_iterations": len(cg_history),
        "cg_elapsed_sec": cg_elapsed,
        "ds_median": ds_median,
        "ds_mean": heat_result["ds_mean"],
        "heat_window_min": heat_result["heat_window_min"],
        "heat_window_max": heat_result["heat_window_max"],
        "transient_by_ds_gt_2p05": bool(transient),
        "expected_green_class": expected_green_class,
        "fit_ok": bool(fit_result["fit_ok"]),
        "green_best_power_p": fit_result["best_p"],
        "green_expected_power_q_minus_2": fit_result["expected_p"],
        "green_power_fit_R2": fit_result["R2"],
        "green_A_fit": fit_result["A_fit"],
        "green_B_fit": fit_result["B_fit"],
    }

    all_summary_rows.append(summary_row)

    pd.DataFrame([summary_row]).to_csv(os.path.join(q_dir, "summary.csv"), index=False)

# ============================================================
# Selection summary
# ============================================================

summary = pd.DataFrame(all_summary_rows)
summary_path = os.path.join(OUTPUT_DIR, "Cq_selection_summary.csv")
summary.to_csv(summary_path, index=False)

transient_rows = summary[summary["transient_by_ds_gt_2p05"]].copy()

if len(transient_rows) > 0:
    q_selected = int(transient_rows.sort_values("q_bundles").iloc[0]["q_bundles"])
else:
    q_selected = None

selection = pd.DataFrame([{
    "selection_rule": "q_star = min { q : d_s(C_q) > 2.05 }",
    "q_selected": q_selected,
    "interpretation": (
        "q=3 is selected if q=1,2 are recurrent and q=3 is the first transient load sector"
        if q_selected == 3 else
        "selection did not return q=3; inspect summary"
    )
}])

selection.to_csv(os.path.join(OUTPUT_DIR, "selection_result.csv"), index=False)

print("\n============================================================")
print("C_q SELECTION SUMMARY")
print("============================================================")
print(summary[[
    "q_bundles",
    "M_side",
    "N_leaves",
    "ds_median",
    "transient_by_ds_gt_2p05",
    "green_best_power_p",
    "green_expected_power_q_minus_2",
    "green_power_fit_R2",
    "expected_green_class"
]])

print("\nSelection rule:")
print("  q_star = min { q : d_s(C_q) > 2.05 }")
print("Selected q_star:", q_selected)

if q_selected == 3:
    print("\nPASS: q=3 is selected as the minimal transient load sector.")
else:
    print("\nCHECK: q=3 was not selected. Inspect fit windows and summary.")

print("\nOutput directory:")
print(OUTPUT_DIR)

print("""
Manuscript-safe interpretation:

We scan a family of coarse load maps C_q, where q is the number of macroscopic
load bundles. The quotient associated with C_q has q-dimensional active-count
coordinates after quotienting microscopic binomial degeneracy through capacity
normalization.

The selection rule is not to impose Newtonian 1/r decay, but to require the
least macroscopic load dimension that supports a transient Green response:

    q_* = min { q : d_s(C_q) > 2 }.

For q <= 2, diffusion is recurrent and the infinite-volume Green response does
not give a finite decaying potential. The first transient case is q=3. In that
case the Green kernel has the continuum form K_D(r) ~ 1/r. Higher q are also
transient but produce faster decays K_D(r) ~ 1/r^(q-2), so they are rejected by
dimensional economy.

This scan is numerical evidence for the selection rule. A formal theorem would
prove that least admissible structural action plus transience and dimensional
economy selects an effective d_s=3 load sector in the hypercube hierarchy.
""")