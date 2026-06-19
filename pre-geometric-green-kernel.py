# ============================================================
# Pre-geometric Green-kernel test:
# Does the projected load kernel behave like 1/r in a 3D continuum window?
#
# Colab:
#   Runtime -> Change runtime type -> GPU
#
# What this tests:
#   1. Build a projected admissible load quotient with 3 large-scale axes.
#   2. Define L_grav as the graph Laplacian.
#   3. Compute K_D = L_grav^+ by solving L_grav phi = delta_source - uniform.
#   4. Radial-average phi(r).
#   5. Fit phi(r) ≈ A/r + B.
#   6. Estimate spectral dimension from heat-kernel return probability.
#
# Important:
#   This is a numerical scaling test, not a formal theorem.
#   A theorem would still need to prove that the chosen projected admissible
#   load sector has effective spectral dimension 3 in the large-system limit.
# ============================================================

import os
import time
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------

M = 48
# The quotient has (M+1)^3 load leaves.
# M=32 is quick. M=48 is good. M=64 is heavier but still reasonable on a GPU.

MAX_CG_ITERS = 2500
CG_TOL = 1e-6

HEAT_STEPS = 512
LAZY = 0.50

FIT_R_MIN = 4
FIT_R_MAX_FRACTION = 0.33
# Fit range will be r in [FIT_R_MIN, FIT_R_MAX_FRACTION * M].
# Avoid r=0 singularity and avoid boundary-dominated large r.

OUTPUT_DIR = "/content/pregeo_green_kernel_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32 if device.type == "cuda" else torch.float64

print("Device:", device)
print("Torch dtype:", dtype)
print("M:", M)

# ----------------------------
# Build projected 3D load quotient
# ----------------------------

def idx(i, j, k, n):
    return (i * n + j) * n + k

def build_flat_3d_quotient(m):
    """
    Builds a 3D cubic quotient graph with nodes (i,j,k), 0<=i,j,k<=m.
    This is the projected load sector whose shell law is |B(r)| ~ r^3.

    In manuscript language:
      This is not the raw hypercube. It is the projected admissible
      load quotient whose continuum limit is being tested.
    """
    n = m + 1
    N = n ** 3

    ids = np.arange(N, dtype=np.int64).reshape(n, n, n)

    edge_u = []
    edge_v = []

    # x-neighbours
    edge_u.append(ids[:-1, :, :].ravel())
    edge_v.append(ids[1:, :, :].ravel())

    # y-neighbours
    edge_u.append(ids[:, :-1, :].ravel())
    edge_v.append(ids[:, 1:, :].ravel())

    # z-neighbours
    edge_u.append(ids[:, :, :-1].ravel())
    edge_v.append(ids[:, :, 1:].ravel())

    edge_u = np.concatenate(edge_u)
    edge_v = np.concatenate(edge_v)
    edge_w = np.ones_like(edge_u, dtype=np.float64)

    return N, edge_u, edge_v, edge_w

N, edge_u, edge_v, edge_w = build_flat_3d_quotient(M)
E = len(edge_u)

print(f"Nodes: {N:,}")
print(f"Undirected edges: {E:,}")

# ----------------------------
# Sparse Laplacian
# ----------------------------

def build_laplacian_torch(N, u, v, w, device, dtype):
    """
    Weighted graph Laplacian:
      L_ii = degree(i)
      L_ij = -w_ij
    """
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

L, deg_t, deg_np = build_laplacian_torch(N, edge_u, edge_v, edge_w, device, dtype)

def spmv(A, x):
    return torch.sparse.mm(A, x.reshape(-1, 1)).reshape(-1)

# ----------------------------
# Solve Green equation:
#     L phi = delta_source - uniform
# with zero-mean gauge
# ----------------------------

source_coord = np.array([M // 2, M // 2, M // 2], dtype=np.int64)
source = idx(source_coord[0], source_coord[1], source_coord[2], M + 1)

print("Source coordinate:", tuple(source_coord))
print("Source index:", source)

b_np = np.full(N, -1.0 / N, dtype=np.float64)
b_np[source] += 1.0
b = torch.tensor(b_np, dtype=dtype, device=device)

diag_inv = torch.where(deg_t > 0, 1.0 / deg_t, torch.zeros_like(deg_t))

def project_zero_mean(x):
    return x - torch.mean(x)

def pcg_laplacian(L, b, diag_inv, max_iter=2000, tol=1e-6):
    """
    Preconditioned conjugate gradient on singular Laplacian,
    restricted to zero-mean subspace.
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

        if it % 100 == 0 or it == 1:
            print(f"CG iter {it:5d} | relative residual {rel_res:.3e}")

        if rel_res < tol:
            print(f"CG converged at iter {it} | relative residual {rel_res:.3e}")
            break

        z = diag_inv * r
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old
        p = project_zero_mean(z + beta * p)
        rz_old = rz_new

    elapsed = time.time() - start
    print(f"CG elapsed: {elapsed:.2f} s")

    return x, np.array(history, dtype=np.float64)

phi_t, cg_history = pcg_laplacian(
    L, b, diag_inv,
    max_iter=MAX_CG_ITERS,
    tol=CG_TOL
)

phi = phi_t.detach().cpu().numpy().astype(np.float64)

# Save CG convergence
pd.DataFrame({
    "iteration": np.arange(1, len(cg_history) + 1),
    "relative_residual": cg_history
}).to_csv(os.path.join(OUTPUT_DIR, "cg_convergence.csv"), index=False)

# ----------------------------
# Radial averaging
# ----------------------------

coords = np.stack(
    np.unravel_index(np.arange(N), (M + 1, M + 1, M + 1)),
    axis=1
).astype(np.float64)

r = np.linalg.norm(coords - source_coord.reshape(1, 3), axis=1)
r_bin = np.rint(r).astype(np.int64)

radial_rows = []
for rb in range(0, int(r_bin.max()) + 1):
    mask = (r_bin == rb)
    if not np.any(mask):
        continue
    radial_rows.append({
        "r_bin": rb,
        "r_mean": float(np.mean(r[mask])),
        "K_mean": float(np.mean(phi[mask])),
        "K_std": float(np.std(phi[mask])),
        "count": int(np.sum(mask))
    })

radial = pd.DataFrame(radial_rows)
radial_path = os.path.join(OUTPUT_DIR, "green_kernel_radial_average.csv")
radial.to_csv(radial_path, index=False)

# ----------------------------
# Fit K(r) = A/r + B
# ----------------------------

fit_r_max = FIT_R_MAX_FRACTION * M

fit_df = radial[
    (radial["r_mean"] >= FIT_R_MIN) &
    (radial["r_mean"] <= fit_r_max) &
    (radial["count"] >= 20)
].copy()

x = (1.0 / fit_df["r_mean"].values).reshape(-1, 1)
y = fit_df["K_mean"].values

X = np.column_stack([x[:, 0], np.ones_like(x[:, 0])])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
A_fit, B_fit = coef

y_hat = X @ coef
ss_res = float(np.sum((y - y_hat) ** 2))
ss_tot = float(np.sum((y - np.mean(y)) ** 2))
R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

fit_summary = pd.DataFrame([{
    "M": M,
    "N_nodes": N,
    "source_i": int(source_coord[0]),
    "source_j": int(source_coord[1]),
    "source_k": int(source_coord[2]),
    "fit_r_min": FIT_R_MIN,
    "fit_r_max": fit_r_max,
    "A_fit": A_fit,
    "B_fit": B_fit,
    "R2_for_A_over_r_plus_B": R2,
    "final_cg_relative_residual": float(cg_history[-1]) if len(cg_history) else np.nan
}])

fit_summary_path = os.path.join(OUTPUT_DIR, "green_kernel_fit_summary.csv")
fit_summary.to_csv(fit_summary_path, index=False)

print("\n--- Green kernel fit ---")
print(f"K(r) ≈ A/r + B")
print(f"A = {A_fit:.8e}")
print(f"B = {B_fit:.8e}")
print(f"R^2 = {R2:.8f}")
print(f"Fit window: r in [{FIT_R_MIN}, {fit_r_max:.2f}]")

# ----------------------------
# Plot Green radial profile
# ----------------------------

plt.figure(figsize=(8, 5))
plt.scatter(radial["r_mean"], radial["K_mean"], s=18, label="Radial average")
rr = np.linspace(FIT_R_MIN, fit_r_max, 300)
plt.plot(rr, A_fit / rr + B_fit, linewidth=2, label="Fit: A/r + B")
plt.xlabel("Radial distance r in projected load quotient")
plt.ylabel("Green response K_D(r)")
plt.title("Projected Green kernel radial average")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "green_kernel_radial_fit.png")
plt.savefig(plot_path, dpi=180)
plt.show()

plt.figure(figsize=(8, 5))
plateau = radial[radial["r_mean"] > 0].copy()
plt.plot(plateau["r_mean"], plateau["r_mean"] * (plateau["K_mean"] - B_fit), marker="o", linewidth=1)
plt.axhline(A_fit, linestyle="--", label="A from A/r+B fit")
plt.xlabel("Radial distance r")
plt.ylabel("r · (K_D(r) - B)")
plt.title("1/r plateau diagnostic")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plateau_path = os.path.join(OUTPUT_DIR, "green_kernel_plateau_diagnostic.png")
plt.savefig(plateau_path, dpi=180)
plt.show()

# ----------------------------
# Heat-kernel spectral dimension test
# ----------------------------

def build_transition_transpose_torch(N, u, v, w, deg, device, dtype):
    """
    Builds P^T for the random walk:
      p_{t+1} = P^T p_t
    with P_{i->j} = w_ij / deg_i.
    """
    rows = np.concatenate([v, u])
    cols = np.concatenate([u, v])
    vals = np.concatenate([w / deg[u], w / deg[v]]).astype(np.float64)

    indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
    values = torch.tensor(vals, dtype=dtype, device=device)

    PT = torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()
    return PT

PT = build_transition_transpose_torch(
    N, edge_u, edge_v, edge_w, deg_np, device, dtype
)

p = torch.zeros(N, dtype=dtype, device=device)
p[source] = 1.0

returns = []

start = time.time()

for t in range(1, HEAT_STEPS + 1):
    p_walk = spmv(PT, p)
    p = LAZY * p + (1.0 - LAZY) * p_walk
    returns.append(float(p[source].detach().cpu()))

elapsed = time.time() - start
print(f"\nHeat diffusion elapsed: {elapsed:.2f} s")

t_arr = np.arange(1, HEAT_STEPS + 1, dtype=np.float64)
ret = np.array(returns, dtype=np.float64)

# Estimate spectral dimension:
#   p(t) ~ t^{-d_s/2}
#   d_s(t) = -2 d log p / d log t
valid = ret > 0
log_t = np.log(t_arr[valid])
log_p = np.log(ret[valid])

ds = -2.0 * np.gradient(log_p, log_t)
t_valid = t_arr[valid]
ret_valid = ret[valid]

heat_df = pd.DataFrame({
    "t": t_valid,
    "return_probability": ret_valid,
    "spectral_dimension_estimate": ds
})

heat_path = os.path.join(OUTPUT_DIR, "heat_kernel_spectral_dimension.csv")
heat_df.to_csv(heat_path, index=False)

# Choose an intermediate window: avoid very early lattice effects and late finite-size effects.
ds_window_min = max(12, int(0.03 * HEAT_STEPS))
ds_window_max = int(0.35 * HEAT_STEPS)

window = heat_df[
    (heat_df["t"] >= ds_window_min) &
    (heat_df["t"] <= ds_window_max)
]

ds_median = float(window["spectral_dimension_estimate"].median())
ds_mean = float(window["spectral_dimension_estimate"].mean())

print("\n--- Heat-kernel spectral dimension ---")
print(f"Window: t in [{ds_window_min}, {ds_window_max}]")
print(f"Median d_s ≈ {ds_median:.4f}")
print(f"Mean d_s   ≈ {ds_mean:.4f}")

plt.figure(figsize=(8, 5))
plt.loglog(heat_df["t"], heat_df["return_probability"], linewidth=2)
plt.xlabel("Diffusion time t")
plt.ylabel("Return probability p(t; source, source)")
plt.title("Heat-kernel return probability")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
return_plot_path = os.path.join(OUTPUT_DIR, "heat_return_probability.png")
plt.savefig(return_plot_path, dpi=180)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(heat_df["t"], heat_df["spectral_dimension_estimate"], linewidth=2)
plt.axhline(3.0, linestyle="--", label="Target d_s = 3")
plt.axvspan(ds_window_min, ds_window_max, alpha=0.15, label="Measurement window")
plt.xlabel("Diffusion time t")
plt.ylabel("Effective spectral dimension d_s(t)")
plt.title("Effective spectral dimension")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
ds_plot_path = os.path.join(OUTPUT_DIR, "spectral_dimension_estimate.png")
plt.savefig(ds_plot_path, dpi=180)
plt.show()

# ----------------------------
# Pass/fail diagnostic
# ----------------------------

green_pass = R2 >= 0.995
ds_pass = (2.7 <= ds_median <= 3.3)

print("\n============================================================")
print("SUMMARY")
print("============================================================")
print(f"Green 1/r fit R^2: {R2:.8f}  -> {'PASS' if green_pass else 'CHECK'}")
print(f"Median spectral dimension: {ds_median:.4f} -> {'PASS' if ds_pass else 'CHECK'}")
print()
print("Output directory:")
print(OUTPUT_DIR)
print()
print("Files written:")
for fn in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", fn)

# ----------------------------
# Manuscript interpretation helper
# ----------------------------

print("\nManuscript-safe interpretation:")
print("""
This numerical experiment computes the Green response K_D=L_grav^+ on a projected
three-dimensional admissible load quotient. The radial Green response is tested
against K_D(r)=A/r+B, while the heat-kernel return probability is used to estimate
the effective spectral dimension. A simultaneous 1/r Green window and d_s≈3 heat
window supports, but does not prove, the claim that the projected load sector has
a Newtonian weak-field continuum limit. A formal proof must still establish that
the admissible quotient selected by the Hodge-Dirac/least-action construction has
three-dimensional volume growth and heat-kernel bounds in the large-system limit.
""")