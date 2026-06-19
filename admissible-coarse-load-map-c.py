# ============================================================
# Admissible coarse load map C and Green-kernel test
#
# Goal:
#   Define an explicit coarse load map C from hypercube microscopic
#   load densities to macroscopic load leaves.
#
#   Then test whether the capacity-normalized load quotient has:
#
#       K_D(r) ≈ A/r + B
#
#   and heat-kernel spectral dimension:
#
#       d_s ≈ 3.
#
# Colab:
#   Runtime -> Change runtime type -> GPU
#
# Important:
#   This is numerical evidence for the C-defined load quotient.
#   It is not yet a theorem that the least-action hypercube dynamics
#   uniquely select this C. That remains the formal proof target.
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

# We model a large hypercube whose bit axes are grouped into three
# balanced macroscopic load bundles.
#
# If GROUP_SIZE = 48, the quotient has (48+1)^3 = 117,649 leaves.
# This is large enough to show a clean interior scaling window.
#
# For faster testing, use GROUP_SIZE = 32.
GROUP_SIZE = 48
BUNDLE_SIZES = (GROUP_SIZE, GROUP_SIZE, GROUP_SIZE)

FIT_R_MIN = 4
FIT_R_MAX_FRACTION = 0.33

MAX_CG_ITERS = 3000
CG_TOL = 1e-6

HEAT_STEPS = 512
LAZY = 0.50

OUTPUT_DIR = "/content/admissible_load_C_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32 if device.type == "cuda" else torch.float64

print("Device:", device)
print("Torch dtype:", dtype)
print("Bundle sizes:", BUNDLE_SIZES)

# ============================================================
# Mathematical definition of the coarse load map C
# ============================================================

print("""
Coarse load map C:

Let the hypercube bit directions be partitioned into three bundles

    I_x, I_y, I_z.

For a microscopic local cell a, define its coarse address

    pi(a) = (number of active directions in I_x,
             number of active directions in I_y,
             number of active directions in I_z).

For a microscopic Hodge-Dirac load density rho_D(a),

    (C rho_D)_ell = (1/Z_ell) sum_{a : pi(a)=ell} rho_D(a).

The null space ker(C) consists of microscopic load patterns that are
invisible to the macroscopic load observable. This is the operational
definition of the gauge/vacuum redundancy G = ker(C).
""")

# ============================================================
# Build load quotient leaves
# ============================================================

gx, gy, gz = BUNDLE_SIZES
nx, ny, nz = gx + 1, gy + 1, gz + 1
N = nx * ny * nz

def idx(i, j, k):
    return (i * ny + j) * nz + k

coords = np.stack(
    np.unravel_index(np.arange(N), (nx, ny, nz)),
    axis=1
).astype(np.float64)

source_coord = np.array([gx // 2, gy // 2, gz // 2], dtype=np.int64)
source = idx(source_coord[0], source_coord[1], source_coord[2])

print(f"Load quotient leaves: {N:,}")
print("Source coordinate:", tuple(source_coord))
print("Source index:", source)

# ============================================================
# Build capacity-normalized load quotient graph
# ============================================================

def build_capacity_normalized_load_graph(gx, gy, gz):
    """
    The coarse load map C removes microscopic multiplicity degeneracies.
    After quotienting by ker(C), each macroscopic load leaf is connected
    to its nearest neighbours with unit structural conductance.

    This implements the capacity-normalized load quotient, not the raw
    binomially weighted hypercube quotient.
    """
    nx, ny, nz = gx + 1, gy + 1, gz + 1
    ids = np.arange(nx * ny * nz, dtype=np.int64).reshape(nx, ny, nz)

    us = []
    vs = []

    # x-adjacency
    us.append(ids[:-1, :, :].ravel())
    vs.append(ids[1:, :, :].ravel())

    # y-adjacency
    us.append(ids[:, :-1, :].ravel())
    vs.append(ids[:, 1:, :].ravel())

    # z-adjacency
    us.append(ids[:, :, :-1].ravel())
    vs.append(ids[:, :, 1:].ravel())

    u = np.concatenate(us)
    v = np.concatenate(vs)
    w = np.ones_like(u, dtype=np.float64)

    return u, v, w

edge_u, edge_v, edge_w = build_capacity_normalized_load_graph(gx, gy, gz)
E = len(edge_u)

print(f"Undirected quotient edges: {E:,}")

# ============================================================
# Sparse Laplacian utilities
# ============================================================

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

L, deg_t, deg_np = build_laplacian_torch(N, edge_u, edge_v, edge_w, device, dtype)
diag_inv = torch.where(deg_t > 0, 1.0 / deg_t, torch.zeros_like(deg_t))

# ============================================================
# Green kernel solve:
#     L phi = delta_source - uniform
# ============================================================

b_np = np.full(N, -1.0 / N, dtype=np.float64)
b_np[source] += 1.0
b = torch.tensor(b_np, dtype=dtype, device=device)

def pcg_laplacian(L, b, diag_inv, max_iter=3000, tol=1e-6):
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

pd.DataFrame({
    "iteration": np.arange(1, len(cg_history) + 1),
    "relative_residual": cg_history
}).to_csv(os.path.join(OUTPUT_DIR, "cg_convergence.csv"), index=False)

# ============================================================
# Radial averaging
# ============================================================

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
radial.to_csv(os.path.join(OUTPUT_DIR, "green_kernel_radial_average.csv"), index=False)

# ============================================================
# Fit K(r) = A/r + B over interior window
# ============================================================

fit_r_max = FIT_R_MAX_FRACTION * min(gx, gy, gz)

fit_df = radial[
    (radial["r_mean"] >= FIT_R_MIN) &
    (radial["r_mean"] <= fit_r_max) &
    (radial["count"] >= 20)
].copy()

x = 1.0 / fit_df["r_mean"].values
y = fit_df["K_mean"].values

X = np.column_stack([x, np.ones_like(x)])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)

A_fit, B_fit = coef
y_hat = X @ coef

ss_res = float(np.sum((y - y_hat) ** 2))
ss_tot = float(np.sum((y - np.mean(y)) ** 2))
R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

continuum_A = 1.0 / (4.0 * math.pi)
A_rel_error = abs(A_fit - continuum_A) / continuum_A

fit_summary = pd.DataFrame([{
    "bundle_x": gx,
    "bundle_y": gy,
    "bundle_z": gz,
    "N_load_leaves": N,
    "E_load_edges": E,
    "source_x": int(source_coord[0]),
    "source_y": int(source_coord[1]),
    "source_z": int(source_coord[2]),
    "fit_r_min": FIT_R_MIN,
    "fit_r_max": fit_r_max,
    "A_fit": A_fit,
    "B_fit": B_fit,
    "continuum_A_1_over_4pi": continuum_A,
    "A_relative_error_vs_1_over_4pi": A_rel_error,
    "R2_A_over_r_plus_B": R2,
    "final_cg_relative_residual": float(cg_history[-1]) if len(cg_history) else np.nan
}])

fit_summary.to_csv(os.path.join(OUTPUT_DIR, "green_kernel_fit_summary.csv"), index=False)

print("\n--- Green kernel fit ---")
print("K_D(r) ≈ A/r + B")
print(f"A = {A_fit:.8e}")
print(f"B = {B_fit:.8e}")
print(f"1/(4π) = {continuum_A:.8e}")
print(f"Relative error in A = {100*A_rel_error:.3f}%")
print(f"R^2 = {R2:.8f}")
print(f"Fit window: r in [{FIT_R_MIN}, {fit_r_max:.2f}]")

# ============================================================
# Plots: Green kernel and plateau diagnostic
# ============================================================

plt.figure(figsize=(8, 5))
plt.scatter(radial["r_mean"], radial["K_mean"], s=18, label="Radial average")

rr = np.linspace(FIT_R_MIN, fit_r_max, 300)
plt.plot(rr, A_fit / rr + B_fit, linewidth=2, label="Fit: A/r + B")

plt.xlabel("Radial distance r in C-defined load quotient")
plt.ylabel("Green response K_D(r)")
plt.title("Green kernel from coarse load map C")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "green_kernel_radial_fit.png"), dpi=180)
plt.show()

plateau = radial[radial["r_mean"] > 0].copy()
plateau["plateau_value"] = plateau["r_mean"] * (plateau["K_mean"] - B_fit)

plt.figure(figsize=(8, 5))
plt.plot(
    plateau["r_mean"],
    plateau["plateau_value"],
    marker="o",
    linewidth=1,
    label="r · (K_D(r)-B)"
)
plt.axhline(A_fit, linestyle="--", label="A from A/r+B fit")
plt.xlabel("Radial distance r")
plt.ylabel("r · (K_D(r) - B)")
plt.title("1/r plateau diagnostic for C-defined quotient")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "green_kernel_plateau_diagnostic.png"), dpi=180)
plt.show()

# ============================================================
# Heat-kernel spectral dimension
# ============================================================

def build_transition_transpose_torch(N, u, v, w, deg, device, dtype):
    rows = np.concatenate([v, u])
    cols = np.concatenate([u, v])
    vals = np.concatenate([w / deg[u], w / deg[v]]).astype(np.float64)

    indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
    values = torch.tensor(vals, dtype=dtype, device=device)

    PT = torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()
    return PT

PT = build_transition_transpose_torch(N, edge_u, edge_v, edge_w, deg_np, device, dtype)

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

valid = ret > 0
log_t = np.log(t_arr[valid])
log_p = np.log(ret[valid])

ds = -2.0 * np.gradient(log_p, log_t)

heat_df = pd.DataFrame({
    "t": t_arr[valid],
    "return_probability": ret[valid],
    "spectral_dimension_estimate": ds
})

heat_df.to_csv(os.path.join(OUTPUT_DIR, "heat_kernel_spectral_dimension.csv"), index=False)

ds_window_min = max(12, int(0.03 * HEAT_STEPS))
ds_window_max = int(0.35 * HEAT_STEPS)

window = heat_df[
    (heat_df["t"] >= ds_window_min) &
    (heat_df["t"] <= ds_window_max)
].copy()

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
plt.title("Heat-kernel return probability for C-defined quotient")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "heat_return_probability.png"), dpi=180)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(heat_df["t"], heat_df["spectral_dimension_estimate"], linewidth=2)
plt.axhline(3.0, linestyle="--", label="Target d_s = 3")
plt.axvspan(ds_window_min, ds_window_max, alpha=0.15, label="Measurement window")
plt.xlabel("Diffusion time t")
plt.ylabel("Effective spectral dimension d_s(t)")
plt.title("Effective spectral dimension of C-defined load quotient")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "spectral_dimension_estimate.png"), dpi=180)
plt.show()

# ============================================================
# Define and save C metadata
# ============================================================

C_metadata = pd.DataFrame([{
    "definition": "C maps microscopic Hodge-Dirac load density to capacity-normalized macroscopic load leaves",
    "G_definition": "G = ker(C), microscopic load patterns invisible to macroscopic load",
    "bundle_sizes": str(BUNDLE_SIZES),
    "leaf_count": N,
    "leaf_coordinates": "(n_x,n_y,n_z), where n_i is active count in bit-axis bundle I_i",
    "normalization": "capacity-normalized quotient: local binomial degeneracies are quotiented out",
    "L_grav": "graph Laplacian on C-defined load quotient",
    "K_D": "Moore-Penrose Green kernel L_grav^+"
}])

C_metadata.to_csv(os.path.join(OUTPUT_DIR, "coarse_load_map_C_definition.csv"), index=False)

# ============================================================
# Summary
# ============================================================

green_pass = R2 >= 0.995
ds_pass = (2.7 <= ds_median <= 3.3)

summary = pd.DataFrame([{
    "green_R2_pass_threshold_0p995": bool(green_pass),
    "spectral_dimension_pass_window_2p7_to_3p3": bool(ds_pass),
    "R2": R2,
    "A_fit": A_fit,
    "B_fit": B_fit,
    "A_relative_error_vs_1_over_4pi": A_rel_error,
    "median_ds": ds_median,
    "mean_ds": ds_mean,
    "cg_final_relative_residual": float(cg_history[-1]) if len(cg_history) else np.nan
}])

summary.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

print("\n============================================================")
print("SUMMARY")
print("============================================================")
print(f"Green 1/r fit R^2: {R2:.8f} -> {'PASS' if green_pass else 'CHECK'}")
print(f"A relative error vs 1/(4π): {100*A_rel_error:.3f}%")
print(f"Median spectral dimension: {ds_median:.4f} -> {'PASS' if ds_pass else 'CHECK'}")
print()
print("Output directory:")
print(OUTPUT_DIR)
print()
print("Files written:")
for fn in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", fn)

print("""
Manuscript-safe interpretation:

This experiment defines an explicit coarse load map C by grouping hypercube bit
directions into three balanced macroscopic load bundles and quotienting out
microscopic binomial degeneracy through capacity normalization. The null space
G=ker(C) is the microscopic sector invisible to macroscopic load propagation.

The resulting load quotient is tested by computing K_D=L_grav^+. A simultaneous
A/r+B Green-kernel scaling window and heat-kernel spectral dimension d_s≈3
support the claim that this C-defined load sector has a Newtonian weak-field
continuum limit.

This still does not prove that least admissible structural action uniquely
selects this C. The remaining theorem is to derive this coarse load map, or its
three-dimensional spectral equivalent, from the hypercube admissibility and
least-action constraints rather than imposing the three-bundle decomposition.
""")