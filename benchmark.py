# ============================================================
# PHASE 5 BENCHMARK SCRIPT
# Large-scale robustness regime for the Binary Framework
#
# Colab notes:
#   - Runtime -> Change runtime type -> GPU
#   - This script uses PyTorch on GPU when available
#   - Start with D_VALUES = [4,5,6,7,8] if you want a faster first run
#   - Then extend to D_VALUES = [4,5,6,7,8,9]
#
# Frozen ingredients:
#   - admissible sector C(Q_d) = ker A
#   - canonical recursive square-based basis
#   - operator families:
#       * restricted line-graph Laplacian
#       * restricted covariance operator
#   - launch families:
#       * single_square
#       * block_average
#   - diagnostics:
#       * top-1 / top-3 / top-5 modal fractions
#       * FFT peak fraction
#       * best single-frequency sinusoid fit R^2
#       * first-mode dominant sector / purity
#       * mean dominant sector / purity over first displayed modes
#       * launched-sector retention
#       * time-averaged dominant sector
#       * peak filtration drift
#       * dominant-sector switches
#       * time-averaged exact and cumulative masses
# ============================================================

import os
import gc
import math
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ------------------------------------------------------------
# 0) CONFIGURATION
# ------------------------------------------------------------
OUTPUT_DIR = "/content/phase5_outputs"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Use [4,5,6,7,8] for a first test. Then add 9.
# D_VALUES = [4, 5, 6, 7, 8, 9]
D_VALUES = [4, 5, 6, 7, 8, 9, 10]

RUN_LAPLACIAN = True
RUN_COVARIANCE = True
ALPHAS = [0.25, 0.5, 1.0, 2.0]

LAUNCH_TYPES = ["single_square", "block_average"]

# Time grid for exact wave evolution
T_MAX = 60.0
N_STEPS = 1500

# Number of low modes displayed in the mode summary
N_MODES_DISPLAY = 12

# Dtype:
# float32 is much faster on GPU.
# If you want maximum numerical stability on CPU, use float64.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64

# Plot controls
MAKE_PLOTS = True
PLOT_COVARIANCE_ALPHAS = [1.0]   # set to ALPHAS if you want all covariance plots

print(f"Using device: {DEVICE}")
print(f"Torch dtype: {TORCH_DTYPE}")
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))


# ------------------------------------------------------------
# 1) BASIC UTILITIES
# ------------------------------------------------------------
BYTE_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def beta1_hypercube(d: int) -> int:
    return (2 ** (d - 1)) * (d - 2) + 1

def theoretical_dim_Ck(d: int, k: int) -> int:
    return (d - k) * (2 ** (d - 1 - k))

def operator_label(operator: str, alpha=None) -> str:
    if operator == "laplacian":
        return "laplacian"
    return f"covariance_alpha_{alpha}"

def save_dataframe(df: pd.DataFrame, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

def to_torch(x_np: np.ndarray):
    return torch.tensor(x_np, dtype=TORCH_DTYPE, device=DEVICE)

def safe_empty_cache():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


# ------------------------------------------------------------
# 2) HYPERCUBE AND CANONICAL BASIS
# ------------------------------------------------------------
def hypercube_edges(d: int):
    """
    Vertices are integers 0..2^d-1.
    Edge = single-bit flip from a vertex with bit j = 0.
    We keep the orientation u < v, i.e. u -> u | (1<<j).
    """
    n = 2 ** d
    edges = []
    dirs = []
    u_lo = []
    u_hi = []

    for u in range(n):
        for j in range(d):
            if ((u >> j) & 1) == 0:
                v = u | (1 << j)
                edges.append((u, v))
                dirs.append(j)
                u_lo.append(u)
                u_hi.append(v)

    edge_index = {e: i for i, e in enumerate(edges)}
    return n, edges, edge_index, np.array(u_lo, dtype=np.uint32), np.array(u_hi, dtype=np.uint32), np.array(dirs, dtype=np.int32)

def incidence_matrix(n: int, edges):
    """
    Dense oriented incidence matrix A of shape (n, m).
    Orientation is from lower vertex to higher vertex.
    """
    m = len(edges)
    A = np.zeros((n, m), dtype=np.float32)
    for j, (u, v) in enumerate(edges):
        A[u, j] = -1.0
        A[v, j] = +1.0
    return A

def cycle_vector_from_path(path, edge_index, m):
    """
    Convert a closed vertex path into an oriented edge vector in R^m.
    """
    x = np.zeros(m, dtype=np.float32)
    for a, b in zip(path[:-1], path[1:]):
        e = (min(a, b), max(a, b))
        idx = edge_index[e]
        x[idx] += 1.0 if a < b else -1.0
    return x

def canonical_basis_paths(d: int):
    """
    Recursive canonical square-based basis paths.
    Each item:
        {"path": [...], "sig": (i,j), "sep": j-i}
    """
    if d == 2:
        return [{
            "path": [0, 1, 3, 2, 0],
            "sig": (0, 1),
            "sep": 1
        }]

    prev = canonical_basis_paths(d - 1)
    mask = 1 << (d - 1)
    out = []

    # Lower-layer lift
    for item in prev:
        out.append({
            "path": item["path"][:],
            "sig": item["sig"],
            "sep": item["sep"]
        })

    # Upper-layer lift
    for item in prev:
        out.append({
            "path": [v | mask for v in item["path"]],
            "sig": item["sig"],
            "sep": item["sep"]
        })

    # Vertical squares from canonical spanning tree of lower layer
    for u in range(1, 2 ** (d - 1)):
        bit = u.bit_length() - 1
        parent = u ^ (1 << bit)
        path = [parent, u, u | mask, parent | mask, parent]
        sig = (bit, d - 1)
        out.append({
            "path": path,
            "sig": sig,
            "sep": sig[1] - sig[0]
        })

    return out

def canonical_basis_vectors(d: int, edge_index, m: int):
    basis_paths = canonical_basis_paths(d)
    basis_items = []
    vecs = []

    for item in basis_paths:
        vec = cycle_vector_from_path(item["path"], edge_index, m)
        basis_items.append({
            "sig": item["sig"],
            "sep": item["sep"],
            "path": item["path"],
        })
        vecs.append(vec)

    B = np.column_stack(vecs).astype(np.float32)  # shape (m, beta1)
    return basis_items, B


# ------------------------------------------------------------
# 3) EDGE-SPACE OPERATORS
# ------------------------------------------------------------
def line_graph_laplacian_from_incidence(A: np.ndarray):
    """
    Dense line-graph Laplacian on edge space.
    Two edges are adjacent in the line graph iff they share a vertex.
    """
    Babs = np.abs(A).astype(np.uint8)
    S = Babs.T @ Babs
    Adj = (S > 0).astype(np.float32)
    np.fill_diagonal(Adj, 0.0)
    deg = Adj.sum(axis=1)
    L = np.diag(deg) - Adj
    return L

def pairwise_hamming_u32(a: np.ndarray, b: np.ndarray):
    """
    Pairwise Hamming distances between uint32 arrays a and b.
    """
    x = np.bitwise_xor(a[:, None], b[None, :]).astype(np.uint32, copy=False)
    xb = x.view(np.uint8).reshape(x.shape + (4,))
    return BYTE_POPCOUNT[xb].sum(axis=-1, dtype=np.uint16)

def line_graph_distance_matrix_from_endpoints(u_lo: np.ndarray, u_hi: np.ndarray, cache_file=None):
    """
    For distinct edges e,f in any simple graph:
        dist_L(e,f) = 1 + min_{a in e, b in f} dist_G(a,b)
    Here dist_G is Hamming distance in the hypercube.
    """
    if cache_file is not None and os.path.exists(cache_file):
        print(f"Loading cached line-graph distances: {cache_file}")
        return np.load(cache_file)

    print("Computing line-graph distance matrix...")
    h00 = pairwise_hamming_u32(u_lo, u_lo)
    h01 = pairwise_hamming_u32(u_lo, u_hi)
    h10 = pairwise_hamming_u32(u_hi, u_lo)
    h11 = pairwise_hamming_u32(u_hi, u_hi)

    D = np.minimum(np.minimum(h00, h01), np.minimum(h10, h11)).astype(np.float32)
    D = D + 1.0
    np.fill_diagonal(D, 0.0)

    if cache_file is not None:
        np.save(cache_file, D)
        print(f"Saved cached distances: {cache_file}")

    return D


# ------------------------------------------------------------
# 4) STRUCTURE BUILD
# ------------------------------------------------------------
def build_structure(d: int, need_covariance_distances: bool = True):
    """
    Build all frozen structural ingredients for a given d.
    """
    print(f"\n=== Building structure for Q_{d} ===")
    n, edges, edge_index, u_lo, u_hi, dirs = hypercube_edges(d)
    m = len(edges)

    basis_items, B_np = canonical_basis_vectors(d, edge_index, m)
    beta1 = beta1_hypercube(d)
    basis_size = B_np.shape[1]
    basis_rank = int(np.linalg.matrix_rank(B_np))

    if basis_size != beta1 or basis_rank != beta1:
        raise ValueError(
            f"Canonical basis failed on Q_{d}: size={basis_size}, rank={basis_rank}, beta1={beta1}"
        )

    block_indices = {}
    for idx, item in enumerate(basis_items):
        k = item["sep"]
        block_indices.setdefault(k, []).append(idx)

    # Dimension table
    dim_rows = []
    for k in range(1, d):
        dim_rows.append({
            "d": d,
            "k": k,
            "actual_dim": len(block_indices.get(k, [])),
            "theory_dim": theoretical_dim_Ck(d, k),
        })
    dim_df = pd.DataFrame(dim_rows)

    # Incidence and Laplacian on edge space
    A_np = incidence_matrix(n, edges)
    L_edge_np = line_graph_laplacian_from_incidence(A_np)

    # Optional line-graph distances for covariance family
    D_line_np = None
    if need_covariance_distances:
        cache_file = os.path.join(CACHE_DIR, f"line_graph_distance_Q{d}.npy")
        D_line_np = line_graph_distance_matrix_from_endpoints(u_lo, u_hi, cache_file=cache_file)

    # Torch tensors
    B_t = to_torch(B_np)
    U_cyc, _ = torch.linalg.qr(B_t, mode="reduced")  # m x beta1 orthonormal basis

    G = B_t.T @ B_t
    left = torch.linalg.solve(G, B_t.T)             # beta1 x m
    C_map = left @ U_cyc                            # beta1_can x beta1_orth

    L_edge_t = to_torch(L_edge_np)
    D_line_t = to_torch(D_line_np) if D_line_np is not None else None

    # Block membership matrix for canonical coefficients
    block_mat = torch.zeros((d - 1, beta1), dtype=TORCH_DTYPE, device=DEVICE)
    for k in range(1, d):
        idx = block_indices.get(k, [])
        if len(idx) > 0:
            block_mat[k - 1, idx] = 1.0

    struct = {
        "d": d,
        "n": n,
        "m": m,
        "beta1": beta1,
        "edges": edges,
        "dirs": dirs,
        "basis_items": basis_items,
        "block_indices": block_indices,
        "block_mat": block_mat,
        "basis_vectors": B_t,      # canonical basis vectors in edge space
        "U_cyc": U_cyc,            # orthonormal cycle basis in edge space
        "C_map": C_map,            # z -> canonical coefficients
        "L_edge": L_edge_t,
        "D_line": D_line_t,
        "dim_df": dim_df,
    }

    print(f"Q_{d}: n={n}, m={m}, beta1={beta1}")
    print(dim_df.to_string(index=False))
    return struct


# ------------------------------------------------------------
# 5) RESTRICTED OPERATORS AND MODES
# ------------------------------------------------------------
def restricted_eigensystem(struct, operator="laplacian", alpha=None):
    """
    Build restricted operator and eigensystem on the admissible sector.
    """
    U_cyc = struct["U_cyc"]

    if operator == "laplacian":
        Op_edge = struct["L_edge"]
    elif operator == "covariance":
        if struct["D_line"] is None:
            raise ValueError("Covariance requested but D_line not available.")
        Op_edge = torch.exp(-float(alpha) * struct["D_line"])
    else:
        raise ValueError(f"Unknown operator: {operator}")

    H = U_cyc.T @ Op_edge @ U_cyc
    H = 0.5 * (H + H.T)

    evals, V = torch.linalg.eigh(H)
    evals = torch.clamp(evals, min=0.0)
    omega = torch.sqrt(evals)

    return {
        "operator": operator,
        "alpha": alpha,
        "H": H,
        "evals": evals,
        "omega": omega,
        "V": V,
    }

def mode_summary_rows(struct, eigsys, n_modes_display=12):
    """
    Low-mode summary:
      - dominant sector
      - sector purity
    """
    C_map = struct["C_map"]
    block_mat = struct["block_mat"]
    d = struct["d"]

    evals = eigsys["evals"]
    omega = eigsys["omega"]
    V = eigsys["V"]

    rows = []
    n_show = min(n_modes_display, V.shape[1])

    for i in range(n_show):
        z_mode = V[:, i]
        coeff = C_map @ z_mode
        coeff2 = coeff.pow(2)
        sector_energy = block_mat @ coeff2
        sector_probs = sector_energy / torch.clamp(sector_energy.sum(), min=1e-12)

        dom_k = int(torch.argmax(sector_probs).item() + 1)
        purity = float(torch.max(sector_probs).item())

        rows.append({
            "operator": eigsys["operator"],
            "alpha": eigsys["alpha"],
            "d": d,
            "mode": i,
            "lambda": float(evals[i].item()),
            "omega": float(omega[i].item()),
            "dominant_k": dom_k,
            "sector_purity": purity,
        })

    return rows


# ------------------------------------------------------------
# 6) LAUNCHES AND DYNAMICS
# ------------------------------------------------------------
def build_launch(struct, k0: int, launch_type: str):
    """
    Canonical launches:
      - single_square
      - block_average
    """
    idx = struct["block_indices"].get(k0, [])
    if len(idx) == 0:
        raise ValueError(f"No basis block for d={struct['d']} and k={k0}")

    B = struct["basis_vectors"]

    if launch_type == "single_square":
        x0 = B[:, idx[0]].clone()
    elif launch_type == "block_average":
        x0 = B[:, idx].sum(dim=1)
    else:
        raise ValueError(f"Unknown launch type: {launch_type}")

    x0 = x0 / torch.clamp(torch.linalg.norm(x0), min=1e-12)
    return x0

def fft_peak_fraction_and_best_r2(obs_np: np.ndarray, t_np: np.ndarray):
    """
    From the observable f(t), compute:
      - FFT peak fraction
      - best single-frequency sinusoid-fit R^2
    """
    y = obs_np.astype(np.float64)
    dt = float(t_np[1] - t_np[0])

    spec = np.fft.rfft(y - y.mean())
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(len(y), d=dt)

    if len(power) <= 1 or power[1:].sum() <= 1e-15:
        return np.nan, np.nan

    peak_idx = 1 + np.argmax(power[1:])
    peak_frac = float(power[peak_idx] / np.sum(power[1:]))

    # Best single-frequency sinusoid fit at the dominant FFT frequency
    w = 2.0 * np.pi * freqs[peak_idx]
    M = np.column_stack([
        np.cos(w * t_np),
        np.sin(w * t_np),
        np.ones_like(t_np)
    ])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    fit = M @ coef

    ss_res = np.sum((y - fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = np.nan if ss_tot <= 1e-15 else float(1.0 - ss_res / ss_tot)

    return peak_frac, r2

def simulate_launch(struct, eigsys, k0: int, launch_type: str, t_grid: torch.Tensor):
    """
    Exact wave evolution:
        z'' + H z = 0
    with x(0)=x0, xdot(0)=0.

    Returns one result row for the benchmark.
    """
    U_cyc = struct["U_cyc"]
    C_map = struct["C_map"]
    block_mat = struct["block_mat"]
    d = struct["d"]

    omega = eigsys["omega"]
    V = eigsys["V"]

    x0 = build_launch(struct, k0=k0, launch_type=launch_type)
    z0 = U_cyc.T @ x0
    a = V.T @ z0

    # Modal fractions
    modal_weights = a.pow(2)
    modal_weights = modal_weights / torch.clamp(modal_weights.sum(), min=1e-12)
    modal_sorted, _ = torch.sort(modal_weights, descending=True)

    top1 = float(modal_sorted[0].item())
    top3 = float(modal_sorted[:3].sum().item())
    top5 = float(modal_sorted[:5].sum().item())

    # Exact time evolution in orthonormal cycle coordinates
    coswt = torch.cos(omega[:, None] * t_grid[None, :])
    Z = V @ (a[:, None] * coswt)  # beta1 x T

    # Observable f(t) = <x(t), x0> = <z(t), z0>
    obs = torch.sum(Z * z0[:, None], dim=0)

    # Canonical-basis coefficients of x(t)
    coeff_t = C_map @ Z
    coeff_energy = coeff_t.pow(2)

    sector_energy = block_mat @ coeff_energy            # (d-1) x T
    sector_mass = sector_energy / torch.clamp(sector_energy.sum(dim=0, keepdim=True), min=1e-12)

    # Cumulative masses
    cum_short = torch.cumsum(sector_mass, dim=0)
    cum_long = torch.flip(torch.cumsum(torch.flip(sector_mass, dims=[0]), dim=0), dims=[0])

    # Dominant sector trajectory
    dom_series = torch.argmax(sector_mass, dim=0) + 1
    switches = int(torch.sum(dom_series[1:] != dom_series[:-1]).item())

    # Averages
    mean_masses = torch.mean(sector_mass, dim=1)
    mean_short = torch.mean(cum_short, dim=1)
    mean_long = torch.mean(cum_long, dim=1)

    launched_retention = float(mean_masses[k0 - 1].item())
    initial_dom = int(dom_series[0].item())
    time_avg_dom = int(torch.argmax(mean_masses).item() + 1)

    # Filtration drift
    drift_series = torch.sum(torch.abs(sector_mass - sector_mass[:, :1]), dim=0)
    peak_drift = float(torch.max(drift_series).item())

    # FFT + best sinusoid fit
    obs_np = obs.detach().cpu().numpy()
    t_np = t_grid.detach().cpu().numpy()
    fft_peak, best_r2 = fft_peak_fraction_and_best_r2(obs_np, t_np)

    # First-mode and mean low-mode diagnostics
    mode_rows = mode_summary_rows(struct, eigsys, n_modes_display=N_MODES_DISPLAY)
    first_mode_dom = int(mode_rows[0]["dominant_k"])
    first_mode_purity = float(mode_rows[0]["sector_purity"])
    mean_mode_dom = float(np.mean([row["dominant_k"] for row in mode_rows]))
    mean_mode_purity = float(np.mean([row["sector_purity"] for row in mode_rows]))

    row = {
        "operator": eigsys["operator"],
        "alpha": eigsys["alpha"],
        "operator_label": operator_label(eigsys["operator"], eigsys["alpha"]),
        "d": d,
        "n": struct["n"],
        "m": struct["m"],
        "beta1": struct["beta1"],
        "launch_type": launch_type,
        "k0": k0,
        "top1_modal_fraction": top1,
        "top3_modal_fraction": top3,
        "top5_modal_fraction": top5,
        "fft_peak_fraction": fft_peak,
        "best_sinusoid_r2": best_r2,
        "first_mode_dominant_sector": first_mode_dom,
        "first_mode_sector_purity": first_mode_purity,
        "mean_low_mode_dominant_sector": mean_mode_dom,
        "mean_low_mode_sector_purity": mean_mode_purity,
        "launched_sector_retention": launched_retention,
        "initial_dominant_sector": initial_dom,
        "time_averaged_dominant_sector": time_avg_dom,
        "peak_filtration_drift_L1": peak_drift,
        "dominant_sector_switches": switches,
    }

    # Store time-averaged exact and cumulative masses up to max possible d in this run
    for kk in range(1, d):
        row[f"mean_M_{kk}"] = float(mean_masses[kk - 1].item())
        row[f"mean_Mle_{kk}"] = float(mean_short[kk - 1].item())
        row[f"mean_Mge_{kk}"] = float(mean_long[kk - 1].item())

    return row


# ------------------------------------------------------------
# 7) MAIN BENCHMARK LOOP
# ------------------------------------------------------------
def run_phase5_benchmark():
    t0_all = time.time()

    t_grid = torch.linspace(0.0, T_MAX, N_STEPS, dtype=TORCH_DTYPE, device=DEVICE)

    master_rows = []
    mode_rows_all = []
    dim_rows_all = []

    for d in D_VALUES:
        struct = build_structure(d, need_covariance_distances=RUN_COVARIANCE)
        dim_rows_all.append(struct["dim_df"])

        operator_jobs = []
        if RUN_LAPLACIAN:
            operator_jobs.append(("laplacian", None))
        if RUN_COVARIANCE:
            for alpha in ALPHAS:
                operator_jobs.append(("covariance", alpha))

        for op_name, alpha in operator_jobs:
            print(f"\n--- Operator build: d={d}, operator={op_name}, alpha={alpha} ---")
            t_op = time.time()
            eigsys = restricted_eigensystem(struct, operator=op_name, alpha=alpha)
            print(f"Operator built in {time.time() - t_op:.2f} s")

            # Save low-mode summary rows
            mode_rows = mode_summary_rows(struct, eigsys, n_modes_display=N_MODES_DISPLAY)
            mode_rows_all.extend(mode_rows)

            # Launch grid
            for k0 in range(1, d):
                for launch_type in LAUNCH_TYPES:
                    print(f"Run: d={d}, operator={op_name}, alpha={alpha}, k={k0}, launch={launch_type}")
                    row = simulate_launch(struct, eigsys, k0=k0, launch_type=launch_type, t_grid=t_grid)
                    master_rows.append(row)

            # Free operator-specific tensors
            del eigsys
            safe_empty_cache()

        # Free structure tensors before next d
        del struct
        safe_empty_cache()

    master_df = pd.DataFrame(master_rows)
    mode_df = pd.DataFrame(mode_rows_all)
    dim_df = pd.concat(dim_rows_all, ignore_index=True)

    # Aggregated views
    agg_by_d_k_launch = (
        master_df
        .groupby(["operator_label", "d", "launch_type", "k0"], dropna=False)
        .agg({
            "top1_modal_fraction": "mean",
            "top3_modal_fraction": "mean",
            "top5_modal_fraction": "mean",
            "fft_peak_fraction": "mean",
            "best_sinusoid_r2": "mean",
            "launched_sector_retention": "mean",
            "time_averaged_dominant_sector": "mean",
            "peak_filtration_drift_L1": "mean",
            "dominant_sector_switches": "mean",
        })
        .reset_index()
    )

    agg_by_d_launch = (
        master_df
        .groupby(["operator_label", "d", "launch_type"], dropna=False)
        .agg({
            "top1_modal_fraction": "mean",
            "top3_modal_fraction": "mean",
            "top5_modal_fraction": "mean",
            "fft_peak_fraction": "mean",
            "best_sinusoid_r2": "mean",
            "launched_sector_retention": "mean",
            "time_averaged_dominant_sector": "mean",
            "peak_filtration_drift_L1": "mean",
            "dominant_sector_switches": "mean",
        })
        .reset_index()
    )

    # Save outputs
    save_dataframe(dim_df, "phase5_basis_dimensions.csv")
    save_dataframe(mode_df, "phase5_mode_summary.csv")
    save_dataframe(master_df, "phase5_master_results.csv")
    save_dataframe(agg_by_d_k_launch, "phase5_agg_by_d_k_launch.csv")
    save_dataframe(agg_by_d_launch, "phase5_agg_by_d_launch.csv")

    # Save config
    config = {
        "device": DEVICE,
        "torch_dtype": str(TORCH_DTYPE),
        "D_VALUES": D_VALUES,
        "RUN_LAPLACIAN": RUN_LAPLACIAN,
        "RUN_COVARIANCE": RUN_COVARIANCE,
        "ALPHAS": ALPHAS,
        "LAUNCH_TYPES": LAUNCH_TYPES,
        "T_MAX": T_MAX,
        "N_STEPS": N_STEPS,
        "N_MODES_DISPLAY": N_MODES_DISPLAY,
    }
    with open(os.path.join(OUTPUT_DIR, "phase5_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTotal runtime: {time.time() - t0_all:.2f} s")
    return master_df, mode_df, agg_by_d_k_launch, agg_by_d_launch


# ------------------------------------------------------------
# 8) PLOTTING
# ------------------------------------------------------------
def plot_metric_vs_k(df, operator_label_value, metric, title, filename):
    sub = df[df["operator_label"] == operator_label_value].copy()
    if sub.empty:
        return

    launches = sorted(sub["launch_type"].unique())

    fig, axes = plt.subplots(1, len(launches), figsize=(7 * len(launches), 5), sharey=True)
    if len(launches) == 1:
        axes = [axes]

    for ax, launch in zip(axes, launches):
        ss = sub[sub["launch_type"] == launch]
        for d in sorted(ss["d"].unique()):
            ssd = ss[ss["d"] == d].sort_values("k0")
            ax.plot(ssd["k0"], ssd[metric], marker="o", label=f"d={d}")
        ax.set_title(f"{title} | {launch}")
        ax.set_xlabel("k")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")

def make_phase5_plots(master_df):
    if not MAKE_PLOTS:
        return

    # Laplacian plots
    if RUN_LAPLACIAN:
        op_label = "laplacian"
        plot_metric_vs_k(
            master_df, op_label,
            metric="top3_modal_fraction",
            title="Top-3 modal fraction vs k",
            filename="phase5_laplacian_top3_vs_k.png"
        )
        plot_metric_vs_k(
            master_df, op_label,
            metric="fft_peak_fraction",
            title="FFT peak fraction vs k",
            filename="phase5_laplacian_fft_vs_k.png"
        )
        plot_metric_vs_k(
            master_df, op_label,
            metric="time_averaged_dominant_sector",
            title="Time-averaged dominant sector vs k",
            filename="phase5_laplacian_domsector_vs_k.png"
        )
        plot_metric_vs_k(
            master_df, op_label,
            metric="peak_filtration_drift_L1",
            title="Peak filtration drift vs k",
            filename="phase5_laplacian_drift_vs_k.png"
        )

    # Selected covariance plots
    if RUN_COVARIANCE:
        for alpha in PLOT_COVARIANCE_ALPHAS:
            op_label = operator_label("covariance", alpha)
            plot_metric_vs_k(
                master_df, op_label,
                metric="top3_modal_fraction",
                title=f"Top-3 modal fraction vs k (alpha={alpha})",
                filename=f"phase5_cov_alpha_{alpha}_top3_vs_k.png"
            )
            plot_metric_vs_k(
                master_df, op_label,
                metric="fft_peak_fraction",
                title=f"FFT peak fraction vs k (alpha={alpha})",
                filename=f"phase5_cov_alpha_{alpha}_fft_vs_k.png"
            )
            plot_metric_vs_k(
                master_df, op_label,
                metric="time_averaged_dominant_sector",
                title=f"Time-averaged dominant sector vs k (alpha={alpha})",
                filename=f"phase5_cov_alpha_{alpha}_domsector_vs_k.png"
            )
            plot_metric_vs_k(
                master_df, op_label,
                metric="peak_filtration_drift_L1",
                title=f"Peak filtration drift vs k (alpha={alpha})",
                filename=f"phase5_cov_alpha_{alpha}_drift_vs_k.png"
            )


# ------------------------------------------------------------
# 9) RUN
# ------------------------------------------------------------
master_df, mode_df, agg_by_d_k_launch, agg_by_d_launch = run_phase5_benchmark()
make_phase5_plots(master_df)

print("\nPhase 5 benchmark complete.")
print("\nMaster results preview:")
print(master_df.head().to_string(index=False))

print("\nAggregated by d and launch preview:")
print(agg_by_d_launch.head().to_string(index=False))