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
