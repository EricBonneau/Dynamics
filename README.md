# Dynamics
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
