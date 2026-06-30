import sys
import os
import CoolProp.CoolProp as CP
from scipy.optimize import brentq
import neb_pipeline_utils as npu
sys.path.append(os.getcwd())

# ============================================================ #
#  WATER FLASHING — MASTER SCRIPT (legacy pipeline)            #
# ============================================================ #
# Water has a negative liquid spinodal for all these conditions, so neb_utils
# auto-detects it and anchors the closure coordinate at zero (Pi = P/P_nuc,
# inception-anchored; no equilibrium tail– equilibrium is reached by the relaxation law itself).
#
# The closure exponent n is LAW-DRIVEN: per operating point it is evaluated from
# the DEM->NEB mapping  n(Pi_nuc) = 1.428 + 1.216 * Pi_nuc^6.37,  with k = 100. | Deviation is expected and has to be treated with care.
# The law is evaluated HERE in the master (it needs the operating point), and
# the resulting n is handed to the pipeline via config['N'] — the pipeline and
# neb_utils stay fluid-agnostic.
#
# n_correction lets you perturb the law prior for CFD calibration.

FLUID = "HEOS::Water"

# --- Base (shared) configuration -------------------------------------------
base_config = {
    "Fluid": FLUID,
    # --- GEOMETRY (X-aligned; replace x_max with the SMD nozzle length) ---
    "x_min": 0.0,
    "x_max": 0.4631,
    "n_slices": 100,
    # --- CLUSTER SETTINGS ---
    "CFX_BIN": "/software/ansys/v242/CFX/bin",
    "NODES": "node-4-16*40",
    "MEM_FLAGS": "-S 1.9 -sizepar 1.5",
}

# --- Calibration grid -------------------------------------------------------
# n is law-driven (one value per operating point); k and the multiplicative
# n_correction are the calibration knobs. Defaults below give a single bare-law
# run per point (k=100, correction=1.0 -> 24 cases total). Widen either list to
# sweep:  e.g. K_LIST = [30, 100, 300]  and/or  N_CORRECTION_LIST = [0.8,1.0,1.2].
# The full water calibration grid is  {points} x {K_LIST} x {N_CORRECTION_LIST}.
K_LIST            = [100]      # prefactor: My thought on this is that it will require tuning to compensate for Richardson's wall shear stress in 1D.
N_CORRECTION_LIST = [1.0]      # multiplies the law-predicted n (1.0 = bare law) | The higher n is, the more choked mass will be allowed through the nozzle: I mean if the law predicted pair (n, K) gave a choked mass lower than the experimental
# mass flow rate, this means we first increase N and then tune K!

# --- The 24 experimental stagnation conditions -----------------------------
# (P_in [bar], T_in [K], label)
RAW_POINTS = [
    # (20,  484.7907913, "P20_s00"),
    # (20,  482.3876905, "P20_s03"),
    # (20,  477.2714115, "P20_s08"),
    # (20,  476.4186983, "P20_s09"),
    # (20,  465.2946673, "P20_s20"),
    # (20,  464.4031944, "P20_s21"),
    # (40,  522.4962934, "P40_s01"),
    # (40,  521.8761383, "P40_s02"),
    # (40,  513.6978437, "P40_s10"),
    # (40,  512.6125724, "P40_s11"),
    # (40,  500.6358282, "P40_s23"),
    # (40,  500.4420298, "P40_s23b"),
    # (80,  567.7313703, "P80_s00"),
    # (80,  566.8398974, "P80_s01"),
    # (80,  559.8243936, "P80_s08"),
    # (80,  558.6228432, "P80_s10"),
    # (80,  549.7081145, "P80_s18"),
    # (80,  548.5453238, "P80_s20"),
    # (120, 597.786418,  "P120_s00"),
    # (120, 596.7786661, "P120_s01"),
    # (120, 592.8639374, "P120_s05"),
    # (120, 592.282542,  "P120_s06"),
    (120, 578.8716893, "P120_s19"),
    # (120, 578.7166506, "P120_s19b"),
]


# def outlet_pressure(P_in_bar):
#     """Outlet (back) pressure per case. PLACEHOLDER: replace with measured
#     exit pressures. Here a simple fraction of stagnation."""
#     return max(0.2 * P_in_bar, 1.0) * 1e5

def outlet_pressure(P_in_bar):
    """Outlet (back) pressure per case. PLACEHOLDER: replace with measured
    exit pressures. Here a simple fraction of stagnation."""
    return 68.52e5



def compute_pi_nuc(P_in, T_in, fluid):
    """
    Inception ratio Pi_nuc = P_nuc / P_in for the inlet isentrope. P_nuc is the
    pressure where the saturated-liquid entropy equals the inlet entropy --- the
    same nucleation pressure neb_utils computes internally, recomputed here so
    the master can evaluate the law.
    """
    s_in = CP.PropsSI("S", "P", P_in, "T", T_in, fluid)
    P_trip = CP.PropsSI("ptriple", fluid)
    P_crit = CP.PropsSI("Pcrit", fluid)
    P_nuc = brentq(lambda P: CP.PropsSI("S", "P", P, "Q", 0, fluid) - s_in,
                   P_trip * 1.01, P_crit * 0.999)
    return P_nuc / P_in


def n_law_water(pi_nuc):
    """DEM -> NEB exponent law for water flashing (k = 100 companion)."""
    return 1.428 + 1.216 * pi_nuc ** 6.37


# ==========================================================================
# RUN
# ==========================================================================
if __name__ == "__main__":
    n_cases = len(RAW_POINTS) * len(K_LIST) * len(N_CORRECTION_LIST)
    print(f"Water calibration grid: {len(RAW_POINTS)} points x "
          f"{len(K_LIST)} K x {len(N_CORRECTION_LIST)} corrections = {n_cases} cases\n")

    for P_in_bar, T_in, label in RAW_POINTS:
        P_in = P_in_bar * 1e5

        # Law-driven n for this operating point (independent of k / correction).
        pi_nuc = compute_pi_nuc(P_in, T_in, FLUID)
        n_law = n_law_water(pi_nuc)
        print(f"\n=== {label}: P_in={P_in_bar} bar, T_in={T_in:.2f} K, "
              f"Pi_nuc={pi_nuc:.4f}, n_law={n_law:.4f} ===")

        for k_val in K_LIST:
            for corr in N_CORRECTION_LIST:
                n_val = n_law * corr

                cfg = base_config.copy()
                cfg["P_in"]  = P_in
                cfg["T_in"]  = T_in
                cfg["P_out"] = outlet_pressure(P_in_bar)
                cfg["K"]     = k_val
                cfg["N"]     = round(n_val, 4)
                cfg["label"] = label

                print(f"   -> k={k_val}, correction={corr}, n={cfg['N']}")

                # Generate physics, files, and the per-case folder + run_case.sh
                run_script = npu.run_neb_pipeline_in_folder(cfg)

                # Submit (uncomment as needed):
                # os.system(f"sbatch {run_script}")
                # os.system(f"bash {run_script}")