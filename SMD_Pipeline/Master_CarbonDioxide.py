import sys
import os
import matplotlib.pyplot as plt
import neb_pipeline_utils as npu
sys.path.append(os.getcwd())
# ========================= #
#  SIMULATION CONFIGURATION #
# ========================= #
sim_config = {
    "Fluid": "HEOS::CO2",  # Backend::Fluid
    "P_in": 61e5,  # [Pa] Inlet Total Pressure
    "T_in": 293.15,  # [K] Inlet Total Temp
    "P_out": 16e5,  # [Pa] Outlet Static Pressure
    "K": None,  # Relaxation rate
    "N": None,  # Delay index
    # --- GEOMETRY (post-processing extent; X-aligned by convention) ---
    "x_min": 0.0,
    "x_max": 0.0835,   # CO2 reference nozzle length [m]
    "n_slices": 100,
    # --- CLUSTER SETTINGS ---
    "CFX_BIN": "/software/ansys/v242/CFX/bin",
    # --- Parallel Configuration ---
    "NODES": "node-4-16*40", # node-4-16*40 -> Run on node-4-16 with 40 cores
    "MEM_FLAGS": "-S 1.9 -sizepar 1.5" # $SLURM_NODELIST*$SLURM_NTASKS -> For SLURM jobs
}
# ==========================================
# RUN PIPELINE
# ==========================================
# This one line generates:
# 1. The Physics Data
# 2. The Excel File (with metadata)
# 3. The CCL File (for the solver)
# 4. The CSE File (for the post-processor)
# ==========================================
# final_df = npu.run_neb_pipeline(sim_config)
# ==========================================

# Notebook Cell
k_list = [1,2,3,4,5]
n_list = [1,2]

for k in k_list:
    for n in n_list:
        cfg = sim_config.copy()
        cfg['K'] = k
        cfg['N'] = n
        # 1. Generate everything
        run_script = npu.run_neb_pipeline_in_folder(cfg)

        # 2. Run (or Submit)
        # os.system(run_script) # Run sequentially
        # OR
        # os.system(f"sbatch {run_script}") # Submit to scheduler
