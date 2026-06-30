import os
import pandas as pd
import shutil
from neb_utils import generate_neb_table, stretched_exponential_closure, plot_extended_verification
import matplotlib.pyplot as plt
# ==========================================
# 1. CONFIGURATION | Change to CFX 25
# ==========================================
CFX_BIN_PATH = "/software/ansys191/ansys_inc/v191/CFX/bin"
NODE_LIST = "node-4-16*40"
MEMORY_FLAGS = "-S 1.9 -sizepar 1.5"

def _format_pairs(x_data, y_data):
    """Helper for formatting CCL data pairs."""
    pairs = [f"{x:.6E}, {y:.6E}" for x, y in zip(x_data, y_data)]
    full_str = ", ".join(pairs)
    chunk_size = 75
    lines = []
    current = 0
    while current < len(full_str):
        end = min(current + chunk_size, len(full_str))
        if end < len(full_str):
            break_pt = full_str.rfind(',', current, end) + 1
            if break_pt <= current: break_pt = end
        else:
            break_pt = end
        lines.append(f" {full_str[current:break_pt]}" + ("\\\n" if break_pt < len(full_str) else "\n"))
        current = break_pt
    return "".join(lines)


# ==========================================
# 2. FILE WRITERS
# ==========================================
def write_ccl_file(filename, df, p_in, p_out):
    """
    Generates CCL with:
    1. Core Physics: rhoP, muP
    2. Analysis Vars: SoS, Gamma, AND Phase Fractions (Vol & Mass)
    3. Proper CEL Syntax & Variable Type
    """

    # Helper to chunk long data lines
    def _format_pairs(x_data, y_data):
        pairs = [f"{x:.6E}, {y:.6E}" for x, y in zip(x_data, y_data)]
        full_str = ", ".join(pairs)
        chunk_size = 75
        lines = []
        current = 0
        while current < len(full_str):
            end = min(current + chunk_size, len(full_str))
            if end < len(full_str):
                break_pt = full_str.rfind(',', current, end) + 1
                if break_pt <= current: break_pt = end
            else:
                break_pt = end
            segment = full_str[current:break_pt]
            suffix = "\\\n" if break_pt < len(full_str) else "\n"
            lines.append(f"          {segment}{suffix}")
            current = break_pt
        return "".join(lines)

    # ---------------------------------------------------------
    # DEFINE VARIABLES TO EXPORT
    # ---------------------------------------------------------
    # Core Physics
    vars_to_export = {
        "rhoP": ("Rho_Mixture", "kg m^-3"),
        "muP": ("Mu_Mixture", "Pa s"),
    }

    # Analysis Variables (Standard)
    av_to_export = {
        "GammaRelaxation": ("Gamma", "[]"),
        "SoSMixture": ("SoS_Mixture", "m s^-1"),
    }

    # Phase Fractions (Volume)
    vol_frac_map = {
        "VolFracMetaLiquid": ("eps_l_meta", "[]"),
        "VolFracSatLiquid": ("eps_l_sat", "[]"),
        "VolFracVapor": ("eps_v", "[]")
    }

    # Mass Fractions (Optional)
    mass_frac_map = {
        "MassFracMetaLiquid": ("X_l_meta", "[]"),
        "MassFracSatLiquid": ("X_l_sat", "[]"),
        "MassFracVapor": ("X_v", "[]")
    }

    # Add to AV list only if column exists in DataFrame
    for cfx_name, (col, unit) in vol_frac_map.items():
        if col in df.columns:
            av_to_export[cfx_name] = (col, unit)

    for cfx_name, (col, unit) in mass_frac_map.items():
        if col in df.columns:
            av_to_export[cfx_name] = (col, unit)

    with open(filename, 'w') as f:
        f.write(f"# Auto-generated Material Update: {filename}\n")
        f.write("LIBRARY:\n")

        # 1. DEFINE ADDITIONAL VARIABLES
        for av_name, (col, unit) in av_to_export.items():
            f.write(f"  ADDITIONAL VARIABLE: {av_name}\n")
            f.write("    Option = Definition\n")
            f.write(f"    Units = {unit}\n")
            f.write("    Tensor Type = SCALAR\n")
            f.write("    Variable Type = Unspecified\n")
            f.write("  END\n")

        # 2. DEFINE FUNCTIONS
        f.write("  CEL:\n")
        f.write("    EXPRESSIONS:\n")
        f.write(f"      PTin = {p_in / 1e5:.4f} [bar]\n")
        f.write(f"      Pout = {p_out / 1e5:.4f} [bar]\n")
        f.write("    END\n")

        # Combine Core and AVs for function writing
        all_funcs = {**vars_to_export, **av_to_export}

        for cfx_name, (col_name, unit) in all_funcs.items():
            # Core vars keep their name (rhoP), AVs get 'func' prefix
            func_name = cfx_name if cfx_name in vars_to_export else f"func{cfx_name}"

            f.write(f"    FUNCTION: {func_name}\n")
            f.write("      Argument Units = Pa\n")
            f.write("      Option = Interpolation\n")
            f.write(f"      Result Units = {unit}\n")
            f.write("      INTERPOLATION DATA:\n")
            f.write("        Extend Max = On\n")
            f.write("        Extend Min = On\n")
            f.write("        Option = One Dimensional\n")
            f.write("        Data Pairs = \\\n")

            clean_series = df[col_name].interpolate().fillna(0.0)
            f.write(_format_pairs(df['Pressure'], clean_series))
            f.write("      END\n")
            f.write("    END\n")

        f.write("  END\n")  # End CEL
        f.write("END\n")  # End LIBRARY

        # 3. LINK VARIABLES TO DOMAIN
        f.write("\nFLOW: Flow Analysis 1\n")
        f.write("  DOMAIN: Default Domain\n")
        f.write("    FLUID MODELS:\n")

        for av_name, (col, unit) in av_to_export.items():
            func_name = f"func{av_name}"
            f.write(f"      ADDITIONAL VARIABLE: {av_name}\n")
            f.write(f"        Additional Variable Value = {func_name}(Absolute Pressure)\n")
            f.write("        Option = Algebraic Equation\n")
            f.write("      END\n")

        f.write("    END\n")
        f.write("  END\n")
        f.write("END\n")

    print(f"[Pipeline] CCL Generated: {filename} (Includes {len(av_to_export)} AVs)")


def write_post_process_script(filename, k_val, n_val, x_min=0.0, x_max=0.0835, n_slices=100):
    """
    Generates the proven CSE script using ISOSURFACE and detailed table formatting.
    """
    k_str = f"{k_val}".replace('.', '_')
    n_str = f"{n_val}".replace('.', '_')
    csv_output_name = f"Axial_Profile_K{k_str}_N{n_str}.csv"

    # Note: Double backslashes \\\\@ are critical for Python -> Perl string injection
    cse_content = f"""# COMMAND FILE:
#   CFX Post Version = 17.2
# END

# --- 1. CONFIGURATION ---
!$x_min = {x_min};
!$x_max = {x_max};
!$n = {n_slices};

!for ($i = 1; $i < $n; $i++){{
    !$value = $x_min + ($x_max - $x_min) * $i/$n;
    !$j = $i + 1;

    # Create Slice
    ISOSURFACE: X_slice
        Domain List = /DOMAIN GROUP:All Domains
        Variable = X
        Value = $value
        Apply Instancing Transform = On
        Range = Global
    END

    # Evaluate Variables (Using \\\\@ for correct Perl parsing)
    !($p_stat, $u) = evaluate("ave(Pressure)\\@X_slice");
    !($rho,    $u) = evaluate("ave(Density)\\@X_slice");
    !($vel,    $u) = evaluate("ave(Velocity)\\@X_slice");
    !($sos,    $u) = evaluate("ave(SoSMixture)\\@X_slice");
    !($gam,    $u) = evaluate("ave(GammaRelaxation)\\@X_slice");

    # Mach Calc
    !if ($sos > 1.0) {{ $mach = $vel / $sos; }} else {{ $mach = 0.0; }}

    # Fractions
    !($vf_ml, $u) = evaluate("ave(VolFracMetaLiquid)\\@X_slice");
    !($vf_sl, $u) = evaluate("ave(VolFracSatLiquid)\\@X_slice");
    !($vf_v,  $u) = evaluate("ave(VolFracVapor)\\@X_slice");
    !($mf_ml, $u) = evaluate("ave(MassFracMetaLiquid)\\@X_slice");
    !($mf_sl, $u) = evaluate("ave(MassFracSatLiquid)\\@X_slice");
    !($mf_v,  $u) = evaluate("ave(MassFracVapor)\\@X_slice");

    # --- Header (First Iteration) ---
    !if ($i == 1) {{
        TABLE: DataExport
          Table Exists = True
          TABLE CELLS:
            A1 = "X [m]", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
            B1 = "Pressure [Pa]", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
            C1 = "Density [kg/m3]", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
            D1 = "Velocity [m/s]", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
            E1 = "SoS [m/s]", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
            F1 = "Mach [-]", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
            G1 = "Gamma [-]", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
            H1 = "VF MetaLiq [-]", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
            I1 = "VF SatLiq [-]", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
            J1 = "VF Vapor [-]", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
            K1 = "MF MetaLiq [-]", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
            L1 = "MF SatLiq [-]", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
            M1 = "MF Vapor [-]", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
          END
        END
    ! }} 

    # --- Data Row ---
    TABLE: DataExport
      Table Exists = True
      TABLE CELLS:
        A$j = "$value", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
        B$j = "$p_stat", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
        C$j = "$rho", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
        D$j = "$vel", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
        E$j = "$sos", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
        F$j = "$mach", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
        G$j = "$gam", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
        H$j = "$vf_ml", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
        I$j = "$vf_sl", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
        J$j = "$vf_v", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
        K$j = "$mf_ml", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
        L$j = "$mf_sl", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
        M$j = "$mf_v", False, False, False, Left, True, 0, Font Name, 1|1, %10.3e, True, ffffff, 000000, True
      END
    END

!}}

# --- EXPORT ---
>table save={csv_output_name}, name=DataExport
"""
    with open(filename, 'w') as f:
        f.write(cse_content)
    print(f"[Pipeline] CSE Generated: {filename}")


# ==========================================
# 3. PIPELINE FUNCTIONS
# ==========================================
def run_neb_pipeline(config):
    """
    Runs physics, generates verification plot, and writes all output files.
    Returns filenames for the wrapper to move.
    """
    # 1. Setup Naming
    k_val, n_val = config['K'], config['N']
    k_str = f"{k_val}".replace('.', '_')
    n_str = f"{n_val}".replace('.', '_')

    excel_name = f"NEB_Relaxation_K{k_str}_N{n_str}.xlsx"
    ccl_name = f"Material_K{k_str}_N{n_str}.ccl"
    cse_name = f"PostProcess_K{k_str}_N{n_str}.cse"
    dashboard_name = f"Verification_K{k_str}_N{n_str}.png"  # Unique name

    # 2. Run Physics
    t_in = config.get('T_in') or config.get('Temperature') or config.get('T')
    if t_in is None: raise ValueError("Config missing Temperature key")

    # Assuming stretched_exponential_closure is available in scope
    gamma_func = lambda Pi: stretched_exponential_closure(Pi, k=k_val, n=n_val)

    # Generate Table
    df_neb = generate_neb_table(P_in=config['P_in'], T_in=t_in, fluid=config['Fluid'], gamma_func=gamma_func)

    if df_neb is None: raise ValueError("Physics Failed")

    # 2b. HEM reference (gamma = 1 everywhere -> pure equilibrium limit).
    # Built with the same machinery at the same inlet state, so it overlays
    # point-for-point on the dashboard as a genuine HEM limit (not the NEB
    # curve drawn over itself). Optional caller override via config['df_ref'].
    df_ref = config.get('df_ref')
    if df_ref is None:
        print("   [Ref] Building HEM limit (gamma=1)...")
        try:
            df_ref = generate_neb_table(P_in=config['P_in'], T_in=t_in,
                                        fluid=config['Fluid'],
                                        gamma_func=lambda Pi: 1.0)
        except Exception as e:
            print(f"   [Warning] HEM limit build failed: {e}; "
                  f"falling back to self-overlay.")
            df_ref = df_neb

    # 3. Generate Verification Dashboard
    print(f"   [Plot] Generating Dashboard: {dashboard_name}...")
    try:
        # Overlay the HEM limit on the density / SoS / viscosity panels.
        plot_extended_verification(df_neb, df_ref, fluid_name=config['Fluid'])

        # Save to current dir (to be moved later)
        plt.savefig(dashboard_name, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"   [Warning] Dashboard generation failed: {e}")
        dashboard_name = None  # Flag that it failed

    # 4. Write Data Files
    with pd.ExcelWriter(excel_name) as w:
        df_neb.to_excel(w, sheet_name='Data', index=False)

    write_ccl_file(ccl_name, df_neb, config['P_in'], config['P_out'])
    # Geometry from config (defaults to the CO2 reference nozzle if absent).
    write_post_process_script(
        cse_name, k_val, n_val,
        x_min=config.get('x_min', 0.0),
        x_max=config.get('x_max', 0.0835),
        n_slices=config.get('n_slices', 100),
    )

    # Return ALL filenames (including the plot)
    return df_neb, excel_name, ccl_name, cse_name, dashboard_name


def run_neb_pipeline_in_folder(config, base_dir="."):
    """
    Wrapper: Runs pipeline then moves output to dedicated folder with run script.
    Generates a smart bash script that finds the .res file automatically.
    """

    # 1. Generate files in root
    df, excel, ccl, cse, dashboard = run_neb_pipeline(config)
    # 2. Setup Folder
    k_str = f"{config['K']}".replace('.', '_')
    n_str = f"{config['N']}".replace('.', '_')
    # Optional operating-point label (multi-point sweeps, e.g. water): keeps each
    # condition in a distinct, identifiable folder.
    label = config.get('label')
    if label:
        case_name = f"Sim_{label}_K{k_str}_N{n_str}"
    else:
        case_name = f"Sim_K{k_str}_N{n_str}"
    case_dir = os.path.join(base_dir, case_name)

    if not os.path.exists(case_dir): os.makedirs(case_dir)

    # 3. Move Files
    shutil.move(excel, os.path.join(case_dir, excel))
    shutil.move(ccl, os.path.join(case_dir, ccl))
    shutil.move(cse, os.path.join(case_dir, cse))
    # Check if dashboard exists before moving (safety first)
    if dashboard and os.path.exists(dashboard):
        shutil.move(dashboard, os.path.join(case_dir, dashboard))
    # --- 4. Generate Smart Bash Script ---
    res_base_name = f"Result_K{k_str}_N{n_str}"

    # Extract Cluster Params from Config
    cfx_bin = config.get("CFX_BIN", "")
    node_str = config.get("NODES", "node-4-16*40")
    mem_flags = config.get("MEM_FLAGS", "-S 1.9 -sizepar 1.5")

    # Resolve Paths
    solver = os.path.join(cfx_bin, 'cfx5solve') if cfx_bin else 'cfx5solve'
    post = os.path.join(cfx_bin, 'cfx5post') if cfx_bin else 'cfx5post'

    # The BASH Script
    bash_content = f"""#!/bin/bash
# Auto-generated NEB Script for {case_name}
# Config: Nodes={node_str}, Mem={mem_flags}

# Uncomment if you need to load modules explicitly here
# module load ansys/2024R2

cd "{os.path.abspath(case_dir)}"

echo "[INFO] Starting Solver..."
# Run Solver
{solver} -def ../base_setup.def -ccl {ccl} -double \\
  -start-method 'Intel MPI Local Parallel' -par-dist '{node_str}' \\
  {mem_flags} -name {res_base_name}

if [ $? -ne 0 ]; then echo "[ERROR] Solver Failed"; exit 1; fi

echo "[INFO] Starting Post-Process..."

# --- SMART RES FINDER ---
# CFX appends _001.res, _002.res etc. We need the latest one.
# ls -t sorts by time (newest first). head -n 1 takes the top one.
RES_FILE=$(ls -t {res_base_name}*.res 2>/dev/null | head -n 1)

if [ -f "$RES_FILE" ]; then
  echo "Found Result: $RES_FILE"
  {post} -batch {cse} -res "$RES_FILE"
  echo "[SUCCESS] Pipeline Complete"
else
  echo "[ERROR] No .res file found matching {res_base_name}*"
  exit 1
fi
"""

    bash_path = os.path.join(case_dir, "run_case.sh")
    with open(bash_path, 'w', encoding='utf-8') as f: f.write(bash_content)
    os.chmod(bash_path, 0o755)

    return bash_path