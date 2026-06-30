import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from scipy.optimize import fsolve
from tqdm import tqdm
from colorama import Fore, Style
import pandas as pd
import seaborn as sns

# Color palette for consistent plotting
COLORS = {
    'saturation': '#1f77b4',  # Blue
    'spinodal': '#d62728',    # Red
    'isentrope': '#2ca02c',   # Green
    'points': '#ff7f0e'       # Orange
}

def set_plot_style():
    """Set up the default plotting style."""
    sns.set_theme(style='whitegrid')
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'font.size': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

def find_Tsat(T, s, fluid):
    q = np.where(
        s > CP.PropsSI("S", "T", CP.PropsSI("Tcrit", fluid),
                       "P", CP.PropsSI("Pcrit", fluid), fluid),
        1, 0
    )
    return s - CP.PropsSI("S", "T", T, "Q", int(q), fluid)

def find_Psat(P, s, fluid):
    q = np.where(
        s > CP.PropsSI("S", "T", CP.PropsSI("Tcrit", fluid),
                       "P", CP.PropsSI("Pcrit", fluid), fluid),
        1, 0
    )
    return s - CP.PropsSI("S", "P", P, "Q", int(q), fluid)

def compute_viscosity(T, rho, fluid):
    """
    Compute viscosity of the metastable liquid state.
    Parameters:
        T: Temperature [K].
        rho: Density [kg/m³].
        fluid: Fluid ID for CoolProp (e.g., "REFPROP::WATER").
    Returns:
        Viscosity [Pa.s] or None if calculation fails.
    """
    try:
        return CP.PropsSI('V', 'T', T, 'D', rho, fluid)  # Compute viscosity
    except ValueError:
        return None

def compute_thermal_conductivity(T, rho, fluid):
    """
    Compute thermal conductivity of the metastable liquid state.
    Parameters:
        T: Temperature [K].
        rho: Density [kg/m³].
        fluid: Fluid ID for CoolProp (e.g., "REFPROP::WATER").
    Returns:
        Thermal conductivity [W/(m.K)] or None if calculation fails.
    """
    try:
        return CP.PropsSI('L', 'T', T, 'D', rho, fluid)  # Compute thermal conductivity
    except ValueError:
        return None

def find_metastable(x, dP, ds, rho_prev, T_prev, fluid):
    """
    Equation system to extrapolate into metastable liquid.
    Args:
        x: [rho, T] for next point.
        dP: Pressure step.
        ds: Entropy change (isentropic, so ~0).
        rho_prev: Density from previous step.
        T_prev: Temperature from previous step.
        fluid: CoolProp fluid name.
    Returns:
        Residuals for the pressure and entropy variations.
    """
    rho, T = x
    dpd_rho = CP.PropsSI("d(P)/d(D)|T", "D", rho, "T", T, fluid)
    dpd_T = CP.PropsSI("d(P)/d(T)|D", "D", rho, "T", T, fluid)
    dsd_rho = CP.PropsSI("d(S)/d(D)|T", "D", rho, "T", T, fluid)
    dsd_T = CP.PropsSI("d(S)/d(T)|D", "D", rho, "T", T, fluid)
    return [
        dpd_rho * (rho - rho_prev) + dpd_T * (T - T_prev) + dP,
        dsd_rho * (rho - rho_prev) + dsd_T * (T - T_prev) + ds,
    ]

def calc_meta(var, rho, T, rho_prev, T_prev, fluid):
    dvar_drho = CP.PropsSI(f"d({var})/d(D)|T", "D", rho, "T", T, fluid)
    dvar_dT   = CP.PropsSI(f"d({var})/d(T)|D", "D", rho, "T", T, fluid)
    return dvar_drho*(rho - rho_prev) + dvar_dT*(T - T_prev)

def calc_derivative_dD_dP_S(T, rho, fluid):
    """
    Compute the derivative d(D)/d(P)|S (density with respect to pressure at constant entropy).
    """
    return CP.PropsSI("d(D)/d(P)|S", "T", T, "D", rho, fluid)

def calc_derivative_dD_dP_H(T, rho, fluid):
    """
    Compute the derivative d(D)/d(P)|H (density with respect to pressure at constant enthalpy).
    """
    return CP.PropsSI("d(D)/d(P)|H", "T", T, "D", rho, fluid)

def calc_derivative_dD_dH_P(T, rho, fluid):
    """
    Compute the derivative d(D)/d(H)|P (density with respect to enthalpy at constant pressure).
    """
    return CP.PropsSI("d(D)/d(H)|P", "T", T, "D", rho, fluid)

def find_spinodal_density(T, rho_guess, fluid):
    """
    Determines spinodal density where dp/drho|T = 0.
    """
    return fsolve(
        lambda rho: CP.PropsSI("d(P)/d(D)|T", "D", rho, "T", T, fluid),
        rho_guess, xtol=1e-12
    )[0]

def print_isentrope_info(T_sat, P_sat, rho_sat, s_sat, h_sat):
    """Prints information about the starting conditions of an isentrope."""
    print(f"{Fore.CYAN}Starting new isentrope: {Style.RESET_ALL}"
          f"{Fore.YELLOW}T = {T_sat - 273.15:.2f} °C, {Style.RESET_ALL}"
          f"{Fore.GREEN}P = {P_sat / 1e5:.2f} bar, {Style.RESET_ALL}"
          f"{Fore.BLUE}ρ = {rho_sat:.3f} kg/m³, {Style.RESET_ALL}"
          f"{Fore.MAGENTA}S = {s_sat:.3f} J/kg.K, {Style.RESET_ALL}"
          f"{Fore.RED}H = {h_sat:.3f} J/kg{Style.RESET_ALL}")

def extrapolate_isentrope(sat_P, sat_T, sat_rho, sat_h, sat_s, fluid, min_pressure=0):
    """
    Extrapolate properties isentropically into the metastable region,
    stopping at either the spinodal limit or zero pressure.
    """
    pressure_range = np.linspace(sat_P, min_pressure, 500)
    dP = pressure_range[0] - pressure_range[1]

    isentrope_data = {
        "Pressure [Pa]": [],
        "Density [kg/m3]": [],
        "Temperature [K]": [],
        "Entropy [J/kg.K]": [],
        "Enthalpy [J/kg]": [],
        "Viscosity [Pa.s]": [],
        "SpeedOfSound [m/s]": [],
        "Thermal Conductivity [W/m.K]": [],
        "d(D)/d(P)|S [kg/m3/Pa]": [],
        "d(D)/d(P)|H [kg/m3/Pa]": [],
        "d(D)/d(H)|P [m3/(J.kg)]": [],
    }

    rho_prev, T_prev, h_prev = sat_rho, sat_T, sat_h
    spinodal_point = None

    for P in pressure_range:
        rho_next, T_next = fsolve(
            lambda x: find_metastable(x, dP, 0, rho_prev, T_prev, fluid),
            [rho_prev, T_prev],
            xtol=1e-10
        )

        dP_dRho = CP.PropsSI("d(P)/d(D)|T", "D", rho_next, "T", T_next, fluid)
        if dP_dRho <= 1e-8:
            spinodal_point = (P, T_next)
            print(f"[INFO] Spinodal reached at P = {P / 1e5:.3f} bar")
            break

        dh = calc_meta("H", rho_next, T_next, rho_prev, T_prev, fluid)
        h_next = h_prev + dh

        dD_dP_S = calc_derivative_dD_dP_S(T_next, rho_next, fluid)
        dD_dP_H = calc_derivative_dD_dP_H(T_next, rho_next, fluid)
        dD_dH_P = calc_derivative_dD_dH_P(T_next, rho_next, fluid)

        try:
            viscosity = compute_viscosity(T_next, rho_next, fluid)
            conductivity = compute_thermal_conductivity(T_next, rho_next, fluid)
            if dD_dP_S > 1e-20:
                c_meta = np.sqrt(1.0 / dD_dP_S)
            else:
                c_meta = np.nan
        except ValueError:
            viscosity, conductivity, c_meta = None, None, np.nan

        isentrope_data["Pressure [Pa]"].append(P)
        isentrope_data["Density [kg/m3]"].append(rho_next)
        isentrope_data["Temperature [K]"].append(T_next)
        isentrope_data["Entropy [J/kg.K]"].append(sat_s)
        isentrope_data["Enthalpy [J/kg]"].append(h_next)
        isentrope_data["Viscosity [Pa.s]"].append(viscosity)
        isentrope_data["SpeedOfSound [m/s]"].append(c_meta)
        isentrope_data["Thermal Conductivity [W/m.K]"].append(conductivity)
        isentrope_data["d(D)/d(P)|S [kg/m3/Pa]"].append(dD_dP_S)
        isentrope_data["d(D)/d(P)|H [kg/m3/Pa]"].append(dD_dP_H)
        isentrope_data["d(D)/d(H)|P [m3/(J.kg)]"].append(dD_dH_P)

        rho_prev, T_prev, h_prev = rho_next, T_next, h_next

    return isentrope_data, spinodal_point

def compute_equilibrium_leg(pressure_array, s_in, fluid):
    """
    Computes equilibrium properties (Sat Liquid, Sat Vapor, and Equilibrium Mixture)
    for the given pressure array along the isentrope defined by s_in.
    """
    eq_data = {
        "rho_l": [], "rho_v": [],
        "mu_l": [], "mu_v": [],
        "h_l": [], "h_v": [],
        "x_eq": [],
        "c_hem": []
    }

    DP_DERIV = 10.0 # Pa

    try:
        iterator = tqdm(pressure_array, desc="Computing Equilibrium Leg", leave=False)
    except:
        iterator = pressure_array

    for P in iterator:
        try:
            r_l = CP.PropsSI('D', 'P', P, 'Q', 0, fluid)
            r_v = CP.PropsSI('D', 'P', P, 'Q', 1, fluid)
            mu_l = CP.PropsSI('V', 'P', P, 'Q', 0, fluid)
            mu_v = CP.PropsSI('V', 'P', P, 'Q', 1, fluid)
            h_l = CP.PropsSI('H', 'P', P, 'Q', 0, fluid)
            h_v = CP.PropsSI('H', 'P', P, 'Q', 1, fluid)
            s_l = CP.PropsSI('S', 'P', P, 'Q', 0, fluid)
            s_v = CP.PropsSI('S', 'P', P, 'Q', 1, fluid)

            if abs(s_v - s_l) < 1e-5:
                x = 0.0
            else:
                x = (s_in - s_l) / (s_v - s_l)
            x = np.clip(x, 0.0, 1.0)

            c_l_phase = CP.PropsSI('A', 'P', P, 'Q', 0, fluid)
            c_v_phase = CP.PropsSI('A', 'P', P, 'Q', 1, fluid)
            cp_l = CP.PropsSI('C', 'P', P, 'Q', 0, fluid)
            cp_v = CP.PropsSI('C', 'P', P, 'Q', 1, fluid)
            T_sat = CP.PropsSI('T', 'P', P, 'Q', 0, fluid)

            vol_v = x / r_v
            vol_l = (1.0 - x) / r_l
            vol_total = vol_v + vol_l

            if vol_total < 1e-12:
                 alpha_v = 0.0 if x < 0.5 else 1.0
            else:
                 alpha_v = vol_v / vol_total
            alpha_l = 1.0 - alpha_v

            rho_mix_eq = alpha_v * r_v + alpha_l * r_l

            sl_p = CP.PropsSI('S', 'P', P + DP_DERIV, 'Q', 0, fluid)
            sl_m = CP.PropsSI('S', 'P', P - DP_DERIV, 'Q', 0, fluid)
            ds_dp_l = (sl_p - sl_m) / (2 * DP_DERIV)

            sv_p = CP.PropsSI('S', 'P', P + DP_DERIV, 'Q', 1, fluid)
            sv_m = CP.PropsSI('S', 'P', P - DP_DERIV, 'Q', 1, fluid)
            ds_dp_v = (sv_p - sv_m) / (2 * DP_DERIV)

            term_mech = (alpha_l / (r_l * c_l_phase**2)) + \
                        (alpha_v / (r_v * c_v_phase**2))

            term_therm = T_sat * (
                ((alpha_l * r_l / cp_l) * ds_dp_l**2) +
                ((alpha_v * r_v / cp_v) * ds_dp_v**2)
            )

            inv_rho_c2 = term_mech + term_therm
            c_mix_hem = np.sqrt(1.0 / (rho_mix_eq * inv_rho_c2))

            eq_data["rho_l"].append(r_l)
            eq_data["rho_v"].append(r_v)
            eq_data["mu_l"].append(mu_l)
            eq_data["mu_v"].append(mu_v)
            eq_data["h_l"].append(h_l)
            eq_data["h_v"].append(h_v)
            eq_data["x_eq"].append(x)
            eq_data["c_hem"].append(c_mix_hem)

        except ValueError:
            for k in eq_data: eq_data[k].append(np.nan)

    return pd.DataFrame(eq_data)

def apply_complex_physics(df):
    """
    Augments the DataFrame with Mass Fractions (PRIMARY), Volume Fractions
    (derived), Viscosity, and the DEM Speed of Sound (harmonic compressibility).

    MASS-PRIMARY formulation: Gamma is the equilibrated MASS fraction of the
    mixture. The mass fractions are therefore computed first and the volume
    fractions are derived from them, so that simultaneously
        sum_i X_i = 1,   sum_i eps_i = 1,   rho_mix = sum_i eps_i rho_i,
    and X_v + X_l_sat = Gamma exactly. (Relaxing Gamma on volume instead would
    make the realized equilibrated mass drift from the prescribed Gamma by the
    factor rho_eq/rho_mix.)
    """
    # 1. PRIMARY mass fractions (Gamma = equilibrated mass fraction)
    df['X_v']      = df['Gamma'] * df['x_eq']
    df['X_l_sat']  = df['Gamma'] * (1.0 - df['x_eq'])
    df['X_l_meta'] = 1.0 - df['Gamma']                 # closure by difference

    # 2. Mixture density from mass-weighted specific volumes
    v_mix = (df['X_v']      / df['rho_v']) + \
            (df['X_l_sat']  / df['rho_l']) + \
            (df['X_l_meta'] / df['rho_m'])
    df['Rho_Mixture'] = 1.0 / v_mix

    # 3. Volume fractions DERIVED from mass: eps_i = X_i * rho_mix / rho_i
    df['eps_v']      = df['X_v']      * df['Rho_Mixture'] / df['rho_v']
    df['eps_l_sat']  = df['X_l_sat']  * df['Rho_Mixture'] / df['rho_l']
    df['eps_l_meta'] = df['X_l_meta'] * df['Rho_Mixture'] / df['rho_m']

    # 4. Viscosity: volume-weighted homogeneous-mixture average
    df['Mu_Mixture'] = (df['eps_v'] * df['mu_v']) + \
                       (df['eps_l_sat'] * df['mu_l']) + \
                       (df['eps_l_meta'] * df['mu_m'])

    # 5. Equilibrium reference density (for the acoustic blend)
    v_v = 1.0 / df['rho_v']
    v_l = 1.0 / df['rho_l']
    v_mix_eq = df['x_eq'] * v_v + (1.0 - df['x_eq']) * v_l
    df['Rho_Equilibrium'] = 1.0 / v_mix_eq

    # 6. Speed of Sound (DEM mass-weighted compliance average)
    #    1/(rho_mix^2 c^2) = Gamma/(rho_eq^2 c_hem^2) + (1-Gamma)/(rho_m^2 c_m^2)
    inv_rho_c2_eq   = 1.0 / ((df['Rho_Equilibrium']**2) * (df['c_hem']**2))
    inv_rho_c2_meta = 1.0 / ((df['rho_m']**2) * (df['c_m']**2))

    inv_rho_c2_mix = (df['Gamma'] * inv_rho_c2_eq) + ((1.0 - df['Gamma']) * inv_rho_c2_meta)
    df['SoS_Mixture'] = np.sqrt(1.0 / (inv_rho_c2_mix * (df['Rho_Mixture']**2)))

    return df

def build_neb_closure(meta_dict, eq_df, P_spin, P_nuc, gamma_func):
    """
    Constructs the NEB table using vectorized operations.
    Combines Metastable and Equilibrium legs, then applies complex physics.
    """
    df = pd.DataFrame()
    df['Pressure'] = meta_dict['Pressure [Pa]']
    df['rho_m']    = meta_dict['Density [kg/m3]']
    df['mu_m']     = meta_dict['Viscosity [Pa.s]']
    df['c_m']      = meta_dict['SpeedOfSound [m/s]']

    df['rho_l'] = eq_df['rho_l'].values
    df['rho_v'] = eq_df['rho_v'].values
    df['mu_l']  = eq_df['mu_l'].values
    df['mu_v']  = eq_df['mu_v'].values
    df['x_eq']  = eq_df['x_eq'].values
    df['c_hem'] = eq_df['c_hem'].values

    P_vals = df['Pressure'].values
    denom = P_nuc - P_spin
    if abs(denom) < 1e-9:
        Pi_vals = np.where(P_vals >= P_nuc, 1.0, 0.0)
    else:
        Pi_vals = (P_vals - P_spin) / denom

    Pi_vals = np.clip(Pi_vals, 0.0, 1.0)
    Gamma_vals = [gamma_func(pi) for pi in Pi_vals]

    df['Pi'] = Pi_vals
    df['Gamma'] = Gamma_vals
    df = apply_complex_physics(df)

    return df

def sigmoid_closure(Pi, k=10):
    """Sigmoidal transition."""
    if Pi >= 1: return 0.0
    if Pi <= 0: return 1.0
    val = 1.0 / (1.0 + np.exp(-k * (0.5 - Pi)))
    return val

def stretched_exponential_closure(Pi, k=5, n=1):
    """Stretched Exponential Closure (Avrami-like)."""
    Pi = np.clip(Pi, 0.0, 1.0)
    driver = 1.0 - Pi
    numerator = 1.0 - np.exp(-k * (driver**n))
    denominator = 1.0 - np.exp(-k)
    return numerator / denominator

def shifted_sigmoid_closure(Pi, k=10, Pi_0=0.5):
    """Sigmoid with controllable center Pi_0."""
    if Pi >= 1: return 0.0
    if Pi <= 0: return 1.0
    val = 1.0 / (1.0 + np.exp(-k * (Pi_0 - Pi)))
    return val

def generate_neb_table(P_in, T_in, fluid, gamma_func, P_min_limit=0):
    """
    The main orchestrator for generating the full barotropic property table.
    
    This function constructs a high-resolution dataset covering three physical regimes:
    1. **Single Phase (Liquid)**: From inlet pressure down to the nucleation pressure.
    2. **NEB Transition (Metastable)**: Extrapolates into the metastable region using an
       isentropic assumption and applies the Relaxation Closure ($\gamma$) to mix 
       metastable liquid with equilibrium properties.
    3. **Equilibrium Tail**: Once the spinodal is reached or the relaxation is complete,
       the fluid follows the standard HEM (Homogeneous Equilibrium Model) path.
       
    Parameters:
        P_in, T_in: Stagnation/Inlet conditions.
        fluid: CoolProp-compatible fluid string.
        gamma_func: The relaxation closure function (e.g., stretched exponential).
        P_min_limit: Optional minimum pressure cutoff for the table.
        
    Returns:
        DataFrame: A consolidated table with mixture density, viscosity, speed of sound,
                   and volume/mass fractions.
    """
    print(f"{Fore.CYAN}--- Starting NEB Table Generation ---{Style.RESET_ALL}")

    P_TR = CP.PropsSI("ptriple", fluid)
    P_trip = P_TR
    if P_min_limit and P_min_limit > P_trip:
        P_target = P_min_limit
    else:
        P_target = P_trip * 1.01

    s_in = CP.PropsSI("S", "P", P_in, "T", T_in, fluid)
    P_sat_guess = CP.PropsSI("P", "T", T_in, "Q", 0, fluid)

    try:
        P_nuc = fsolve(lambda p: CP.PropsSI("S", "P", p, "Q", 0, fluid) - s_in, P_sat_guess)[0]
    except:
        P_nuc = P_in

    T_nuc = CP.PropsSI("T", "P", P_nuc, "Q", 0, fluid)
    rho_nuc = CP.PropsSI("D", "P", P_nuc, "Q", 0, fluid)
    h_nuc = CP.PropsSI("H", "P", P_nuc, "Q", 0, fluid)

    print(f"Nucleation Pressure: {P_nuc/1e5:.3f} bar")

    print(f"Computing Region 1: Single Phase...")
    p_single = np.linspace(P_in, P_nuc, 50)
    rho_single = []
    mu_single  = []
    c_single   = []

    for p in p_single:
        rho_single.append(CP.PropsSI("D", "P", p, "S", s_in, fluid))
        mu_single.append(CP.PropsSI("V", "P", p, "S", s_in, fluid))
        c_single.append((CP.PropsSI("D(P)/D(D)|S", "P", p, "S", s_in, fluid))**0.5)
    df_single = pd.DataFrame({
        "Pressure": p_single,
        "Rho_Mixture": rho_single,
        "Mu_Mixture": mu_single,
        "SoS_Mixture": c_single,
        "Rho_Equilibrium": rho_single,
        "Gamma": np.zeros_like(p_single),
        "Region": "1-SinglePhase",
        "eps_l_meta": np.ones_like(p_single),
        "eps_l_sat": np.zeros_like(p_single),
        "eps_v": np.zeros_like(p_single),
        "X_l_meta": np.ones_like(p_single),
        "X_l_sat": np.zeros_like(p_single),
        "X_v": np.zeros_like(p_single)
    })

    print(f"Computing Region 2: NEB Transition...")
    meta_dict, spinodal_pt = extrapolate_isentrope(P_nuc, T_nuc, rho_nuc, h_nuc, s_in, fluid, min_pressure=P_target)

    if spinodal_pt:
        # Positive spinodal located (e.g. CO2): anchor Pi there.
        P_spin = spinodal_pt[0]
        print(f"{Fore.YELLOW}Spinodal reached at {P_spin/1e5:.3f} bar{Style.RESET_ALL}")
    else:
        # Stability held down to the (near-zero) detection floor: the liquid
        # spinodal lies in negative-pressure territory and is physically
        # inaccessible (e.g. water). Anchor Pi at zero, giving the
        # inception-anchored coordinate Pi = P / P_nuc. Region 3 then switches
        # off and the NEB blend runs from P_nuc down to the table floor.
        P_spin = 0.0
        print(f"{Fore.YELLOW}Spinodal in negative-pressure region; "
              f"anchoring Pi at zero (inception-anchored).{Style.RESET_ALL}")

    eq_df_r2 = compute_equilibrium_leg(meta_dict["Pressure [Pa]"], s_in, fluid)
    df_neb = build_neb_closure(meta_dict, eq_df_r2, P_spin, P_nuc, gamma_func)
    df_neb["Region"] = "2-NEB"

    df_eq_tail = pd.DataFrame()
    if P_spin > P_target:
        print(f"Computing Region 3: Equilibrium Tail...")
        p_tail = np.linspace(P_spin, P_target, 200)
        eq_tail = compute_equilibrium_leg(p_tail, s_in, fluid)
        x = eq_tail['x_eq'].values
        v_v = 1.0 / eq_tail['rho_v'].values
        v_l = 1.0 / eq_tail['rho_l'].values
        v_mix = x * v_v + (1.0 - x) * v_l
        rho_mix = 1.0 / v_mix
        
        eps_v_vals = (x * v_v) / v_mix
        eps_l_sat_vals = ((1.0 - x) * v_l) / v_mix
        mu_mix = eps_v_vals * eq_tail['mu_v'].values + eps_l_sat_vals * eq_tail['mu_l'].values

        df_eq_tail = pd.DataFrame({
            "Pressure": p_tail,
            "Rho_Mixture": rho_mix,
            "Mu_Mixture": mu_mix,
            "SoS_Mixture": eq_tail['c_hem'].values,
            "Rho_Equilibrium": rho_mix,
            "Gamma": np.ones_like(p_tail),
            "Region": "3-Equilibrium",
            "eps_v": eps_v_vals,
            "eps_l_sat": eps_l_sat_vals,
            "eps_l_meta": np.zeros_like(p_tail),
            "X_v": x,
            "X_l_sat": 1.0 - x,
            "X_l_meta": np.zeros_like(p_tail)
        })

    full_df = pd.concat([df_single, df_neb, df_eq_tail], ignore_index=True)
    full_df = full_df.drop_duplicates(subset="Pressure") \
                     .sort_values(by="Pressure", ascending=False) \
                     .reset_index(drop=True)

    return full_df

def export_neb_table(final_table, fluid, P_inlet, T_inlet, excel_filename="NEB_FullRange_with_Metadata.xlsx", csv_filename="NEB_FullRange.csv", **extra_metadata):
    """
    Exports the generated NEB table to:
    1. A CSV file formatted for ANSYS CFX.
    2. An Excel file with a separate metadata sheet for record keeping.
    
    Parameters:
        final_table (DataFrame): The result from generate_neb_table.
        fluid (str): Fluid name used in simulation.
        P_inlet (float): Inlet pressure [Pa].
        T_inlet (float): Inlet temperature [K].
        excel_filename (str): Target filename for the Excel export.
        csv_filename (str): Target filename for the CFX CSV export.
        **extra_metadata: Arbitrary additional metadata to include in the export (e.g., closure coefficients).
    """
    if final_table is None:
        print(f"{Fore.RED}Export failed: final_table is None.{Style.RESET_ALL}")
        return

    # 1. Export for CFX (CSV)
    try:
        final_table[["Pressure", "Rho_Mixture"]].to_csv(csv_filename, index=False, header=False)
        print(f"{Fore.GREEN}Table exported to {csv_filename} covering {final_table['Pressure'].max()/1e5:.1f} to {final_table['Pressure'].min()/1e5:.1f} bar.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}CSV export failed: {e}{Style.RESET_ALL}")

    # 2. Export to Excel with Metadata
    # Prepare Metadata
    metadata_data = {
        "Parameter": [
            "Fluid", 
            "Inlet Pressure [bar]", 
            "Inlet Temperature [K]", 
            "Export Date",
            "Nucleation Pressure [bar]",
            "Spinodal Pressure [bar]"
        ],
        "Value": [
            fluid,
            P_inlet / 1e5,
            T_inlet,
            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            final_table[final_table['Region'] == '2-NEB']['Pressure'].max() / 1e5 if '2-NEB' in final_table['Region'].values else "N/A",
            final_table[final_table['Region'] == '2-NEB']['Pressure'].min() / 1e5 if '2-NEB' in final_table['Region'].values else "N/A"
        ]
    }

    # Add extra metadata (e.g. relaxation parameters)
    for k, v in extra_metadata.items():
        metadata_data["Parameter"].append(k)
        metadata_data["Value"].append(v)

    metadata_df = pd.DataFrame(metadata_data)

    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            final_table.to_excel(writer, sheet_name='NEB_Table_Data', index=False)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        print(f"{Fore.GREEN}Table also exported to {excel_filename} with metadata sheets.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Excel export failed (check if openpyxl is installed): {e}{Style.RESET_ALL}")
        
        # Fallback to CSV for full table and metadata
        full_csv = excel_filename.replace(".xlsx", "_FullTable.csv")
        meta_csv = excel_filename.replace(".xlsx", "_Metadata.csv")
        try:
            final_table.to_csv(full_csv, index=False)
            metadata_df.to_csv(meta_csv, index=False)
            print(f"{Fore.YELLOW}Fallback: Full table exported to {full_csv} and metadata to {meta_csv}.{Style.RESET_ALL}")
        except Exception as e_csv:
            print(f"{Fore.RED}Fallback CSV export failed: {e_csv}{Style.RESET_ALL}")

def plot_verification_dashboard(df, fluid_name, p_nuc=None, p_spin=None):
    """Plotting verification dashboard."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(df["Pressure"]/1e5, df["Rho_Mixture"], 'b-', linewidth=2, label="NEB Mixture")
    axes[0].plot(df["Pressure"]/1e5, df["Rho_Equilibrium"], 'k--', alpha=0.5, label="Equilibrium Ref")
    axes[0].set_title("Density")
    axes[0].set_ylabel("Density [kg/m3]")

    axes[1].plot(df["Pressure"]/1e5, df["SoS_Mixture"], 'r-', linewidth=2)
    axes[1].set_title("Speed of Sound (DEM)")
    axes[1].set_ylabel("Speed of Sound [m/s]")

    axes[2].plot(df["Pressure"]/1e5, df["Mu_Mixture"], 'g-', linewidth=2)
    axes[2].set_title("Viscosity")
    axes[2].set_ylabel("Viscosity [Pa.s]")

    for ax in axes:
        ax.set_xlabel("Pressure [bar]")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        if p_nuc is not None: ax.axvline(p_nuc/1e5, color='k', linestyle=':', alpha=0.5)
        if p_spin is not None: ax.axvline(p_spin/1e5, color='r', linestyle=':', alpha=0.5)

    plt.suptitle(f"Verification Dashboard: {fluid_name}", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_fractions(df, p_nuc=None, p_spin=None):
    """Plot volume fractions."""
    plt.figure(figsize=(10, 6))

    plt.plot(df["Pressure"]/1e5, df["eps_l_meta"], label=r"$\epsilon_{meta}$ (Metastable Liq)", color='green')
    plt.plot(df["Pressure"]/1e5, df["eps_l_sat"], label=r"$\epsilon_{sat}$ (Saturated Liq)", color='blue')
    plt.plot(df["Pressure"]/1e5, df["eps_v"], label=r"$\epsilon_{v}$ (Vapor)", color='red')

    total = df["eps_l_meta"] + df["eps_l_sat"] + df["eps_v"]
    plt.plot(df["Pressure"]/1e5, total, 'k--', label="Sum Check", alpha=0.3)

    plt.xlabel("Pressure [bar]")
    plt.ylabel("Volume Fraction [-]")
    plt.title("Phase Volume Fractions Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    if p_nuc is not None: plt.axvline(p_nuc/1e5, color='k', linestyle=':', alpha=0.5)
    if p_spin is not None: plt.axvline(p_spin/1e5, color='r', linestyle=':', alpha=0.5)

    plt.show()

def plot_n_sensitivity(P_in, T_in, fluid, n_values=[0.0, 1.0, 3.0, 5.0], k=5.0, zoomed=False):
    """
    Perform sensitivity analysis on the parameter 'n' in the stretched exponential closure.
    
    Args:
        P_in, T_in: Inlet conditions.
        fluid: Fluid name.
        n_values: List of 'n' values to investigate. n=0 is treated as HEM.
        k: Fixed 'k' value.
        zoomed: If True, zoom into the metastable region.
    """
    results_db = {}
    print(f"Running n-Sensitivity Simulations (Fixed k={k})...")

    for n in n_values:
        func = lambda Pi: stretched_exponential_closure(Pi, k=k, n=n)
        df = generate_neb_table(P_in, T_in, fluid=fluid, gamma_func=func, P_min_limit=0)
        if df is not None:
            results_db[n] = df

    if not results_db:
        print("No results to plot.")
        return

    plt.figure(figsize=(12, 7))

    # Identify a reference dataframe for limits
    any_df = next(iter(results_db.values()))
    
    for n, df in results_db.items():
        if n == 0:
            plt.plot(df["Pressure"]/1e5, df["Rho_Mixture"],
                     color='black', linestyle='--', linewidth=1.5, alpha=0.6, zorder=10,
                     label="Reference Case (HEM)")
        else:
            plt.plot(df["Pressure"]/1e5, df["Rho_Mixture"],
                     linewidth=2, alpha=0.8, label=f'NEB (n={n})')

    # Add limits
    neb_subset = any_df[any_df["Region"] == "2-NEB"]
    if not neb_subset.empty:
        p_nuc = neb_subset["Pressure"].max()
        p_spin = neb_subset["Pressure"].min()

        plt.axvline(x=p_nuc/1e5, color='k', linestyle=':', alpha=0.5)
        plt.text(p_nuc/1e5, plt.ylim()[0], ' Nucleation', rotation=90, va='bottom', ha='right')

        plt.axvline(x=p_spin/1e5, color='r', linestyle=':', alpha=0.5)
        plt.text(p_spin/1e5, plt.ylim()[0], ' Spinodal', rotation=90, va='bottom', ha='right', color='r')

        if zoomed:
            P_range = p_nuc - p_spin
            plt.xlim((p_spin - 0.1 * P_range) / 1e5, (p_nuc + 0.1 * P_range) / 1e5)
            plt.title(f"Zoomed Analysis: Metastable Transition Zone\nFluid: {fluid} | $k={k}$", fontsize=16)
            plt.minorticks_on()
        else:
            plt.title(f"Non-Equilibrium Density Profiles vs. Reference\nFluid: {fluid} | k={k}")

    plt.xlabel("Pressure [bar]")
    plt.ylabel("Density [kg/m³]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return results_db

def plot_k_sensitivity(P_in, T_in, fluid, k_values=[1.0, 3.0, 10.0, 50.0], n=3.0):
    """
    Perform sensitivity analysis on the parameter 'k' in the stretched exponential closure.
    
    Args:
        P_in, T_in: Inlet conditions.
        fluid: Fluid name.
        k_values: List of 'k' values to investigate.
        n: Fixed 'n' value.
    """
    results_db_k = {}
    print(f"Running k-Sensitivity Simulations (Fixed n={n})...")

    # --- Run Reference (Equilibrium) ---
    print("  > Simulating Reference...")
    df_ref = generate_neb_table(P_in, T_in, fluid=fluid,
                                gamma_func=lambda Pi: 1.0,
                                P_min_limit=0)

    # --- Run K Variations ---
    for k in k_values:
        print(f"  > Simulating k={k}...")
        func = lambda Pi: stretched_exponential_closure(Pi, k=k, n=n)
        df = generate_neb_table(P_in, T_in, fluid=fluid, gamma_func=func, P_min_limit=0)
        if df is not None:
            results_db_k[k] = df

    if not results_db_k:
        print("No results to plot.")
        return

    plt.figure(figsize=(14, 7))

    # --- Subplot 1: Gamma Function Shape ---
    plt.subplot(1, 2, 1)
    pi_range = np.linspace(0, 1, 100)
    for k in k_values:
        gamma_vals = [stretched_exponential_closure(pi, k=k, n=n) for pi in pi_range]
        plt.plot(pi_range, gamma_vals, linewidth=2, label=f'k={k}')

    plt.xlabel(r'Normalized Pressure $\Pi$')
    plt.ylabel(r'Equilibrium Fraction $\gamma$')
    plt.title(f'Effect of K on Closure Shape (n={n})')
    # plt.gca().invert_xaxis()
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.text(0.1, 0.2, r"$\leftarrow$ Spinodal ($P_{spin}$)", transform=plt.gca().transAxes)
    plt.text(0.9, 0.2, r"Nucleation ($P_{nuc}$) $\rightarrow$", transform=plt.gca().transAxes, ha='right')
    plt.axvline(x=0.0, color='k', linestyle=':', linewidth=1.5)
    plt.axvline(x=1.0, color='r', linestyle=':', linewidth=1.5)

    # --- Subplot 2: Density Path (Zoomed) ---
    plt.subplot(1, 2, 2)
    plt.plot(df_ref["Pressure"]/1e5, df_ref["Rho_Equilibrium"],
             color='black', linestyle='--', linewidth=2, alpha=0.6, label="Reference ($\gamma=1$)")

    for k, df in results_db_k.items():
        plt.plot(df["Pressure"]/1e5, df["Rho_Mixture"], linewidth=2, label=f'k={k}')

    neb_subset = df_ref[df_ref["Region"] == "2-NEB"]
    if not neb_subset.empty:
        p_nuc = neb_subset["Pressure"].max()
        p_spin = neb_subset["Pressure"].min()
        plt.axvline(x=p_nuc/1e5, color='k', linestyle=':', linewidth=1.5)
        plt.text(p_nuc/1e5, plt.ylim()[0], ' $P_{nuc}$', rotation=90, va='bottom')
        plt.axvline(x=p_spin/1e5, color='r', linestyle=':', linewidth=1.5)
        plt.text(p_spin/1e5, plt.ylim()[0], ' $P_{spin}$', rotation=90, va='bottom', color='r')
        P_width = p_nuc - p_spin
        plt.xlim((p_spin - 0.2*P_width)/1e5, (p_nuc + 0.2*P_width)/1e5)

    plt.xlabel("Pressure [bar]")
    plt.ylabel("Density [kg/m³]")
    plt.title(f"Density Sensitivity to Rate K (n={n})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return results_db_k


# ==========================================
# 1. DATA STRUCTURE
# ==========================================
class PolynomialSegment:
    """
    Atomic data structure representing a single piecewise polynomial segment.
    
    This class handles the normalization, evaluation, and derivative calculation
    for a specific pressure range. It uses Horner's Method for efficient evaluation
    and ensures consistency between the mathematical fit and the physical units.
    """

    def __init__(self, coeffs, p_min, p_max, p_inlet):
        """
        Initialize a polynomial segment.
        
        Parameters:
            coeffs (list): Coefficients in ascending order [a0, a1, a2, ...].
            p_min (float): Lower pressure boundary of the segment [Pa].
            p_max (float): Upper pressure boundary of the segment [Pa].
            p_inlet (float): Reference pressure used for normalization [Pa].
        """
        self.coeffs = coeffs  # [a0, a1, a2...] (Ascending powers)
        self.p_min = p_min
        self.p_max = p_max
        self.p_inlet = p_inlet  # Normalization scale

    def evaluate(self, p_raw):
        """
        Evaluate the polynomial at a given raw pressure using Horner's Method.
        The pressure is internally normalized by p_inlet to maintain numerical stability.
        """
        p_hat = p_raw / self.p_inlet
        val = self.coeffs[-1]
        for c in reversed(self.coeffs[:-1]):
            val = val * p_hat + c
        return val

    def derivative(self, p_raw):
        """
        Calculate the analytical derivative (dy/dP) at a given pressure.
        Applies the chain rule to account for the internal pressure normalization.
        """
        p_hat = p_raw / self.p_inlet
        # Derivative coefficients: [1*a1, 2*a2, 3*a3...]
        deriv_coeffs = [i * c for i, c in enumerate(self.coeffs) if i > 0]

        if not deriv_coeffs: return 0.0

        val = deriv_coeffs[-1]
        for c in reversed(deriv_coeffs[:-1]):
            val = val * p_hat + c

        return val / self.p_inlet  # Chain rule: d(p_hat)/dp = 1/p_inlet


# ==========================================
# 2. FITTER CLASS (High-to-Low Stitching)
# ==========================================
class NebFitter:
    """
    Orchestrates the piecewise fitting of thermodynamic properties across multiple regions.
    
    The fitter implements a 'High-to-Low' stitching strategy:
    1. It starts at the highest pressure (Single Phase), which is considered the 'Anchor'.
    2. It moves down in pressure to the NEB region and then the Equilibrium region.
    3. At each boundary, it calculates the 'Gap' between segments and shifts the new 
       segment's constant term (a0) to ensure C0 continuity.
    
    This approach prevents numerical 'jumps' at phase transitions that would otherwise
    cause CFD solvers to diverge.
    """
    def __init__(self, df, P_inlet):
        """
        Initialize the Fitter with a raw simulation DataFrame.
        
        Parameters:
            df (DataFrame): Table containing 'Pressure', 'Region', and property columns.
            P_inlet (float): The inlet (maximum) pressure for normalization.
        """
        self.df = df.sort_values('Pressure')
        self.P_inlet = P_inlet

        self.regions = {
            '1-SinglePhase': df[df['Region'] == '1-SinglePhase'],
            '2-NEB': df[df['Region'] == '2-NEB'],
            '3-Equilibrium': df[df['Region'] == '3-Equilibrium']
        }

    def fit_variable(self, col_name, degrees=None):
        """
        Perform the piecewise fitting for a specific variable using Adaptive Degrees.
        
        Adaptive Degrees allow for high-order polynomials in complex transition zones (NEB)
        and low-order polynomials in smooth regions (Equilibrium) to prevent 
        Runge's Phenomenon (unwanted oscillations).
        
        Parameters:
            col_name (str): The column name in the DataFrame to fit (e.g., 'Rho_Mixture').
            degrees (dict/int): A mapping of region names to polynomial degrees.
                               If None, smart defaults are used.
        
        Returns:
            list: A list of PolynomialSegment objects sorted from Low to High pressure.
        """
        print(f"{Fore.CYAN}Fitting {col_name} (Stitched High-to-Low)...{Style.RESET_ALL}")

        # 1. Smart Defaults (Prevents Runge's Phenomenon in Eq Tail)
        if degrees is None:
            degrees = {
                '1-SinglePhase': 5,  # Moderate (Linear-ish compressibility)
                '2-NEB': 8,  # High (Complex S-shape transition)
                '3-Equilibrium': 4  # Low (Smooth saturation tail - Prevents peaks!)
            }
        elif isinstance(degrees, int):
            # Fallback if user passes a single integer
            d = degrees
            degrees = {'1-SinglePhase': d, '2-NEB': d, '3-Equilibrium': d}

        fit_order = ['1-SinglePhase', '2-NEB', '3-Equilibrium']

        temp_segments = {}
        target_val_at_stitch = None
        stitch_pressure = None

        for name in fit_order:
            sub = self.regions[name]
            if sub.empty: continue

            # Get specific degree for this region
            deg = degrees.get(name, 5)

            # 1. Fit Raw Data
            X = sub['Pressure'].values / self.P_inlet
            Y = sub[col_name].values

            # Fit (Highest to Lowest)
            coeffs_high_low = np.polyfit(X, Y, deg)
            coeffs = coeffs_high_low[::-1]  # Flip to Ascending

            # Ensure domain continuity: use stitch_pressure as max if available
            p_min_data = sub['Pressure'].min()
            p_max_data = sub['Pressure'].max()
            effective_p_max = stitch_pressure if stitch_pressure is not None else p_max_data

            seg = PolynomialSegment(coeffs, p_min_data, effective_p_max, self.P_inlet)

            # 2. Stitching Logic (Shift to match previous High-P segment)
            if target_val_at_stitch is not None:
                # We stitch at the EXACT pressure where the previous high-P segment ended
                current_val = seg.evaluate(effective_p_max)

                # Calculate Shift
                shift = target_val_at_stitch - current_val

                # Apply Shift to constant term (a0)
                seg.coeffs[0] += shift
                print(f"  > {name} (Deg {deg}): Shifted {shift:.4e} to match @ {effective_p_max / 1e5:.3f} bar")
            else:
                print(f"  > {name} (Deg {deg}): Anchor Region (No Shift)")

            # 3. Prepare for Next Segment
            stitch_pressure = seg.p_min
            target_val_at_stitch = seg.evaluate(stitch_pressure)

            temp_segments[name] = seg

        # Return sorted Low -> High
        final_list = []
        for name in ['3-Equilibrium', '2-NEB', '1-SinglePhase']:
            if name in temp_segments:
                final_list.append(temp_segments[name])

        return final_list

# ==========================================
# 3. EXPORTER CLASS (Robust Safeguards)
# ==========================================
class CfxExporter:
    """
    Translates fitted PolynomialSegments into solver-ready syntax (ANSYS CFX CEL).
    
    The exporter handles:
    1. Syntax conversion for Horner's Method evaluation.
    2. Recursive IF-statement wrapping for piecewise selection.
    3. Implementation of physical safeguards (Exponential Decay and Linear Extension).
    """
    def __init__(self, fitter):
        """
        Parameters:
            fitter (NebFitter): An instance of the fitter containing the source data and logic.
        """
        self.fitter = fitter

    def _horner_to_string(self, seg):
        """
        Converts a PolynomialSegment into a CEL-compatible Horner's Method string.
        Optimization: Skips negligible coefficients to reduce string length.
        """
        p_str = f"(Pressure / {seg.p_inlet:.2f} [Pa])"
        expr = f"{seg.coeffs[-1]:.8e}"
        for c in reversed(seg.coeffs[:-1]):
            if abs(c) < 1e-16:
                expr = f"{p_str} * ({expr})"
            else:
                expr = f"{c:.8e} + {p_str} * ({expr})"
        return expr

    def generate_cel_safeguarded(self, col_name, var_name, degree=8):
        """
        Generates a complete, robust CFX Expression for a given property.
        
        The resulting CEL expression structure is:
        IF(P < P_min, 
           Exponential_Decay, 
           IF(P > P_max, 
              Linear_Extension, 
              Piecewise_Polynomials
           )
        )
        
        This ensures the property is always defined, physical, and continuous.
        
        Parameters:
            col_name (str): The source column in the fitter's dataframe.
            var_name (str): The desired name for the CFX variable (e.g., 'rhoNEB').
            degree (int/dict): The degree(s) to use for the fit.
        """

        # 1. Get Core Stitched Segments
        segments = self.fitter.fit_variable(col_name, degree)
        if not segments: return "0.0"

        # --- A. Build Core Polynomials ---
        # Start with High P (Single Phase) default
        core_expr = f"({self._horner_to_string(segments[-1])})"

        # Wrap backwards (NEB, then Eq)
        for i in range(len(segments) - 2, -1, -1):
            seg = segments[i]
            # If P < Seg_Max, use Seg, else use Previous
            core_expr = f"if(Pressure < {seg.p_max:.2f} [Pa], {self._horner_to_string(seg)}, {core_expr})"

        # --- B. Lower Safeguard (Exponential) ---
        # Attached to the LOWEST pressure point of the stitched curve
        s_eq = segments[0]
        P_min = s_eq.p_min
        y_min = s_eq.evaluate(P_min)
        dy_min = s_eq.derivative(P_min)

        # ROBUST SLOPE ESTIMATION
        # If the analytical slope at the edge is bad (flat/negative), peek 5% inside the domain
        if dy_min <= 1e-12:
            print(
                f"{Fore.YELLOW}  > Warning: Flat/Neg slope at P_min ({dy_min:.2e}). Peeking 5% inside...{Style.RESET_ALL}")
            P_peek = P_min + (s_eq.p_max - P_min) * 0.05
            dy_min = (s_eq.evaluate(P_peek) - y_min) / (P_peek - P_min)

        # Determine Decay Constant (Continuity of Slope)
        if y_min > 0 and dy_min > 0:
            k = y_min / dy_min
            lower_expr = f"{y_min:.6e} * exp((Pressure - {P_min:.2f} [Pa]) / {k:.6e} [Pa])"
        else:
            lower_expr = f"{y_min:.6e}"  # Constant fallback

        # --- C. Upper Safeguard (Linear) ---
        # Attached to the HIGHEST pressure point (Inlet)
        s_single = segments[-1]
        P_max = s_single.p_max
        y_max = s_single.evaluate(P_max)
        dy_max = s_single.derivative(P_max)

        # Physical constraint: Density slope must be positive
        if "rho" in var_name.lower() and dy_max < 0:
            print(
                f"{Fore.YELLOW}  > Warning: Negative density slope at Inlet. Forcing const extrapolation.{Style.RESET_ALL}")
            dy_max = 0.0

        upper_expr = f"{y_max:.6e} + {dy_max:.6e} * (Pressure - {P_max:.2f} [Pa])"

        # --- D. Final Global Wrap ---
        # If P < P_min -> Exponential
        # Else If P > P_max -> Linear
        # Else -> Core Stitched Polynomials
        full_expr = (
            f"if(Pressure < {P_min:.2f} [Pa], \n"
            f"    {lower_expr}, \n"
            f"    if(Pressure > {P_max:.2f} [Pa], \n"
            f"        {upper_expr}, \n"
            f"        {core_expr}\n"
            f"    )\n"
            f")"
        )

        print(f"{Fore.GREEN}✓ CEL Generated for {var_name}{Style.RESET_ALL}")
        return full_expr


# ==========================================
# 4. VERIFICATION ROUTINE
# ==========================================
def check_global_continuity(fitter, col_name, var_name, degree=8):
    """
    Visualizes the entire pressure range:
    Exponential (Low) -> Eq -> NEB -> Single -> Linear (High)
    """
    print(f"Checking Global Continuity for {var_name} (Degree {degree})...")

    # 1. Fit & Export Logic (Replicated for Plotting)
    segments = fitter.fit_variable(col_name, degree)

    # Lower Safeguard Params
    s0 = segments[0]
    y_min = s0.evaluate(s0.p_min);
    dy_min = s0.derivative(s0.p_min)
    if dy_min <= 1e-12:
        P_peek = s0.p_min + (s0.p_max - s0.p_min) * 0.05
        dy_min = (s0.evaluate(P_peek) - y_min) / (P_peek - s0.p_min)
    k_decay = y_min / dy_min if (y_min > 0 and dy_min > 0) else 1e9

    # Upper Safeguard Params
    sN = segments[-1]
    y_max = sN.evaluate(sN.p_max);
    dy_max = sN.derivative(sN.p_max)
    if "rho" in var_name.lower() and dy_max < 0: dy_max = 0.0

    # 2. DIAGNOSTICS TABLE
    print(f"\n{Fore.CYAN}--- Global Continuity Diagnostics ({var_name}) ---{Style.RESET_ALL}")
    print(
        f"{'Point':<18} | {'Pressure [bar]':<15} | {'Left Value':<15} | {'Right Value':<15} | {'Left Slope':<12} | {'Right Slope':<12}")
    print("-" * 115)

    # A. Lower Safeguard Join
    # Left = Exp Extrap, Right = Polynomial Segment 0
    P_min_sim = s0.p_min
    l_val_low = y_min
    l_slope_low = dy_min if (y_min > 0 and dy_min > 0) else 0.0
    print(
        f"{'Lower Safeguard':<18} | {P_min_sim / 1e5:<15.4f} | {l_val_low:<15.5e} | {y_min:<15.5e} | {l_slope_low:<12.5e} | {dy_min:<12.5e}")

    # B. Internal Stitch Points
    for i in range(len(segments) - 1):
        s_left = segments[i]
        s_right = segments[i + 1]
        P_join = s_left.p_max

        v_l = s_left.evaluate(P_join)
        v_r = s_right.evaluate(P_join)
        d_l = s_left.derivative(P_join)
        d_r = s_right.derivative(P_join)

        label = f"Stitch {i + 1}"
        if len(segments) == 3:
            label = "Spinodal" if i == 0 else "Nucleation"

        print(f"{label:<18} | {P_join / 1e5:<15.4f} | {v_l:<15.5e} | {v_r:<15.5e} | {d_l:<12.5e} | {d_r:<12.5e}")

    # C. Upper Safeguard Join
    # Left = Polynomial Segment N, Right = Linear Extrap
    P_max_sim = sN.p_max
    print(
        f"{'Upper Safeguard':<18} | {P_max_sim / 1e5:<15.4f} | {y_max:<15.5e} | {y_max:<15.5e} | {dy_max:<12.5e} | {dy_max:<12.5e}")

    # 3. Plotting Range (0 Pa to 1.5x Inlet)
    P_eval = np.linspace(0, P_max_sim * 1.5, 1000)
    Y_eval = []

    for p in P_eval:
        if p < P_min_sim:
            val = y_min * np.exp((p - P_min_sim) / k_decay)
        elif p > P_max_sim:
            val = y_max + dy_max * (p - P_max_sim)
        else:
            # Core Stitched Polynomials
            val = np.nan
            for seg in segments:
                if seg.p_min <= p <= seg.p_max + 1.0:  # Tolerance
                    val = seg.evaluate(p)
                    break
            if np.isnan(val): val = segments[-1].evaluate(p)
        Y_eval.append(val)

    # 3. Plot
    plt.figure(figsize=(10, 6))

    # Continuous Model
    plt.plot(P_eval / 1e5, Y_eval, 'b-', linewidth=2, label='Final Continuous Model')

    # Raw Data points
    # plt.plot(fitter.df['Pressure']/1e5, fitter.df[col_name], 'k.', alpha=0.1, label='Raw Data')

    # Boundaries
    plt.axvline(P_min_sim / 1e5, color='r', ls=':', lw=2, label='P_min (Start Exp Decay)')
    plt.axvline(P_max_sim / 1e5, color='g', ls=':', lw=2, label='P_max (Start Lin Extrap)')

    plt.xlabel("Pressure [bar]")
    plt.ylabel(col_name)
    plt.title(f"Global Continuity Check: {var_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Invert X axis for standard expansion view
    plt.gca().invert_xaxis()
    plt.show()

def verify_fit(df, P_inlet, col_name, degree=6):
    """Visual check of the polynomial fit."""
    subset = df[df['Region'] == '2-NEB']
    X = subset['Pressure'].values / P_inlet
    Y = subset[col_name].values

    coeffs = np.polyfit(X, Y, degree)
    Y_pred = np.polyval(coeffs, X)

    plt.figure(figsize=(8, 5))
    plt.plot(subset['Pressure'], Y, 'k-', linewidth=3, alpha=0.5, label='Original Data')
    plt.plot(subset['Pressure'], Y_pred, 'r--', label=f'Poly Fit (Deg {degree})')
    plt.title(f"Fit Verification: {col_name} (NEB Region)")
    plt.legend()
    plt.show()


def plot_extended_verification(df_neb, df_ref=None, fluid_name="Fluid"):
    """
    Generates a comprehensive 6-panel dashboard visualizing:
    1. Density (Macroscopic State)
    2. Speed of Sound (Acoustic State)
    3. Viscosity (Transport Property)
    4. Mass Fractions (Composition Evolution)
    5. Volume Fractions (Topology Evolution)
    6. Closure Variables (Gamma & Pi)
    """
    # Create Layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    # Unpack axes for clarity
    ax_rho, ax_sos, ax_mu = axes[0]  # Top Row: Thermodynamics
    ax_mass, ax_vol, ax_close = axes[1]  # Bottom Row: Internal Physics

    # X-axis data (Pressure in bar)
    P_bar = df_neb["Pressure"] / 1e5

    # ------------------------------------------------------------
    # 1. DENSITY (The Primary Variable)
    # ------------------------------------------------------------
    if df_ref is not None:
        ax_rho.plot(df_ref["Pressure"] / 1e5, df_ref["Rho_Mixture"],
                    'k--', linewidth=2, alpha=0.6, label="Ref (HEM)")

    ax_rho.plot(P_bar, df_neb["Rho_Mixture"], 'b-', linewidth=3, label="NEB Mixture")
    ax_rho.set_title(r"1. Density ($\rho_{mix}$)")
    ax_rho.set_ylabel(r"Density [kg/m$^3$]")
    ax_rho.legend(loc='best')

    # ------------------------------------------------------------
    # 2. SPEED OF SOUND (The Critical Stability Check)
    # ------------------------------------------------------------
    if df_ref is not None:
        ax_sos.plot(df_ref["Pressure"] / 1e5, df_ref["SoS_Mixture"],
                    'k--', linewidth=2, alpha=0.6, label="Ref (HEM)")

    ax_sos.plot(P_bar, df_neb["SoS_Mixture"], 'r-', linewidth=2.5, label="NEB Sound Speed")
    ax_sos.set_title(r"2. Speed of Sound ($c_{DEM}$)")
    ax_sos.set_ylabel("Speed of Sound [m/s]")
    ax_sos.set_yscale('log')  # Crucial to see the drop
    ax_sos.legend(loc='best')

    # ------------------------------------------------------------
    # 3. VISCOSITY (Transport)
    # ------------------------------------------------------------
    if df_ref is not None:
        ax_mu.plot(df_ref["Pressure"] / 1e5, df_ref["Mu_Mixture"],
                   'k--', linewidth=2, alpha=0.6, label="Ref (HEM)")

    ax_mu.plot(P_bar, df_neb["Mu_Mixture"], 'g-', linewidth=2.5, label="NEB Viscosity")
    ax_mu.set_title(r"3. Viscosity ($\mu_{mix}$)")
    ax_mu.set_ylabel("Viscosity [Pa·s]")
    ax_mu.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax_mu.legend(loc='best')

    # ------------------------------------------------------------
    # 4. MASS FRACTIONS (Composition X)
    # ------------------------------------------------------------
    # We use fillna(0) because single-phase regions might have NaNs depending on construction
    X_meta = df_neb.get("X_l_meta", np.zeros_like(P_bar)).fillna(0)
    X_sat = df_neb.get("X_l_sat", np.zeros_like(P_bar)).fillna(0)
    X_vap = df_neb.get("X_v", np.zeros_like(P_bar)).fillna(0)

    ax_mass.stackplot(P_bar, X_meta, X_sat, X_vap,
                      labels=[r"$X_{meta}$ (Metastable Liq)", r"$X_{sat}$ (Sat Liq)", r"$X_{vap}$ (Vapor)"],
                      colors=['#2ca02c', '#1f77b4', '#d62728'], alpha=0.7)

    ax_mass.set_title("4. Mass Fractions (Composition)")
    ax_mass.set_ylabel("Mass Fraction [-]")
    ax_mass.set_ylim(0, 1.0)
    ax_mass.legend(loc='lower left', fontsize=9)

    # ------------------------------------------------------------
    # 5. VOLUME FRACTIONS (Topology Epsilon)
    # ------------------------------------------------------------
    # These drive the viscosity and density mixing
    eps_meta = df_neb.get("eps_l_meta", np.zeros_like(P_bar)).fillna(0)
    eps_sat = df_neb.get("eps_l_sat", np.zeros_like(P_bar)).fillna(0)
    eps_vap = df_neb.get("eps_v", np.zeros_like(P_bar)).fillna(0)

    ax_vol.stackplot(P_bar, eps_meta, eps_sat, eps_vap,
                     labels=[r"$\epsilon_{meta}$", r"$\epsilon_{sat}$", r"$\epsilon_{vap}$"],
                     colors=['#2ca02c', '#1f77b4', '#d62728'], alpha=0.7)

    ax_vol.set_title(r"5. Volume Fractions ($\epsilon$)")
    ax_vol.set_ylabel("Volume Fraction [-]")
    ax_vol.set_ylim(0, 1.0)
    ax_vol.legend(loc='upper left', fontsize=9)

    # ------------------------------------------------------------
    # 6. CLOSURE (Gamma & Pi)
    # ------------------------------------------------------------
    # Visualize the "Driver" of the model
    Gamma = df_neb.get("Gamma", np.zeros_like(P_bar))
    Pi = df_neb.get("Pi", np.zeros_like(P_bar))

    ax_close.plot(P_bar, Gamma, color='purple', linewidth=3, label=r"$\gamma$ (Relaxation)")
    ax_close.plot(P_bar, Pi, color='orange', linestyle='--', linewidth=2, label=r"$\Pi$ (Norm. Pressure)")

    ax_close.set_title(r"6. Closure Variables ($\gamma, \Pi$)")
    ax_close.set_ylabel("Value [-]")
    ax_close.set_ylim(-0.05, 1.05)
    ax_close.legend(loc='best')

    # ------------------------------------------------------------
    # GLOBAL FORMATTING
    # ------------------------------------------------------------
    # Extract Limits for Vertical Lines
    neb_subset = df_neb[df_neb["Region"] == "2-NEB"]
    if not neb_subset.empty:
        p_nuc = neb_subset["Pressure"].max() / 1e5
        p_spin = neb_subset["Pressure"].min() / 1e5

        # Add lines to all subplots
        for ax in axes.flat:
            ax.axvline(p_nuc, color='k', linestyle=':', linewidth=1.5)
            ax.axvline(p_spin, color='r', linestyle=':', linewidth=1.5)

    # Invert X-axis (High P -> Low P) and Add Grid
    for ax in axes.flat:
        ax.set_xlabel("Pressure [bar]")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.minorticks_on()

    plt.suptitle(f"Extended Verification Dashboard: {fluid_name}", fontsize=18, weight='bold')
    # plt.show()
    # return fig

def verify_safeguards(fitter, col_name, degree=8):
    """Plots the polynomial fit AND the extrapolation safeguards."""
    # 1. Fit Data
    segments = fitter.fit_variable(col_name, degree)

    # 2. Get Safeguard Constants
    # Lower (Exponential)
    s0 = segments[0]
    P_min, y_min = s0.p_min, s0.evaluate(s0.p_min)
    dy_min = s0.derivative(s0.p_min)

    # Robust Slope for P_min (consistent with Exporter)
    if dy_min <= 1e-12:
        print(f"{Fore.YELLOW}  > Warning: Flat/Neg slope at P_min ({dy_min:.2e}). Peeking 5% inside...{Style.RESET_ALL}")
        P_peek = P_min + (s0.p_max - P_min) * 0.05
        dy_min = (s0.evaluate(P_peek) - y_min) / (P_peek - P_min)

    # Upper (Linear)
    s1 = segments[-1]
    P_max, y_max = s1.p_max, s1.evaluate(s1.p_max)
    dy_max = s1.derivative(s1.p_max)

    # Robust Slope for P_max (consistent with Exporter)
    if "rho" in col_name.lower() and dy_max < 0:
        print(f"{Fore.YELLOW}  > Warning: Negative density slope at P_max. Forcing const extrapolation.{Style.RESET_ALL}")
        dy_max = 0.0

    # 2.5 DIAGNOSTICS TABLE
    print(f"\n{Fore.CYAN}--- Safeguard Boundary Diagnostics ({col_name}) ---{Style.RESET_ALL}")
    print(f"{'Boundary':<15} | {'Pressure [bar]':<15} | {'Left Value':<15} | {'Right Value':<15} | {'Left Slope':<12} | {'Right Slope':<12}")
    print("-" * 110)

    # Lower Boundary: Left=Extrap, Right=Poly
    k = y_min / dy_min if (y_min > 0 and dy_min > 0) else None
    l_val_low, r_val_low = y_min, y_min
    l_slope_low = dy_min if k is not None else 0.0
    r_slope_low = dy_min
    print(f"{'Lower (P_min)':<15} | {P_min/1e5:<15.4f} | {l_val_low:<15.5e} | {r_val_low:<15.5e} | {l_slope_low:<12.5e} | {r_slope_low:<12.5e}")

    # Upper Boundary: Left=Poly, Right=Extrap
    l_val_high, r_val_high = y_max, y_max
    l_slope_high = dy_max
    r_slope_high = dy_max
    print(f"{'Upper (P_max)':<15} | {P_max/1e5:<15.4f} | {l_val_high:<15.5e} | {r_val_high:<15.5e} | {l_slope_high:<12.5e} | {r_slope_high:<12.5e}")

    # 3. Generate Evaluation Arrays
    # Core Range
    P_core = fitter.df['Pressure'].values

    # Lower Extrapolation (0 to P_min)
    P_low = np.linspace(0, P_min, 100)
    if k is not None:
        Y_low = y_min * np.exp((P_low - P_min) / k)
    else:
        Y_low = np.full_like(P_low, y_min)

    # Upper Extrapolation (P_max to 1.5*P_max)
    P_high = np.linspace(P_max, P_max * 1.5, 100)
    Y_high = y_max + dy_max * (P_high - P_max)

    # 4. Plot
    plt.figure(figsize=(10, 6))

    # Plot Safeguards
    plt.plot(P_low / 1e5, Y_low, 'r--', lw=2, label='Lower Safeguard (Exp)')
    plt.plot(P_high / 1e5, Y_high, 'g--', lw=2, label='Upper Safeguard (Lin)')

    # Plot Core Data
    plt.plot(P_core / 1e5, fitter.df[col_name], 'k-', lw=3, alpha=0.5, label='Simulation Data')

    # Boundaries
    plt.axvline(P_min / 1e5, color='k', ls=':', label='Simulation Bounds')
    plt.axvline(P_max / 1e5, color='k', ls=':')

    plt.xlabel('Pressure [bar]')
    plt.ylabel(col_name)
    plt.title(f'Safeguard Verification: {col_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def verify_and_export_final(fitter, col_name, var_name, degree=8):
    """
    1. Fits the data with C0 Continuity.
    2. Prints Diagnostics at all Stitch Points (Value & Slope).
    3. Plots the Full Continuity (Fit + Safeguards).
    4. Returns CEL string.
    """
    print(f"\n{Fore.MAGENTA}=== FINAL VERIFICATION: {var_name} ==={Style.RESET_ALL}")

    # 1. Fit
    segments = fitter.fit_variable(col_name, degree)

    # 2. DIAGNOSTICS TABLE
    print(f"\n{Fore.CYAN}--- Stitch Point Diagnostics ---{Style.RESET_ALL}")
    print(
        f"{'Point':<15} | {'Pressure [Pa]':<15} | {'Left Value':<15} | {'Right Value':<15} | {'Gap (Abs)':<12} | {'Left Slope':<12} | {'Right Slope':<12}")
    print("-" * 110)

    # List of critical pressures to check
    # Start of Sim, Spinodal (Boundary 1), Nucleation (Boundary 2), End of Sim
    check_points = [
        ("Start (P_min)", segments[0].p_min),
        ("Spinodal", segments[0].p_max),  # Boundary between Eq and NEB
        ("Nucleation", segments[1].p_max),  # Boundary between NEB and Single
        ("End (P_max)", segments[-1].p_max)
    ]

    for label, P in check_points:
        # Find which segments meet here
        # For boundaries, we compare segment i (Left) and segment i+1 (Right)

        val_left, val_right = np.nan, np.nan
        slope_left, slope_right = np.nan, np.nan

        # Check all segments to see who owns this boundary
        for i, seg in enumerate(segments):
            # Check if P is this segment's Max (Left side of join)
            if np.isclose(P, seg.p_max):
                val_left = seg.evaluate(P)
                slope_left = seg.derivative(P)

            # Check if P is this segment's Min (Right side of join)
            if np.isclose(P, seg.p_min):
                val_right = seg.evaluate(P)
                slope_right = seg.derivative(P)

        # Special handling for endpoints (Start/End only have one side)
        if label == "Start (P_min)":
            val_right = segments[0].evaluate(P)  # It's the start of seg 0
            slope_right = segments[0].derivative(P)
            print(
                f"{label:<15} | {P:<15.1f} | {'---':<15} | {val_right:<15.4e} | {'---':<12} | {'---':<12} | {slope_right:<12.2e}")

        elif label == "End (P_max)":
            val_left = segments[-1].evaluate(P)
            slope_left = segments[-1].derivative(P)
            print(
                f"{label:<15} | {P:<15.1f} | {val_left:<15.4e} | {'---':<15} | {'---':<12} | {slope_left:<12.2e} | {'---':<12}")

        else:
            # Internal Stitch Point
            gap = abs(val_left - val_right)
            # Color code the gap
            gap_str = f"{gap:.2e}"
            if gap > 1e-4: gap_str = f"{Fore.RED}{gap_str}{Style.RESET_ALL}"

            print(
                f"{label:<15} | {P:<15.1f} | {val_left:<15.4e} | {val_right:<15.4e} | {gap_str:<12} | {slope_left:<12.2e} | {slope_right:<12.2e}")

    # 3. Generate String
    exporter = CfxExporter(fitter)
    cel_expr = exporter.generate_cel_safeguarded(col_name, var_name, degree)

    # 4. VISUALIZATION (With Safeguards)
    P_min_sim = segments[0].p_min
    P_max_sim = segments[-1].p_max

    # Generate Viz Range (Simulation +/- 20%)
    P_eval = np.linspace(P_min_sim * 0.5, P_max_sim * 1.2, 500)
    Y_eval = []

    # Replicate Exporter Logic for Plotting
    # Get Safeguard Params
    s0 = segments[0]
    y_min = s0.evaluate(s0.p_min);
    dy_min = s0.derivative(s0.p_min)
    k_decay = y_min / dy_min if (y_min > 0 and dy_min > 1e-12) else 1.0

    s_last = segments[-1]
    y_max = s_last.evaluate(s_last.p_max);
    dy_max = s_last.derivative(s_last.p_max)
    if "rho" in var_name.lower() and dy_max < 0: dy_max = 0.0  # Match logic

    for p in P_eval:
        if p < P_min_sim:
            val = y_min * np.exp((p - P_min_sim) / k_decay) if y_min > 0 else y_min
        elif p > P_max_sim:
            val = y_max + dy_max * (p - P_max_sim)
        else:
            # Core
            val = np.nan
            for seg in segments:
                if seg.p_min <= p <= seg.p_max + 1.0:  # Tolerance
                    val = seg.evaluate(p)
                    break
            if np.isnan(val): val = segments[-1].evaluate(p)
        Y_eval.append(val)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(P_eval / 1e5, Y_eval, 'b-', linewidth=2, label='Final Model')
    plt.axvline(P_min_sim / 1e5, color='r', ls=':', label='$P_{min}$')
    plt.axvline(P_max_sim / 1e5, color='g', ls=':', label='$P_{max}$')

    # Add raw data for comparison
    plt.plot(fitter.df['Pressure'] / 1e5, fitter.df[col_name], 'k--', alpha=0.3, label='Raw Data')

    plt.xlabel("Pressure [bar]")
    plt.ylabel(col_name)
    plt.title(f"Continuity & Safeguard Check: {var_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return cel_expr