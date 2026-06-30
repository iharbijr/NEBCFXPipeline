# NEB Closure Calibration — Single-Case Plan

This note sets out how to calibrate the Non-Equilibrium Barotropic (NEB) closure
parameters **N** (delay index) and **K** (relaxation rate) for a single water
flashing case, against experimental nozzle data. It explains what each parameter
does physically, the order in which to tune them, why the calibration cannot be
taken directly from a 1D model, and the design of experiments (DoE) used to map
the response surface.

---

## 1. Background: what is being calibrated

The NEB closure replaces a finite-rate flashing model with a single-valued
barotropic material law `ρ(P)` supplied to the CFD solver. The flow expands
along the inlet isentrope; below the nucleation pressure `P_nuc` the liquid
enters a metastable, non-equilibrium state and relaxes toward equilibrium as it
depressurises. The degree of relaxation is governed by a stretched-exponential
closure on the normalized pressure `Π`:

```
γ(Π) = [1 − exp(−K·(1−Π)^N)] / [1 − exp(−K)]
```

where `γ` is the **equilibrated mass fraction** of the mixture, and for water
`Π = P / P_nuc` (inception-anchored, since the liquid spinodal is at negative
pressure). The two free parameters are:

- **N — the delay index** (penetration / nucleation delay),
- **K — the relaxation rate** (strength of the drive toward equilibrium).

Calibration means finding the `(N, K)` pair for which the CFD reproduces the
measured **axial pressure profile**, **void-fraction distribution**, and
**choked mass flow rate** for this case's boundary conditions.

---

## 2. Physical roles of N and K

### N — delay index (tune first)

`N` controls how deep into the metastable region the liquid penetrates before
relaxation gathers pace. A higher `N` delays the onset of significant
equilibration: the liquid stays denser and the mixture remains closer to the
single-phase (frozen) branch further down the nozzle.

**Consequence for mass flow.** Because the choked mass flow is set at the throat
by the mixture density and sound speed, **a higher N — by keeping the fluid
denser for longer — admits a larger mass flow rate before choking.** N is
therefore the primary lever for matching the experimental choked mass flow rate.

### K — relaxation rate (tune second)

`K` sets how sharply the mixture relaxes toward equilibrium once nucleation is
under way. Large `K` drives the fluid to equilibrium almost immediately below
`P_nuc` (approaching the Homogeneous Equilibrium Model, HEM); small `K` spreads
the relaxation out toward full delay (approaching the frozen-liquid limit).

`K` is calibrated **after** `N`, because the two are not independent in their
effect on the solution: `N` fixes *where* and *how much* the mixture departs
from equilibrium, and `K` then shapes the *path* it takes back. Tuning `K`
before `N` is set would chase a moving target.

---

## 3. Why the calibration is not inherited from the 1D model

A prior 1D Delayed Equilibrium Model (DEM) study produced a law for `N` as a
function of the inlet inception ratio, and fixed `K ≈ 100`. That law is a useful
**prior** — a physically motivated starting point — but it is **not** the final
calibration, for a specific reason that matters here:

> **Wall friction is no longer a 1D closure.** In the 1D DEM, wall shear stress
> is imposed through a correlation. In the 3D CFD solver, wall stress is
> resolved from the **nearest-cell average density and velocity** — it emerges
> from the field, not from a correlation. Because the relaxation path (and hence
> the near-wall density and velocity) depends on `K`, the value of `K` that
> reproduces the experiment in CFD is **expected to differ** from the 1D value,
> and this is exactly what the calibration must check.

In short: `N` from the DEM law is a strong prior; `K` must be re-examined in the
CFD because the physics that constrains it (wall stress) is represented
differently. The calibration treats `N` as "start from the law, refine against
mass flow" and `K` as "genuinely re-calibrate against the experiment."

---

## 4. Calibration procedure (single case)

Hold the boundary conditions fixed (this case's inlet stagnation state and
outlet pressure) throughout.

**Step 1 — Prior.** Evaluate the DEM law to get the starting `N₀` for this
case's inception ratio, with `K = 100`. Run it. This is the law-prior point.

**Step 2 — Tune N against mass flow.** Vary `N` around `N₀` (the `N_CORRECTION`
knob in the master scales the law value) and read the choked mass flow rate from
the solver's monitor point. Because higher `N` admits more mass flow, this is a
near-monotone one-parameter search: bracket the experimental mass flow and home
in on the `N` that matches it.

**Step 3 — Calibrate K against the profiles.** With `N` fixed at its
mass-flow-matched value, vary `K` and compare the **axial pressure profile** and
**void-fraction distribution** to the experiment. This is where the
3D-vs-1D wall-stress difference shows up: the best-fit `K` here is the genuine
CFD-calibrated value, not necessarily 100.

**Step 4 — Check coupling.** Re-confirm that the `K` adjustment in Step 3 did not
move the choked mass flow off target. If it did, a short second pass on `N`
closes the loop. In practice the coupling is weak once `N` sets the throat
behaviour, but it is worth one verification.

---

## 5. Design of Experiments (DoE)

A one-at-a-time search (Steps 2–3) is efficient for homing in, but a **2D DoE
over `(N, K)`** is run alongside it to *see the whole response*, not just the
optimum. For the same boundary conditions, sweep a grid of `N` and `K` values
and record the **choked mass flow rate** (monitor point) for each.

### What to plot

- **Primary:** choked mass flow rate as a surface / heatmap over the `(N, K)`
  plane. This is the headline calibration map: the experimental mass flow is a
  contour on this surface, and every `(N, K)` on that contour matches the mass
  flow (with `K` then resolved by the profile comparison of Step 3).
- **Secondary:** the same surface for a profile-error metric (e.g. RMS pressure
  deviation), to see where on the mass-flow contour the profiles also agree.

### Expected shape

The surface should be a **continuous mapping between two physical limits**:

- as the closure approaches **HEM** (rapid, full equilibration — large `K`,
  low effective delay), the mass flow tends to the equilibrium-limited value;
- as it approaches the **frozen-liquid** limit (strong delay, little
  relaxation — high `N`, and `K` increasingly immaterial), the mass flow tends
  to the metastable/frozen value.

A smooth, monotone-in-`N` surface spanning these limits is the expected result,
and confirms the closure behaves physically across the parameter space. **The
visualisation is the deliverable** — it both locates the calibration and gives
confidence that the model interpolates sensibly between HEM and frozen flow,
with no spurious folds or discontinuities.

---

## 6. Running the DoE with the pipeline

The water master (`Master_Water.py`) expresses this grid directly. For a single
case, reduce the operating-point list to the one case and widen the parameter
lists:

```python
# Single case under study
RAW_POINTS = [(80, 559.8243936, "P80_s08")]   # this case only

# DoE grid: N via the law-value correction, K swept directly
K_LIST            = [30, 60, 100, 150, 300]    # relaxation rate
N_CORRECTION_LIST = [0.8, 0.9, 1.0, 1.1, 1.2]  # scales the law-predicted N
```

This produces `len(K_LIST) × len(N_CORRECTION_LIST)` cluster-ready case folders
(here 5 × 5 = 25), each named `Sim_{label}_K{k}_N{n}`, with its own
`run_case.sh`. Submit them in parallel; after the runs, read each monitor-point
mass flow and assemble the `(N, K)` → mass-flow surface.

> Note on `N`: it is law-driven, so the swept quantity is the **correction
> factor** on the law-predicted `N`, not `N` itself. The actual `N` per case is
> `N_law(Π_nuc) × correction` and is recorded in the folder name and the case
> metadata.

---

## 7. Summary

| Parameter | Physical meaning | Primary effect | Calibration order | Source |
|-----------|------------------|----------------|-------------------|--------|
| **N** | nucleation delay / penetration | sets choked mass flow (higher N → more flow) | **first** | DEM law prior, refined vs mass flow |
| **K** | relaxation rate toward equilibrium | shapes the relaxation path / profiles | **second** | re-calibrated in CFD (wall stress is 3D-resolved) |

1. Start from the DEM law prior `(N₀, K=100)`.
2. Tune **N** against the experimental **choked mass flow rate**.
3. Calibrate **K** against the **pressure and void-fraction profiles** — and
   expect it to differ from the 1D value, because wall stress is resolved from
   near-wall field quantities, not a 1D correlation.
4. Run a **2D `(N, K)` DoE** and plot the mass-flow surface: a continuous map
   from HEM to frozen liquid, used both to locate the calibration and to verify
   physical behaviour.