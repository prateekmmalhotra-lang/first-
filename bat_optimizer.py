"""
Batsimulator — BESS Energy Arbitrage MILP Optimizer
====================================================
Solves the full-day (up to 96 block) battery dispatch as a Mixed Integer
Linear Program using Linopy + HiGHS.

There is NO PPA / FDRE obligation anywhere in this model. The plant may be:
  - BESS only (no renewable generation), or
  - Hybrid (Solar and/or Wind + BESS) with renewable generation available
    purely as a chargeable / sellable energy source.

The objective is pure merchant arbitrage: maximise
    (renewable direct-sale revenue) + (battery discharge revenue)
    - (grid charging cost) - (degradation cost on throughput)

Renewable generation, when present, is free "fuel" — the optimizer decides
block-by-block whether to sell it immediately or route it through the
battery for a better price later. There is no contractual delivery target,
so there is no shortfall/penalty term of any kind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import linopy
    LINOPY_AVAILABLE = True
except ImportError:
    LINOPY_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────
BLOCK_HOURS = 0.25          # hours per 15-minute block
BLOCK_FACTOR = 250.0        # MW * 0.25h * 1000 kWh/MWh -> kWh;  used for ₹ revenue (price in ₹/kWh)

DEFAULT_BATT_ENERGY = 100.0     # MWh
DEFAULT_BATT_POWER = 25.0       # MW
DEFAULT_CHARGE_EFF = 0.95
DEFAULT_DISCHARGE_EFF = 0.95
DEFAULT_DEGRADATION_COST = 0.15  # ₹ / kWh throughput (charge + discharge)


def _extract_objective(m) -> float | None:
    """Robustly read the post-solve objective value from a Linopy model."""
    def _try_float(val):
        if val is None:
            return None
        try:
            f = float(val)
            return None if np.isnan(f) else f
        except (TypeError, ValueError):
            return None

    obj = getattr(m, "objective", None)
    f = _try_float(getattr(obj, "value", None))
    if f is not None:
        return round(-f, 2)
    f = _try_float(getattr(obj, "_value", None))
    if f is not None:
        return round(-f, 2)
    try:
        sol = getattr(m, "solution", None)
        if sol is not None and obj is not None:
            expr = getattr(obj, "expression", None) or getattr(obj, "flat", None)
            if expr is not None:
                coeffs = expr.coeffs.values.ravel().astype(float)
                labels = expr.vars.values.ravel().astype(int)
                flat_sol = sol.to_array().values.ravel().astype(float)
                obj_val = float(np.dot(coeffs, flat_sol[labels]))
                f = _try_float(obj_val)
                if f is not None:
                    return round(-f, 2)
    except Exception:
        pass
    return None


def solve_bess_milp(
    df: pd.DataFrame,
    soc_initial: float,
    plant_config: dict,
    terminal_soc: float | None = None,
    allow_grid_charging: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Solve the day-ahead BESS arbitrage MILP for all T blocks simultaneously.

    Args:
        df: input DataFrame with columns DAM_Price, GDAM_Price, RTM_Price_DA
            and (if hybrid) SY_DA, WY_DA. Must have T <= 96 rows.
        soc_initial: battery SOC at the start of the horizon (MWh).
        plant_config: dict with batt_energy, batt_power, soc_min_pct,
            soc_max_pct, charge_eff, discharge_eff, degradation_cost,
            plant_type, solar_cap, wind_cap, and optional derate array
            under plant_config["_derate"] (per-block multiplier 0-1).
        terminal_soc: minimum SOC at end of horizon (defaults to soc_initial).
        allow_grid_charging: if False, battery can only be charged from
            renewable generation (never from the market).

    Returns:
        (result_df, solver_info)
    """
    if not LINOPY_AVAILABLE:
        raise ImportError(
            "linopy is not installed.\nInstall with:  pip install linopy highspy"
        )

    T = len(df)
    cfg = plant_config or {}
    batt_energy = float(cfg.get("batt_energy", DEFAULT_BATT_ENERGY))
    batt_power = float(cfg.get("batt_power", DEFAULT_BATT_POWER))
    soc_min = batt_energy * float(cfg.get("soc_min_pct", 5)) / 100.0
    soc_max = batt_energy * float(cfg.get("soc_max_pct", 95)) / 100.0
    eta_c = float(cfg.get("charge_eff", DEFAULT_CHARGE_EFF))
    eta_d = float(cfg.get("discharge_eff", DEFAULT_DISCHARGE_EFF))
    degr_cost = float(cfg.get("degradation_cost", DEFAULT_DEGRADATION_COST))
    plant_type = cfg.get("plant_type", "BESS Only")
    has_solar = "Solar" in plant_type
    has_wind = "Wind" in plant_type

    soc_initial = max(soc_min, min(float(soc_initial), soc_max))
    if terminal_soc is None:
        terminal_soc = soc_initial
    terminal_soc = max(soc_min, min(float(terminal_soc), soc_max))

    derate = cfg.get("_derate")
    if derate is None:
        derate = np.ones(T)
    else:
        derate = np.asarray(derate, dtype=float)[:T]

    p_dam = df["DAM_Price"].values.astype(float)
    p_gdam = df["GDAM_Price"].values.astype(float)
    p_rtm = df["RTM_Price_DA"].values.astype(float)
    # Best (highest) sale price and cheapest (lowest) buy price per block
    p_sell = np.maximum(np.maximum(p_dam, p_gdam), p_rtm)
    p_buy = np.minimum(np.minimum(p_dam, p_gdam), p_rtm)

    sy = df["SY_DA"].values.astype(float) if (has_solar and "SY_DA" in df.columns) else np.zeros(T)
    wy = df["WY_DA"].values.astype(float) if (has_wind and "WY_DA" in df.columns) else np.zeros(T)
    renew = sy + wy

    m = linopy.Model()
    t_idx = pd.RangeIndex(T, name="t")
    coords = [t_idx]

    renew_s = pd.Series(renew, index=t_idx)
    p_sell_s = pd.Series(p_sell, index=t_idx)
    p_buy_s = pd.Series(p_buy, index=t_idx)
    power_cap_s = pd.Series(batt_power * derate, index=t_idx)

    # ── Variables ──────────────────────────────────────────────────────────
    sell_renew = m.add_variables(lower=0, coords=coords, name="Sell_Renew")      # renewable -> market, now
    chg_renew = m.add_variables(lower=0, coords=coords, name="Charge_Renew")     # renewable -> battery
    grid_upper = pd.Series(np.inf if allow_grid_charging else 0.0, index=t_idx)
    chg_grid = m.add_variables(lower=0, upper=grid_upper, coords=coords, name="Charge_Grid")  # market -> battery
    discharge = m.add_variables(lower=0, coords=coords, name="Discharge")       # battery -> market
    soc = m.add_variables(lower=soc_min, upper=soc_max, coords=coords, name="SOC")
    b = m.add_variables(binary=True, coords=coords, name="b")                   # 1 = discharging slot

    # ── Objective: maximise net revenue (linopy minimises -> negate) ───────
    charge_total = chg_renew + chg_grid
    throughput = charge_total + discharge
    net_revenue = (
        p_sell_s * sell_renew
        + p_sell_s * discharge
        - p_buy_s * chg_grid
        - degr_cost * throughput
    )
    m.add_objective(-net_revenue.sum())

    # ── Constraints ──────────────────────────────────────────────────────────
    # Renewable energy balance: can't sell/charge more than available yield
    m.add_constraints(sell_renew + chg_renew <= renew_s, name="renew_balance")

    # Power rating limits with binary mutual exclusion (no simultaneous C/D)
    m.add_constraints(
        charge_total + power_cap_s * b <= power_cap_s, name="charge_rate"
    )
    m.add_constraints(
        discharge - power_cap_s * b <= 0, name="discharge_rate"
    )

    # SOC dynamics (efficiency-aware): soc[t] = soc[t-1] + 0.25*(chg*eta_c - dis/eta_d)
    for t in range(T):
        chg_t = charge_total.isel(t=t)
        dis_t = discharge.isel(t=t)
        if t == 0:
            m.add_constraints(
                soc.isel(t=0) - BLOCK_HOURS * eta_c * chg_t + (BLOCK_HOURS / eta_d) * dis_t
                == float(soc_initial),
                name="soc_0",
            )
        else:
            m.add_constraints(
                soc.isel(t=t) - soc.isel(t=t - 1)
                - BLOCK_HOURS * eta_c * chg_t + (BLOCK_HOURS / eta_d) * dis_t == 0,
                name=f"soc_{t}",
            )
    m.add_constraints(soc.isel(t=T - 1) >= float(terminal_soc), name="terminal_soc")

    # ── Solve ────────────────────────────────────────────────────────────────
    m.solve(solver_name="highs", io_api="lp")
    solver_info = {
        "status": m.status,
        "termination_condition": m.termination_condition,
        "objective_value": _extract_objective(m),
    }
    if m.status != "ok":
        raise RuntimeError(
            f"MILP solver did not find an optimal solution.\n"
            f"Status: {m.status} | Condition: {m.termination_condition}"
        )
    sol = getattr(m, "solution", None)
    if sol is None:
        raise RuntimeError(
            "MILP solved (status=ok) but solution dataset is empty. Re-run the dispatch."
        )

    def _sol(name: str) -> np.ndarray:
        try:
            return sol[name].values
        except (KeyError, AttributeError) as exc:
            raise RuntimeError(f"Solution extraction failed for '{name}': {exc}") from exc

    soc_vals = _sol("SOC")
    result = pd.DataFrame({
        "SOC_start": np.concatenate([[float(soc_initial)], soc_vals[:-1]]),
        "SOC_planned": soc_vals,
        "Sell_Renew": np.clip(_sol("Sell_Renew"), 0, None),
        "Charge_Renew": np.clip(_sol("Charge_Renew"), 0, None),
        "Charge_Grid": np.clip(_sol("Charge_Grid"), 0, None),
        "Discharge": np.clip(_sol("Discharge"), 0, None),
    })
    passthrough = ["Block", "Time", "DAM_Price", "GDAM_Price", "RTM_Price_DA",
                   "RTM_Price_ID", "SY_DA", "WY_DA", "SY_ID", "WY_ID", "Temp_C"]
    for col in passthrough:
        if col in df.columns:
            result[col] = df[col].values
    return result, solver_info


def compute_revenue(results_df: pd.DataFrame, plant_config: dict) -> dict:
    """
    Compute the net-revenue breakdown for a completed BESS arbitrage dispatch.
    Works for both the rule-based and MILP engines (same column schema).
    No PPA / penalty terms exist in this model.
    """
    df = results_df.copy()
    degr_cost = float(plant_config.get("degradation_cost", DEFAULT_DEGRADATION_COST))

    def _s(col):
        return df[col].clip(lower=0) if col in df.columns else pd.Series(0.0, index=df.index)

    def _p(col):
        return df[col] if col in df.columns else pd.Series(0.0, index=df.index)

    p_sell = pd.concat([_p("DAM_Price"), _p("GDAM_Price"), _p("RTM_Price_DA")], axis=1).max(axis=1)
    p_buy = pd.concat([_p("DAM_Price"), _p("GDAM_Price"), _p("RTM_Price_DA")], axis=1).min(axis=1)

    renew_direct_rev = float((p_sell * _s("Sell_Renew") * BLOCK_FACTOR).sum())
    discharge_rev = float((p_sell * _s("Discharge") * BLOCK_FACTOR).sum())
    grid_charge_cost = float((p_buy * _s("Charge_Grid") * BLOCK_FACTOR).sum())
    throughput = _s("Charge_Renew") + _s("Charge_Grid") + _s("Discharge")
    degradation_cost_total = float((degr_cost * throughput * BLOCK_FACTOR).sum())

    total = renew_direct_rev + discharge_rev - grid_charge_cost - degradation_cost_total
    return {
        "Renewable Direct-Sale Revenue": round(renew_direct_rev, 2),
        "Battery Discharge Revenue": round(discharge_rev, 2),
        "Grid Charging Cost": round(-grid_charge_cost, 2),
        "Degradation Cost": round(-degradation_cost_total, 2),
        "Total Net Revenue": round(total, 2),
    }
