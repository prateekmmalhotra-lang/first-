"""
Batsimulator — BESS Energy Arbitrage Simulator
===============================================
Standalone merchant arbitrage app. The plant may be BESS-only or a hybrid
(Solar and/or Wind + BESS) — but there is NO PPA / FDRE delivery obligation
anywhere in this model. Renewable generation, when present, is simply a
free energy source that can be sold directly to the market or routed
through the battery for a better price later.

Two dispatch engines:
  - Rule-Based  — fast, transparent, price-threshold heuristic
  - MILP        — globally optimal, solves all blocks at once (Linopy + HiGHS)

Day-Ahead (DA) plans on forecast prices/yields; Intraday (ID) re-dispatches
on revised prices/yields, with SOC chained forward using ID actuals.
"""

from __future__ import annotations

import math
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from bat_optimizer import solve_bess_milp, compute_revenue, LINOPY_AVAILABLE
except Exception:
    LINOPY_AVAILABLE = False
    solve_bess_milp = None   # type: ignore[assignment]
    compute_revenue = None   # type: ignore[assignment]

st.set_page_config(
    page_title="Batsimulator — BESS Arbitrage",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    _fragment = st.fragment
except AttributeError:
    def _fragment(func):
        return func

# ── Constants ─────────────────────────────────────────────────────────────
BLOCK_HOURS = 0.25     # hours per 15-min block
BLOCK_FACTOR = 250.0   # MW * 0.25h * 1000 kWh/MWh -> ₹ when multiplied by ₹/kWh price

PLANT_TYPES = ["BESS Only", "Hybrid (Solar + BESS)", "Hybrid (Wind + BESS)", "Hybrid (Solar + Wind + BESS)"]

REQUIRED_BASE_COLUMNS = ["DAM_Price", "GDAM_Price", "RTM_Price_DA", "RTM_Price_ID", "Temp_C"]
RENEWABLE_COLUMNS = ["SY_DA", "SY_ID", "WY_DA", "WY_ID"]


def _default_plant_config() -> dict:
    return {
        "plant_type": "BESS Only",
        "solar_cap": 50.0,
        "wind_cap": 50.0,
        "batt_energy": 100.0,
        "batt_power": 25.0,
        "soc_min_pct": 5.0,
        "soc_max_pct": 95.0,
        "charge_eff": 0.95,
        "discharge_eff": 0.95,
        "degradation_cost": 0.15,
        "max_cycles_per_day": 1.5,
        "enforce_cycle_cap": False,
        "allow_grid_charging": True,
        "cold_threshold_c": 5.0,
        "hot_threshold_c": 40.0,
        "derate_pct_outside_band": 15.0,
        "soc_initial_pct": 50.0,
        "terminal_soc_pct": 50.0,
    }


def sidebar_plant_config() -> dict:
    with st.sidebar:
        st.header("🔋 Batsimulator")
        st.caption("Pure merchant arbitrage — no PPA, no delivery obligation.")

        plant_type = st.selectbox("Plant Configuration", PLANT_TYPES, index=0)
        has_solar = "Solar" in plant_type
        has_wind = "Wind" in plant_type

        with st.expander("Renewable Capacity", expanded=has_solar or has_wind):
            if has_solar:
                solar_cap = st.number_input("Solar Capacity (MWp)", min_value=1.0, max_value=2000.0, value=50.0, step=5.0)
            else:
                solar_cap = 0.0
                st.caption("No solar in this configuration.")
            if has_wind:
                wind_cap = st.number_input("Wind Capacity (MW)", min_value=1.0, max_value=2000.0, value=50.0, step=5.0)
            else:
                wind_cap = 0.0
                st.caption("No wind in this configuration.")

        with st.expander("Battery Configuration", expanded=True):
            batt_energy = st.number_input("Battery Energy Capacity (MWh)", min_value=1.0, max_value=5000.0, value=100.0, step=10.0)
            batt_power = st.number_input("Battery Power Rating (MW)", min_value=1.0, max_value=2000.0, value=25.0, step=5.0,
                                          help="Max charge/discharge rate. Per 15-min block = Power × 0.25 MWh.")
            soc_min_pct, soc_max_pct = st.slider("Usable SOC Band (%)", 0, 100, (5, 95),
                                                  help="Battery is kept within this band to protect longevity.")
            c1, c2 = st.columns(2)
            charge_eff = c1.number_input("Charge Efficiency", min_value=0.5, max_value=1.0, value=0.95, step=0.01)
            discharge_eff = c2.number_input("Discharge Efficiency", min_value=0.5, max_value=1.0, value=0.95, step=0.01)
            rte = charge_eff * discharge_eff * 100
            st.caption(f"Round-trip efficiency: **{rte:.1f}%**")
            degradation_cost = st.number_input("Degradation Cost (₹/kWh throughput)", min_value=0.0, max_value=5.0, value=0.15, step=0.05,
                                                help="Wear-and-tear cost charged on every MWh charged or discharged. "
                                                     "Prevents the optimizer from cycling for trivial spreads.")
            enforce_cycle_cap = st.checkbox("Enforce daily cycle cap", value=False)
            max_cycles_per_day = st.number_input("Max Cycles / Day", min_value=0.1, max_value=5.0, value=1.5, step=0.1,
                                                  disabled=not enforce_cycle_cap)
            allow_grid_charging = st.checkbox("Allow charging from the grid/market", value=True,
                                               help="Unchecked: battery can only be charged from renewable surplus "
                                                    "(meaningless for BESS-only plants).")

        with st.expander("SOC Settings", expanded=False):
            soc_initial_pct = st.slider("Starting SOC (%)", 0, 100, 50)
            terminal_soc_pct = st.slider("Target End-of-Day SOC (%)", 0, 100, 50,
                                          help="Prevents the optimizer from dumping the battery at the end of the horizon.")

        with st.expander("Battery Temperature Derating", expanded=False):
            st.caption("Ambient temperature outside the comfort band reduces usable power & efficiency.")
            cold_threshold_c = st.number_input("Cold Threshold (°C)", min_value=-20.0, max_value=20.0, value=5.0, step=1.0)
            hot_threshold_c = st.number_input("Hot Threshold (°C)", min_value=20.0, max_value=55.0, value=40.0, step=1.0)
            derate_pct_outside_band = st.slider("Derate Outside Band (%)", 0, 50, 15)

    return {
        "plant_type": plant_type,
        "solar_cap": float(solar_cap),
        "wind_cap": float(wind_cap),
        "batt_energy": float(batt_energy),
        "batt_power": float(batt_power),
        "soc_min_pct": float(soc_min_pct),
        "soc_max_pct": float(soc_max_pct),
        "charge_eff": float(charge_eff),
        "discharge_eff": float(discharge_eff),
        "degradation_cost": float(degradation_cost),
        "max_cycles_per_day": float(max_cycles_per_day),
        "enforce_cycle_cap": bool(enforce_cycle_cap),
        "allow_grid_charging": bool(allow_grid_charging),
        "cold_threshold_c": float(cold_threshold_c),
        "hot_threshold_c": float(hot_threshold_c),
        "derate_pct_outside_band": float(derate_pct_outside_band),
        "soc_initial_pct": float(soc_initial_pct),
        "terminal_soc_pct": float(terminal_soc_pct),
    }


# ── Helpers ───────────────────────────────────────────────────────────────
def block_time_label(block_idx: int) -> str:
    """0-based block index -> 'HH:MM' label."""
    mins = block_idx * 15
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def temp_derate(temp_c: float, plant_config: dict) -> float:
    """Return a 0-1 multiplier applied to battery power rating for this block's temperature."""
    cold = plant_config["cold_threshold_c"]
    hot = plant_config["hot_threshold_c"]
    pct = plant_config["derate_pct_outside_band"] / 100.0
    if temp_c < cold or temp_c > hot:
        return max(0.0, 1.0 - pct)
    return 1.0


def _solar_curve(hour: float) -> float:
    """Bell-shaped solar generation factor (0-1) peaking at solar noon (~12:30)."""
    if hour < 6 or hour > 19:
        return 0.0
    x = (hour - 12.5) / 6.5
    return max(0.0, math.cos(x * math.pi / 2) ** 1.3)


def generate_demo_96block(seed: int, plant_config: dict) -> pd.DataFrame:
    """Deterministic (seeded) 96-block demo dataset: prices, weather, and
    (if hybrid) solar/wind yields. Mimics typical IEX-style daily price shape
    with morning/evening peaks and a cheap midday/overnight trough."""
    rng = np.random.default_rng(seed)
    n = 96
    blocks = np.arange(1, n + 1)
    hours = (blocks - 1) * 0.25

    # ── Price shape: trough at night, shoulder rise, morning + evening peaks
    base = 3.2 + 1.6 * np.sin((hours - 6.5) / 24 * 2 * math.pi * 2.1)
    morning_peak = 2.0 * np.exp(-((hours - 8.5) ** 2) / (2 * 1.3 ** 2))
    evening_peak = 2.8 * np.exp(-((hours - 19.5) ** 2) / (2 * 1.6 ** 2))
    night_trough = -1.4 * np.exp(-((hours - 3.0) ** 2) / (2 * 2.2 ** 2))
    dam = np.clip(base + morning_peak + evening_peak + night_trough + rng.normal(0, 0.12, n), 1.0, 12.0)
    gdam = np.clip(dam * rng.uniform(0.9, 1.05, n) + rng.normal(0, 0.1, n), 1.0, 12.0)
    rtm_da = np.clip(dam * rng.uniform(0.92, 1.15, n) + rng.normal(0, 0.15, n), 1.0, 14.0)
    rtm_id = np.clip(rtm_da * rng.uniform(0.85, 1.2, n) + rng.normal(0, 0.2, n), 1.0, 15.0)

    # ── Temperature: daily sinusoid, trough ~05:00, peak ~15:00
    temp_c = 24 + 9 * np.sin((hours - 7.0) / 24 * 2 * math.pi) + rng.normal(0, 0.6, n)

    plant_type = plant_config.get("plant_type", "BESS Only")
    has_solar = "Solar" in plant_type
    has_wind = "Wind" in plant_type
    solar_cap = plant_config.get("solar_cap", 0.0)
    wind_cap = plant_config.get("wind_cap", 0.0)

    sy_da = np.array([solar_cap * _solar_curve(h) for h in hours]) if has_solar else np.zeros(n)
    if has_solar:
        sy_da = np.clip(sy_da * rng.uniform(0.9, 1.05, n), 0, solar_cap)
        sy_id = np.clip(sy_da * rng.uniform(0.85, 1.1, n), 0, solar_cap)
    else:
        sy_id = np.zeros(n)

    if has_wind:
        wind_base = wind_cap * (0.35 + 0.25 * np.sin((hours - 2.0) / 24 * 2 * math.pi))
        wy_da = np.clip(wind_base + rng.normal(0, wind_cap * 0.08, n), 0, wind_cap)
        wy_id = np.clip(wy_da * rng.uniform(0.8, 1.2, n), 0, wind_cap)
    else:
        wy_da = np.zeros(n)
        wy_id = np.zeros(n)

    df = pd.DataFrame({
        "Block": blocks,
        "Time": [block_time_label(i) for i in range(n)],
        "DAM_Price": np.round(dam, 2),
        "GDAM_Price": np.round(gdam, 2),
        "RTM_Price_DA": np.round(rtm_da, 2),
        "RTM_Price_ID": np.round(rtm_id, 2),
        "Temp_C": np.round(temp_c, 1),
        "SY_DA": np.round(sy_da, 2),
        "SY_ID": np.round(sy_id, 2),
        "WY_DA": np.round(wy_da, 2),
        "WY_ID": np.round(wy_id, 2),
    })
    return df


def validate_input_df(df: pd.DataFrame, plant_config: dict) -> list[str]:
    errors: list[str] = []
    plant_type = plant_config.get("plant_type", "BESS Only")
    required = list(REQUIRED_BASE_COLUMNS)
    if "Solar" in plant_type:
        required += ["SY_DA", "SY_ID"]
    if "Wind" in plant_type:
        required += ["WY_DA", "WY_ID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns for this plant configuration: {', '.join(missing)}")
        return errors
    if len(df) < 2:
        errors.append("File must have at least 2 data rows.")
    if len(df) > 96:
        errors.append(f"File has {len(df)} rows; maximum supported is 96.")
    for col in required:
        bad = df[col].dropna().apply(lambda x: _safe_float(x) < 0).sum()
        if bad:
            errors.append(f"Column '{col}' has {bad} negative value(s).")
        n_na = df[col].isna().sum()
        if n_na:
            errors.append(f"Column '{col}' has {n_na} missing value(s).")
    return errors


# ── Dispatch Engines ─────────────────────────────────────────────────────
def _effective_prices(row) -> tuple[float, float]:
    """Return (sell_price, buy_price) for a DA-stage row: best market to sell to,
    cheapest market to buy from."""
    prices = [row["DAM_Price"], row["GDAM_Price"], row["RTM_Price_DA"]]
    return max(prices), min(prices)


def run_rule_based_dispatch(input_df: pd.DataFrame, soc_initial_mwh: float, plant_config: dict) -> pd.DataFrame:
    """
    Price-threshold heuristic, Day-Ahead stage:
      - Renewable generation is routed to the battery only when a later block's
        sell price clears the round-trip efficiency + degradation cost hurdle;
        otherwise it is sold immediately.
      - Grid charging only happens in the day's cheapest blocks, and only when
        a sufficiently higher future price exists to recover efficiency losses
        + degradation cost.
      - Discharge happens in the day's most expensive blocks, subject to the
        no-simultaneous-charge/discharge rule and available SOC.
    """
    n = len(input_df)
    cfg = plant_config
    batt_energy = cfg["batt_energy"]
    batt_power = cfg["batt_power"]
    soc_min = batt_energy * cfg["soc_min_pct"] / 100.0
    soc_max = batt_energy * cfg["soc_max_pct"] / 100.0
    eta_c = cfg["charge_eff"]
    eta_d = cfg["discharge_eff"]
    degr = cfg["degradation_cost"]
    rte = eta_c * eta_d
    allow_grid = cfg["allow_grid_charging"]

    sell_prices = np.zeros(n)
    buy_prices = np.zeros(n)
    derates = np.zeros(n)
    for i in range(n):
        row = input_df.iloc[i]
        sell_prices[i], buy_prices[i] = _effective_prices(row)
        derates[i] = temp_derate(_safe_float(row.get("Temp_C", 25.0)), cfg)

    cheap_threshold = float(np.percentile(buy_prices, 30))
    expensive_threshold = float(np.percentile(sell_prices, 70))

    renew_yield = np.array([
        _safe_float(input_df.iloc[i].get("SY_DA", 0.0)) + _safe_float(input_df.iloc[i].get("WY_DA", 0.0))
        for i in range(n)
    ])

    soc = float(soc_initial_mwh)
    rows = []
    for t in range(n):
        power_cap = batt_power * derates[t]
        lookahead_max = float(np.max(sell_prices[t + 1:])) if t + 1 < n else -np.inf

        # 1. Renewable routing: store if a materially better future price exists
        charge_renew = 0.0
        if renew_yield[t] > 0:
            energy_room = max(0.0, (soc_max - soc) / (BLOCK_HOURS * eta_c))
            room = min(power_cap, energy_room)
            worth_storing = (lookahead_max * rte - sell_prices[t]) > degr
            if room > 0 and worth_storing:
                charge_renew = min(renew_yield[t], room)
        sell_renew = renew_yield[t] - charge_renew

        # 2. Grid charging: only in cheap blocks, only if future spread clears costs
        charge_grid = 0.0
        if allow_grid and buy_prices[t] <= cheap_threshold:
            remaining_power = power_cap - charge_renew
            energy_room = max(0.0, (soc_max - soc) / (BLOCK_HOURS * eta_c)) - charge_renew
            room = max(0.0, min(remaining_power, energy_room))
            worth = (lookahead_max * rte - buy_prices[t]) > (2 * degr)
            if room > 0 and worth:
                charge_grid = room

        # 3. Discharge: only in expensive blocks, only if not already charging
        discharge = 0.0
        if (charge_renew + charge_grid) == 0 and sell_prices[t] >= expensive_threshold and soc > soc_min:
            energy_room = max(0.0, (soc - soc_min) * eta_d / BLOCK_HOURS)
            discharge = min(power_cap, energy_room)

        soc_start = soc
        soc = soc + BLOCK_HOURS * eta_c * (charge_renew + charge_grid) - (BLOCK_HOURS / eta_d) * discharge
        soc = max(soc_min, min(soc_max, soc))

        rows.append({
            "Block": int(input_df.iloc[t].get("Block", t + 1)),
            "Time": str(input_df.iloc[t].get("Time", block_time_label(t))),
            "Temp_C": _safe_float(input_df.iloc[t].get("Temp_C", 25.0)),
            "Derate": derates[t],
            "DAM_Price": _safe_float(input_df.iloc[t]["DAM_Price"]),
            "GDAM_Price": _safe_float(input_df.iloc[t]["GDAM_Price"]),
            "RTM_Price_DA": _safe_float(input_df.iloc[t]["RTM_Price_DA"]),
            "RTM_Price_ID": _safe_float(input_df.iloc[t].get("RTM_Price_ID", 0.0)),
            "SY_DA": _safe_float(input_df.iloc[t].get("SY_DA", 0.0)),
            "WY_DA": _safe_float(input_df.iloc[t].get("WY_DA", 0.0)),
            "SY_ID": _safe_float(input_df.iloc[t].get("SY_ID", 0.0)),
            "WY_ID": _safe_float(input_df.iloc[t].get("WY_ID", 0.0)),
            "SOC_start": round(soc_start, 4),
            "Sell_Renew": round(sell_renew, 4),
            "Charge_Renew": round(charge_renew, 4),
            "Charge_Grid": round(charge_grid, 4),
            "Discharge": round(discharge, 4),
        })
    da_df = pd.DataFrame(rows)
    return apply_id_reconciliation(da_df, plant_config, soc_initial_mwh)


def run_milp_dispatch(input_df: pd.DataFrame, soc_initial_mwh: float, plant_config: dict) -> tuple[pd.DataFrame, dict]:
    """Solve the DA stage via MILP, then apply the same ID reconciliation pass."""
    n = len(input_df)
    derates = np.array([
        temp_derate(_safe_float(input_df.iloc[i].get("Temp_C", 25.0)), plant_config) for i in range(n)
    ])
    cfg = dict(plant_config)
    cfg["_derate"] = derates
    da_df, solver_info = solve_bess_milp(
        input_df, soc_initial_mwh, cfg,
        terminal_soc=plant_config["batt_energy"] * plant_config["terminal_soc_pct"] / 100.0,
        allow_grid_charging=plant_config["allow_grid_charging"],
    )
    da_df["Temp_C"] = [_safe_float(input_df.iloc[i].get("Temp_C", 25.0)) for i in range(n)]
    da_df["Derate"] = derates
    da_df["Block"] = [int(input_df.iloc[i].get("Block", i + 1)) for i in range(n)]
    da_df["Time"] = [str(input_df.iloc[i].get("Time", block_time_label(i))) for i in range(n)]
    for col in ["SY_DA", "WY_DA", "SY_ID", "WY_ID", "DAM_Price", "GDAM_Price", "RTM_Price_DA", "RTM_Price_ID"]:
        if col not in da_df.columns:
            da_df[col] = [_safe_float(input_df.iloc[i].get(col, 0.0)) for i in range(n)]
    result = apply_id_reconciliation(da_df, plant_config, soc_initial_mwh)
    return result, solver_info


def apply_id_reconciliation(da_df: pd.DataFrame, plant_config: dict, soc_initial_mwh: float) -> pd.DataFrame:
    """
    Intraday reconciliation: grid-cleared volumes (Charge_Grid, Discharge) are
    locked DA commitments and carried forward unchanged. Renewable routing is
    re-applied proportionally to ACTUAL (ID) yield, within actual SOC headroom.
    The resulting SOC is chained forward using these ID actuals — exactly like
    physical battery operation.
    """
    cfg = plant_config
    batt_energy = cfg["batt_energy"]
    soc_min = batt_energy * cfg["soc_min_pct"] / 100.0
    soc_max = batt_energy * cfg["soc_max_pct"] / 100.0
    eta_c = cfg["charge_eff"]
    eta_d = cfg["discharge_eff"]

    n = len(da_df)
    soc = float(soc_initial_mwh)
    out_rows = []
    for i in range(n):
        row = da_df.iloc[i]
        renew_da = _safe_float(row.get("SY_DA", 0)) + _safe_float(row.get("WY_DA", 0))
        renew_id = _safe_float(row.get("SY_ID", 0)) + _safe_float(row.get("WY_ID", 0))
        charge_renew_da = _safe_float(row.get("Charge_Renew", 0))
        charge_grid = _safe_float(row.get("Charge_Grid", 0))    # locked DA commitment
        discharge = _safe_float(row.get("Discharge", 0))         # locked DA commitment
        power_cap = cfg["batt_power"] * _safe_float(row.get("Derate", 1.0))

        da_charge_frac = (charge_renew_da / renew_da) if renew_da > 1e-9 else 0.0
        desired_charge_id = da_charge_frac * renew_id

        energy_room = max(0.0, (soc_max - soc) / (BLOCK_HOURS * eta_c))
        power_room = max(0.0, power_cap - charge_grid)
        charge_renew_id = max(0.0, min(desired_charge_id, energy_room, power_room))
        sell_renew_id = max(0.0, renew_id - charge_renew_id)

        soc_start = soc
        soc = soc + BLOCK_HOURS * eta_c * (charge_renew_id + charge_grid) - (BLOCK_HOURS / eta_d) * discharge
        soc = max(soc_min, min(soc_max, soc))

        out_rows.append({
            **row.to_dict(),
            "SOC_start": round(soc_start, 4),
            "SOC_end": round(soc, 4),
            "Charge_Renew_ID": round(charge_renew_id, 4),
            "Charge_Grid_ID": round(charge_grid, 4),
            "Sell_Renew_ID": round(sell_renew_id, 4),
            "Discharge_ID": round(discharge, 4),
        })
    return pd.DataFrame(out_rows)


# ── Validation ────────────────────────────────────────────────────────────
ENERGY_TOL = 0.001

def validate_dispatch(results_df: pd.DataFrame, plant_config: dict) -> pd.DataFrame:
    batt_energy = plant_config["batt_energy"]
    soc_min = batt_energy * plant_config["soc_min_pct"] / 100.0
    soc_max = batt_energy * plant_config["soc_max_pct"] / 100.0
    checks = []
    for i, row in results_df.iterrows():
        chg = _safe_float(row.get("Charge_Renew_ID", 0)) + _safe_float(row.get("Charge_Grid_ID", 0))
        dis = _safe_float(row.get("Discharge_ID", 0))
        checks.append({
            "Block": int(row["Block"]), "Time": row["Time"],
            "check": "No simultaneous charge/discharge",
            "passed": not (chg > ENERGY_TOL and dis > ENERGY_TOL),
            "detail": f"{chg:g} MW charge and {dis:g} MW discharge",
        })
        in_range = (soc_min - ENERGY_TOL) <= row["SOC_start"] <= (soc_max + ENERGY_TOL) and \
                   (soc_min - ENERGY_TOL) <= row["SOC_end"] <= (soc_max + ENERGY_TOL)
        checks.append({
            "Block": int(row["Block"]), "Time": row["Time"],
            "check": f"SOC within usable band [{soc_min:.0f}, {soc_max:.0f}] MWh",
            "passed": in_range,
            "detail": f"SOC_start={row['SOC_start']:.1f}  SOC_end={row['SOC_end']:.1f}",
        })
    for i in range(len(results_df) - 1):
        cur, nxt = results_df.iloc[i], results_df.iloc[i + 1]
        match = abs(cur["SOC_end"] - nxt["SOC_start"]) < 1e-6
        checks.append({
            "Block": int(cur["Block"]), "Time": cur["Time"],
            "check": "SOC chain continuity",
            "passed": match,
            "detail": f"SOC_end[{int(cur['Block'])}]={cur['SOC_end']:.3f} vs SOC_start[{int(nxt['Block'])}]={nxt['SOC_start']:.3f}",
        })
    # Cycle cap check (aggregate, single row)
    if plant_config.get("enforce_cycle_cap"):
        total_discharge_mwh = float(results_df["Discharge_ID"].clip(lower=0).sum()) * BLOCK_HOURS
        cycles_used = total_discharge_mwh / batt_energy if batt_energy > 0 else 0.0
        checks.append({
            "Block": 0, "Time": "Day Total",
            "check": f"Daily cycle cap ({plant_config['max_cycles_per_day']:.2f} cycles)",
            "passed": cycles_used <= plant_config["max_cycles_per_day"] + 1e-6,
            "detail": f"{cycles_used:.2f} cycles used",
        })
    return pd.DataFrame(checks)


def aggregate_day_summary(results_df: pd.DataFrame, plant_config: dict) -> dict:
    batt_energy = plant_config["batt_energy"]
    total_discharge_mwh = float(results_df["Discharge_ID"].clip(lower=0).sum()) * BLOCK_HOURS
    total_charge_mwh = float((results_df["Charge_Renew_ID"].clip(lower=0) + results_df["Charge_Grid_ID"].clip(lower=0)).sum()) * BLOCK_HOURS
    cycles_used = total_discharge_mwh / batt_energy if batt_energy > 0 else 0.0
    renew_total_mwh = float((results_df.get("SY_ID", 0) + results_df.get("WY_ID", 0)).clip(lower=0).sum()) * BLOCK_HOURS if \
        ("SY_ID" in results_df.columns or "WY_ID" in results_df.columns) else 0.0
    renew_sold_direct_mwh = float(results_df["Sell_Renew_ID"].clip(lower=0).sum()) * BLOCK_HOURS
    renew_to_battery_mwh = float(results_df["Charge_Renew_ID"].clip(lower=0).sum()) * BLOCK_HOURS
    return {
        "Total Charge (MWh)": round(total_charge_mwh, 1),
        "Total Discharge (MWh)": round(total_discharge_mwh, 1),
        "Cycles Used": round(cycles_used, 2),
        "Renewable Generated (MWh)": round(renew_total_mwh, 1),
        "Renewable Sold Direct (MWh)": round(renew_sold_direct_mwh, 1),
        "Renewable to Battery (MWh)": round(renew_to_battery_mwh, 1),
        "End of Day SOC (MWh)": round(float(results_df["SOC_end"].iloc[-1]), 1),
        "Min SOC (MWh)": round(float(results_df["SOC_start"].min()), 1),
        "Max SOC (MWh)": round(float(results_df["SOC_start"].max()), 1),
        "Blocks Charging": int(((results_df["Charge_Renew_ID"] + results_df["Charge_Grid_ID"]) > 0.01).sum()),
        "Blocks Discharging": int((results_df["Discharge_ID"] > 0.01).sum()),
    }


# ── Charts ────────────────────────────────────────────────────────────────
def create_price_chart(results_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, color in [("DAM_Price", "#2563eb"), ("GDAM_Price", "#0891b2"),
                       ("RTM_Price_DA", "#dc2626"), ("RTM_Price_ID", "#f97316")]:
        fig.add_trace(go.Scatter(x=results_df["Time"], y=results_df[col], mode="lines",
                                  name=col.replace("_", " "), line=dict(width=1.8, color=color)))
    # Shade charge / discharge blocks
    charging = results_df[(results_df["Charge_Renew_ID"] + results_df["Charge_Grid_ID"]) > 0.01]
    discharging = results_df[results_df["Discharge_ID"] > 0.01]
    for _, r in charging.iterrows():
        fig.add_vrect(x0=r["Time"], x1=r["Time"], fillcolor="#16a34a", opacity=0.12, line_width=0)
    for _, r in discharging.iterrows():
        fig.add_vrect(x0=r["Time"], x1=r["Time"], fillcolor="#dc2626", opacity=0.12, line_width=0)
    fig.update_layout(title="Market Prices (green = charging, red = discharging)",
                       xaxis_title="Time", yaxis_title="₹/kWh", height=380,
                       xaxis=dict(tickmode="linear", dtick=8), legend=dict(orientation="h", y=-0.3))
    return fig


def create_soc_chart(results_df: pd.DataFrame, plant_config: dict) -> go.Figure:
    batt_energy = plant_config["batt_energy"]
    soc_min = batt_energy * plant_config["soc_min_pct"] / 100.0
    soc_max = batt_energy * plant_config["soc_max_pct"] / 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df["Time"], y=results_df["SOC_end"], mode="lines",
                              name="SOC", fill="tozeroy", line=dict(color="#7c3aed", width=2)))
    fig.add_hline(y=soc_max, line_dash="dash", line_color="#dc2626", annotation_text="SOC max")
    fig.add_hline(y=soc_min, line_dash="dash", line_color="#dc2626", annotation_text="SOC min")
    fig.update_layout(title="Battery State of Charge", xaxis_title="Time", yaxis_title="MWh",
                       height=320, xaxis=dict(tickmode="linear", dtick=8))
    return fig


def create_yield_chart(results_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if results_df["SY_ID"].abs().sum() > 0:
        fig.add_trace(go.Scatter(x=results_df["Time"], y=results_df["SY_ID"], mode="lines",
                                  name="Solar (MW)", line=dict(color="#f59e0b", width=2)))
    if results_df["WY_ID"].abs().sum() > 0:
        fig.add_trace(go.Scatter(x=results_df["Time"], y=results_df["WY_ID"], mode="lines",
                                  name="Wind (MW)", line=dict(color="#06b6d4", width=2)))
    fig.update_layout(title="Renewable Generation (Actual / Intraday)", xaxis_title="Time",
                       yaxis_title="MW", height=320, xaxis=dict(tickmode="linear", dtick=8))
    return fig


def create_dispatch_stack(results_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=results_df["Time"], y=results_df["Charge_Renew_ID"], name="Charge ← Renewable",
                          marker_color="#16a34a"))
    fig.add_trace(go.Bar(x=results_df["Time"], y=results_df["Charge_Grid_ID"], name="Charge ← Grid",
                          marker_color="#2563eb"))
    fig.add_trace(go.Bar(x=results_df["Time"], y=-results_df["Discharge_ID"], name="Discharge → Market",
                          marker_color="#dc2626"))
    fig.add_trace(go.Bar(x=results_df["Time"], y=-results_df["Sell_Renew_ID"], name="Renewable → Market (direct)",
                          marker_color="#f59e0b"))
    fig.update_layout(barmode="relative", title="Energy Flows per Block (+ve = into battery, -ve = to market)",
                       xaxis_title="Time", yaxis_title="MW", height=420,
                       xaxis=dict(tickmode="linear", dtick=8), legend=dict(orientation="h", y=-0.3))
    return fig


def create_revenue_bar(rev: dict) -> go.Figure:
    labels = [k for k in rev if k != "Total Net Revenue"]
    values = [rev[k] for k in labels]
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                            text=[f"₹{v:,.0f}" for v in values], textposition="outside"))
    fig.update_layout(title="Revenue Breakdown", yaxis_title="₹", height=380,
                       xaxis=dict(tickangle=-20), margin=dict(b=100))
    return fig


def generate_block_narrative(row: pd.Series, plant_config: dict) -> str:
    lines: list[str] = []
    sell = max(row["DAM_Price"], row["GDAM_Price"], row["RTM_Price_DA"])
    buy = min(row["DAM_Price"], row["GDAM_Price"], row["RTM_Price_DA"])
    lines.append(f"**Best sell price:** ₹{sell:.2f}/kWh | **Best buy price:** ₹{buy:.2f}/kWh")
    if row.get("Derate", 1.0) < 1.0:
        lines.append(f"⚠ **Temperature derating active** ({row['Temp_C']:.1f}°C) — "
                      f"battery power capped at {row['Derate']*100:.0f}% of rating this block.")
    renew_id = _safe_float(row.get("SY_ID", 0)) + _safe_float(row.get("WY_ID", 0))
    if renew_id > 0.1:
        lines.append(f"**Renewable generation (actual):** {renew_id:.1f} MW")
        if row["Charge_Renew_ID"] > 0.1:
            lines.append(f"- {row['Charge_Renew_ID']:.1f} MW routed to battery (better price expected later).")
        if row["Sell_Renew_ID"] > 0.1:
            lines.append(f"- {row['Sell_Renew_ID']:.1f} MW sold directly to market at ₹{sell:.2f}/kWh.")
    if row["Charge_Grid_ID"] > 0.1:
        lines.append(f"**Grid charging:** {row['Charge_Grid_ID']:.1f} MW bought at ₹{buy:.2f}/kWh "
                      f"— cheap block, battery topping up for a later high-price discharge.")
    if row["Discharge_ID"] > 0.1:
        lines.append(f"**Battery discharge:** {row['Discharge_ID']:.1f} MW sold at ₹{sell:.2f}/kWh "
                      f"— high-price block, battery monetising stored energy.")
    if row["Charge_Renew_ID"] < 0.1 and row["Charge_Grid_ID"] < 0.1 and row["Discharge_ID"] < 0.1:
        lines.append("**Battery idle this block** — price spread insufficient to clear efficiency + degradation costs.")
    lines.append(f"**SOC:** {row['SOC_start']:.1f} → {row['SOC_end']:.1f} MWh")
    return "\n\n".join(lines)


# ── Main App ──────────────────────────────────────────────────────────────
plant_config = sidebar_plant_config()
is_hybrid = plant_config["plant_type"] != "BESS Only"

st.title("🔋 Batsimulator — BESS Energy Arbitrage")
st.caption(
    "Pure merchant arbitrage simulator for a BESS plant, standalone or hybrid with Solar/Wind. "
    "**No PPA, no FDRE agreement, no delivery obligation** — every decision is driven purely by price."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Market & Weather Data")
data_source = st.sidebar.radio("Data source", ["Use demo data", "Upload file"], index=0)
demo_seed = st.sidebar.number_input("Demo data seed", min_value=1, max_value=99999, value=42, step=1,
                                     help="Change the seed to generate a different (but reproducible) demo day.")
demo_df = generate_demo_96block(seed=int(demo_seed), plant_config=plant_config)

if data_source == "Upload file":
    uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel (up to 96 blocks)", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        st.info("👈 Upload a file, or switch to **Use demo data** in the sidebar to try the app immediately.")
        required = list(REQUIRED_BASE_COLUMNS)
        if "Solar" in plant_config["plant_type"]:
            required += ["SY_DA", "SY_ID"]
        if "Wind" in plant_config["plant_type"]:
            required += ["WY_DA", "WY_ID"]
        st.markdown("**Required columns for this plant configuration:**")
        st.code(", ".join(required))
        st.stop()
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        st.stop()
    errs = validate_input_df(input_df, plant_config)
    if errs:
        for e in errs:
            st.error(e)
        st.stop()
else:
    input_df = demo_df.copy()
    st.sidebar.success("Using generated demo data (96 blocks).")

if "Block" not in input_df.columns:
    input_df.insert(0, "Block", range(1, len(input_df) + 1))
if "Time" not in input_df.columns:
    input_df.insert(1, "Time", [block_time_label(i) for i in range(len(input_df))])
if "Temp_C" not in input_df.columns:
    input_df["Temp_C"] = 25.0
for col in RENEWABLE_COLUMNS:
    if col not in input_df.columns:
        input_df[col] = 0.0
input_df = input_df.sort_values("Block").reset_index(drop=True)

with st.expander("✏️ Edit input data before dispatch", expanded=False):
    st.caption("Edit any cell directly. Changes apply when you click outside the cell.")
    input_df = st.data_editor(input_df, use_container_width=True, num_rows="fixed", hide_index=True, key="input_editor")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙ Dispatch Engine")
engine_choice = st.sidebar.radio("Engine", ["Rule-Based", "MILP Optimized"], index=0,
                                  help="MILP requires linopy + highspy to be installed in this environment.")
engine_is_milp = engine_choice == "MILP Optimized"
if engine_is_milp and not LINOPY_AVAILABLE:
    st.sidebar.warning("linopy/highspy not available — running Rule-Based instead.")
    engine_is_milp = False

soc_initial_mwh = plant_config["batt_energy"] * plant_config["soc_initial_pct"] / 100.0

import hashlib
import json


def _make_cache_key(df: pd.DataFrame, cfg: dict, is_milp: bool) -> str:
    df_hash = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f"batsim_{df_hash}_{cfg_hash}_{is_milp}"


_cache_key = _make_cache_key(input_df, plant_config, engine_is_milp)
solver_info: dict | None = None
if st.session_state.get("_bat_cache_key") == _cache_key:
    results_df = st.session_state["_bat_results"]
    solver_info = st.session_state.get("_bat_solver_info")
else:
    with st.spinner(f"Running {'MILP' if engine_is_milp else 'rule-based'} dispatch for {len(input_df)} blocks…"):
        if engine_is_milp:
            try:
                results_df, solver_info = run_milp_dispatch(input_df, soc_initial_mwh, plant_config)
            except Exception as exc:
                st.error(f"MILP solver error: {exc}")
                st.info("Falling back to rule-based engine.")
                results_df = run_rule_based_dispatch(input_df, soc_initial_mwh, plant_config)
                solver_info = {"status": "error", "termination_condition": str(exc), "objective_value": None}
        else:
            results_df = run_rule_based_dispatch(input_df, soc_initial_mwh, plant_config)
    st.session_state["_bat_cache_key"] = _cache_key
    st.session_state["_bat_results"] = results_df
    st.session_state["_bat_solver_info"] = solver_info

validation_df = validate_dispatch(results_df, plant_config)
summary = aggregate_day_summary(results_df, plant_config)
rev = compute_revenue(results_df, plant_config) if compute_revenue is not None else None
n_failed = int((validation_df["passed"] == False).sum())

engine_badge = "🔴 MILP Optimized" if engine_is_milp else "⚙ Rule-Based"
st.subheader(f"Day Schedule — {len(results_df)} blocks | Starting SOC: {soc_initial_mwh:.0f} MWh | {engine_badge} | {plant_config['plant_type']}")
if solver_info and engine_is_milp:
    si_status = solver_info.get("status", "?")
    si_obj = solver_info.get("objective_value")
    si_cond = solver_info.get("termination_condition", "?")
    if si_status == "ok":
        obj_str = f"₹{si_obj:,.2f}" if si_obj is not None else "n/a"
        st.success(f"MILP solved ✅  Objective (net revenue): {obj_str}  |  Condition: {si_cond}")
    else:
        st.error(f"MILP status: {si_status}  |  {si_cond}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Charge (MWh)", f"{summary['Total Charge (MWh)']:,.1f}")
m2.metric("Total Discharge (MWh)", f"{summary['Total Discharge (MWh)']:,.1f}")
m3.metric("Cycles Used", f"{summary['Cycles Used']:.2f}")
m4.metric("End of Day SOC (MWh)", f"{summary['End of Day SOC (MWh)']:.1f}")
m5.metric("Validation Failures", str(n_failed), delta="⚠ check Validation tab" if n_failed else None,
           delta_color="inverse")

if rev is not None:
    with st.expander("💰 Revenue Breakdown (indicative)", expanded=True):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Renewable Direct Sale", f"₹{rev['Renewable Direct-Sale Revenue']:,.0f}")
        r1c2.metric("Battery Discharge Revenue", f"₹{rev['Battery Discharge Revenue']:,.0f}")
        r1c3.metric("Grid Charging Cost", f"₹{rev['Grid Charging Cost']:,.0f}",
                    delta="cost" if rev["Grid Charging Cost"] < 0 else None, delta_color="inverse")
        r1c4.metric("Degradation Cost", f"₹{rev['Degradation Cost']:,.0f}",
                    delta="cost" if rev["Degradation Cost"] < 0 else None, delta_color="inverse")
        st.markdown("---")
        total = rev["Total Net Revenue"]
        colour = "#16a34a" if total >= 0 else "#dc2626"
        bg = "#e6f4ea" if total >= 0 else "#fce8e6"
        tc = st.columns([1, 2, 1])
        tc[1].markdown(
            f"<div style='text-align:center;padding:8px;border-radius:6px;background:{bg};'>"
            f"<span style='font-size:14px;color:#555'>Total Net Revenue</span><br>"
            f"<span style='font-size:28px;font-weight:700;color:{colour}'>₹{total:,.0f}</span></div>",
            unsafe_allow_html=True,
        )
st.divider()

tab_overview, tab_schedule, tab_detail, tab_analytics, tab_compare, tab_validation, tab_data, tab_strategy, tab_multiday, tab_annexure = st.tabs(
    ["Overview", "Dispatch Schedule", "Block Detail", "Analytics", "⚡ Compare Engines",
     "Validation", "Full Data", "📊 Strategy Comparison", "📅 Multi-Day Simulation", "📚 Annexure"]
)

# ── Tab 1: Overview ────────────────────────────────────────────────────────
with tab_overview:
    st.plotly_chart(create_soc_chart(results_df, plant_config), use_container_width=True)
    if is_hybrid:
        col1, col2 = st.columns(2)
        col1.plotly_chart(create_price_chart(results_df), use_container_width=True)
        col2.plotly_chart(create_yield_chart(results_df), use_container_width=True)
    else:
        st.plotly_chart(create_price_chart(results_df), use_container_width=True)

# ── Tab 2: Dispatch Schedule ───────────────────────────────────────────────
with tab_schedule:
    st.plotly_chart(create_dispatch_stack(results_df), use_container_width=True)
    st.caption(
        "Green = renewable charging battery, blue = grid charging battery, "
        "red = battery discharging to market, orange = renewable sold directly."
    )

# ── Tab 3: Block Detail ────────────────────────────────────────────────────
with tab_detail:
    block_options = results_df["Block"].tolist()
    sel_block = st.selectbox("Select a block", block_options,
                              format_func=lambda b: f"Block {b} — {results_df.loc[results_df['Block']==b,'Time'].iloc[0]}")
    row = results_df[results_df["Block"] == sel_block].iloc[0]
    bc1, bc2 = st.columns([2, 1])
    with bc1:
        st.markdown(generate_block_narrative(row, plant_config))
    with bc2:
        st.metric("DAM", f"₹{row['DAM_Price']:.2f}")
        st.metric("GDAM", f"₹{row['GDAM_Price']:.2f}")
        st.metric("RTM (DA)", f"₹{row['RTM_Price_DA']:.2f}")
        st.metric("RTM (ID)", f"₹{row['RTM_Price_ID']:.2f}")

# ── Tab 4: Analytics ───────────────────────────────────────────────────────
with tab_analytics:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Renewable Generated (MWh)", f"{summary['Renewable Generated (MWh)']:,.1f}")
    a2.metric("Renewable Sold Direct (MWh)", f"{summary['Renewable Sold Direct (MWh)']:,.1f}")
    a3.metric("Renewable to Battery (MWh)", f"{summary['Renewable to Battery (MWh)']:,.1f}")
    a4.metric("Min SOC (MWh)", f"{summary['Min SOC (MWh)']:,.1f}")
    b1, b2, b3 = st.columns(3)
    b1.metric("Blocks Charging", str(summary["Blocks Charging"]))
    b2.metric("Blocks Discharging", str(summary["Blocks Discharging"]))
    rte_pct = plant_config["charge_eff"] * plant_config["discharge_eff"] * 100
    b3.metric("Round-Trip Efficiency", f"{rte_pct:.1f}%")
    if rev is not None:
        st.plotly_chart(create_revenue_bar(rev), use_container_width=True)
    avg_charge_price = float(
        (results_df.loc[results_df["Charge_Grid_ID"] > 0.01, "Charge_Grid_ID"] *
         results_df.loc[results_df["Charge_Grid_ID"] > 0.01, ["DAM_Price", "GDAM_Price", "RTM_Price_DA"]].min(axis=1)
         ).sum() / max(results_df["Charge_Grid_ID"].sum(), 1e-9)
    ) if results_df["Charge_Grid_ID"].sum() > 0 else 0.0
    avg_discharge_price = float(
        (results_df.loc[results_df["Discharge_ID"] > 0.01, "Discharge_ID"] *
         results_df.loc[results_df["Discharge_ID"] > 0.01, ["DAM_Price", "GDAM_Price", "RTM_Price_DA"]].max(axis=1)
         ).sum() / max(results_df["Discharge_ID"].sum(), 1e-9)
    ) if results_df["Discharge_ID"].sum() > 0 else 0.0
    st.caption(
        f"**Avg grid charge price:** ₹{avg_charge_price:.2f}/kWh  |  "
        f"**Avg discharge price:** ₹{avg_discharge_price:.2f}/kWh  |  "
        f"**Captured spread:** ₹{avg_discharge_price - avg_charge_price:.2f}/kWh"
    )

# ── Tab 5: Compare Engines ─────────────────────────────────────────────────
with tab_compare:
    st.caption("Runs both engines on the identical input data and compares net revenue.")
    if st.button("▶ Run Comparison", key="run_compare_btn"):
        with st.spinner("Running Rule-Based engine…"):
            rule_df = run_rule_based_dispatch(input_df, soc_initial_mwh, plant_config)
        rows = []
        rule_rev = compute_revenue(rule_df, plant_config) if compute_revenue else None
        rows.append({
            "Engine": "Rule-Based",
            "Net Revenue (₹)": rule_rev["Total Net Revenue"] if rule_rev else None,
            "Cycles Used": round(float(rule_df["Discharge_ID"].clip(lower=0).sum()) * BLOCK_HOURS / plant_config["batt_energy"], 2),
            "End SOC (MWh)": round(float(rule_df["SOC_end"].iloc[-1]), 1),
        })
        milp_df = None
        if LINOPY_AVAILABLE:
            with st.spinner("Running MILP engine…"):
                try:
                    milp_df, milp_solver_info = run_milp_dispatch(input_df, soc_initial_mwh, plant_config)
                    milp_rev = compute_revenue(milp_df, plant_config) if compute_revenue else None
                    rows.append({
                        "Engine": "MILP Optimized",
                        "Net Revenue (₹)": milp_rev["Total Net Revenue"] if milp_rev else None,
                        "Cycles Used": round(float(milp_df["Discharge_ID"].clip(lower=0).sum()) * BLOCK_HOURS / plant_config["batt_energy"], 2),
                        "End SOC (MWh)": round(float(milp_df["SOC_end"].iloc[-1]), 1),
                    })
                except Exception as exc:
                    st.error(f"MILP comparison failed: {exc}")
        else:
            st.warning("MILP scenarios unavailable (linopy not installed). Showing Rule-Based only.")
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        if len(comp_df) > 1:
            fig = go.Figure(go.Bar(
                x=comp_df["Engine"], y=comp_df["Net Revenue (₹)"], marker_color=["#2563eb", "#16a34a"],
                text=[f"₹{v:,.0f}" for v in comp_df["Net Revenue (₹)"]], textposition="outside",
            ))
            fig.update_layout(title="Net Revenue by Engine", yaxis_title="₹", height=380)
            st.plotly_chart(fig, use_container_width=True)
            uplift = comp_df["Net Revenue (₹)"].iloc[1] - comp_df["Net Revenue (₹)"].iloc[0]
            st.info(f"MILP captures **₹{uplift:,.0f}** more than the rule-based heuristic on this day.")

# ── Tab 6: Validation ──────────────────────────────────────────────────────
with tab_validation:
    if n_failed:
        st.error(f"{n_failed} validation check(s) failed.")
    else:
        st.success("All validation checks passed.")
    st.dataframe(validation_df, use_container_width=True, hide_index=True)

# ── Tab 7: Full Data ───────────────────────────────────────────────────────
with tab_data:
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    dl1, dl2 = st.columns(2)
    dl1.download_button("⬇ Download results (CSV)", data=results_df.to_csv(index=False).encode(),
                         file_name="batsimulator_results.csv", mime="text/csv")
    try:
        import openpyxl  # noqa
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="Results", index=False)
        dl2.download_button("⬇ Download results (Excel)", data=buf.getvalue(),
                             file_name="batsimulator_results.xlsx",
                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except ImportError:
        pass

# ── Tab 8: Strategy Comparison ─────────────────────────────────────────────
with tab_strategy:
    st.caption("Compares arbitrage outcomes under different grid-charging and engine assumptions.")
    if st.button("▶ Run Strategy Comparison", key="run_strategy_btn"):
        scenarios = [
            {"label": "Rule-Based | Grid Charging ON", "engine": "rule", "grid": True},
            {"label": "Rule-Based | Grid Charging OFF", "engine": "rule", "grid": False},
        ]
        if LINOPY_AVAILABLE:
            scenarios += [
                {"label": "MILP | Grid Charging ON", "engine": "milp", "grid": True},
                {"label": "MILP | Grid Charging OFF", "engine": "milp", "grid": False},
            ]
        strat_rows = []
        with st.spinner(f"Running {len(scenarios)} scenarios…"):
            for sc in scenarios:
                cfg = dict(plant_config)
                cfg["allow_grid_charging"] = sc["grid"]
                try:
                    if sc["engine"] == "rule":
                        df_res = run_rule_based_dispatch(input_df, soc_initial_mwh, cfg)
                    else:
                        df_res, _ = run_milp_dispatch(input_df, soc_initial_mwh, cfg)
                    sc_rev = compute_revenue(df_res, cfg) if compute_revenue else None
                    cycles = float(df_res["Discharge_ID"].clip(lower=0).sum()) * BLOCK_HOURS / cfg["batt_energy"]
                    strat_rows.append({
                        "Scenario": sc["label"],
                        "Engine": sc["engine"].upper(),
                        "Grid Charging": "✓" if sc["grid"] else "✗",
                        "Net Revenue (₹)": sc_rev["Total Net Revenue"] if sc_rev else None,
                        "Degradation Cost (₹)": sc_rev["Degradation Cost"] if sc_rev else None,
                        "Cycles Used": round(cycles, 2),
                        "End SOC (MWh)": round(float(df_res["SOC_end"].iloc[-1]), 1),
                    })
                except Exception as exc:
                    strat_rows.append({"Scenario": sc["label"], "Engine": sc["engine"].upper(),
                                        "Grid Charging": "✓" if sc["grid"] else "✗",
                                        "Net Revenue (₹)": None, "Degradation Cost (₹)": None,
                                        "Cycles Used": None, "End SOC (MWh)": None})
        strat_df = pd.DataFrame(strat_rows)
        st.dataframe(strat_df, use_container_width=True, hide_index=True)
        valid = strat_df[strat_df["Net Revenue (₹)"].notna()]
        if len(valid) > 0:
            fig = go.Figure(go.Bar(
                x=valid["Scenario"], y=valid["Net Revenue (₹)"], marker_color="#16a34a",
                text=[f"₹{v:,.0f}" for v in valid["Net Revenue (₹)"]], textposition="outside",
            ))
            fig.update_layout(title="Net Revenue by Strategy", yaxis_title="₹", height=420,
                               xaxis=dict(tickangle=-20), margin=dict(b=120))
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 9: Multi-Day Simulation ────────────────────────────────────────────
with tab_multiday:
    st.caption(
        "Runs the current plant configuration across several seeded demo days to show "
        "day-to-day variability in captured arbitrage revenue — a quick stand-in for "
        "historical backtesting when no real multi-day price file is available."
    )
    n_days = st.slider("Number of days to simulate", 3, 30, 7)
    base_seed = st.number_input("Base seed", min_value=1, max_value=99999, value=100, step=1)
    if st.button("▶ Run Multi-Day Simulation", key="run_multiday_btn"):
        daily_rows = []
        with st.spinner(f"Dispatching {n_days} days…"):
            for d in range(n_days):
                day_seed = int(base_seed) + d
                day_df = generate_demo_96block(seed=day_seed, plant_config=plant_config)
                if engine_is_milp:
                    try:
                        day_res, _ = run_milp_dispatch(day_df, soc_initial_mwh, plant_config)
                    except Exception:
                        day_res = run_rule_based_dispatch(day_df, soc_initial_mwh, plant_config)
                else:
                    day_res = run_rule_based_dispatch(day_df, soc_initial_mwh, plant_config)
                day_rev = compute_revenue(day_res, plant_config) if compute_revenue else None
                cycles = float(day_res["Discharge_ID"].clip(lower=0).sum()) * BLOCK_HOURS / plant_config["batt_energy"]
                daily_rows.append({
                    "Day": d + 1, "Seed": day_seed,
                    "Net Revenue (₹)": day_rev["Total Net Revenue"] if day_rev else None,
                    "Cycles Used": round(cycles, 2),
                    "Avg DAM (₹/kWh)": round(float(day_df["DAM_Price"].mean()), 2),
                })
        daily_df = pd.DataFrame(daily_rows)
        st.session_state["_bat_multiday"] = daily_df
    daily_df = st.session_state.get("_bat_multiday")
    if daily_df is not None:
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("Days simulated", str(len(daily_df)))
        hc2.metric("Total Net Revenue", f"₹{daily_df['Net Revenue (₹)'].sum():,.0f}")
        hc3.metric("Avg Cycles / Day", f"{daily_df['Cycles Used'].mean():.2f}")
        fig = go.Figure(go.Bar(x=daily_df["Day"], y=daily_df["Net Revenue (₹)"], marker_color="#16a34a"))
        fig.update_layout(title="Net Revenue by Day", xaxis_title="Day", yaxis_title="₹", height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(daily_df, use_container_width=True, hide_index=True)

# ── Tab 10: Annexure ────────────────────────────────────────────────────────
with tab_annexure:
    with st.expander("📥 Input Column Reference", expanded=True):
        input_ref = {
            "Block": "Block number (1–96). Auto-generated if absent.",
            "Time": "Block start time (HH:MM). Auto-generated if absent.",
            "DAM_Price": "Day-Ahead Market price forecast (₹/kWh).",
            "GDAM_Price": "Green Day-Ahead Market price forecast (₹/kWh).",
            "RTM_Price_DA": "Real-Time Market price forecast, Day-Ahead stage (₹/kWh).",
            "RTM_Price_ID": "Real-Time Market price, Intraday revision (₹/kWh).",
            "Temp_C": "Ambient temperature forecast (°C) — drives battery derating.",
            "SY_DA / SY_ID": "Solar yield forecast / intraday actual (MW). Required only if plant includes Solar.",
            "WY_DA / WY_ID": "Wind yield forecast / intraday actual (MW). Required only if plant includes Wind.",
        }
        st.dataframe(pd.DataFrame({"Column": list(input_ref), "Description": list(input_ref.values())}),
                     use_container_width=True, hide_index=True)
    with st.expander("📤 Output Column Reference", expanded=False):
        output_ref = {
            "Sell_Renew_ID": "Renewable MW sold directly to market (intraday actual).",
            "Charge_Renew_ID": "Renewable MW routed to charge the battery (intraday actual).",
            "Charge_Grid_ID": "Grid/market MW used to charge the battery (locked DA commitment).",
            "Discharge_ID": "Battery MW discharged to market (locked DA commitment).",
            "SOC_start / SOC_end": "Battery state of charge at start/end of block (MWh).",
            "Derate": "Temperature-driven power derating multiplier applied this block (0–1).",
        }
        st.dataframe(pd.DataFrame({"Column": list(output_ref), "Description": list(output_ref.values())}),
                     use_container_width=True, hide_index=True)
    with st.expander("📐 Revenue Formula Reference", expanded=False):
        st.markdown(f"""
**Unit conversion:** `Energy (kWh) = Power (MW) × 0.25 h × 1000 = Power × 250`

| Revenue Component | Formula (per block) |
|---|---|
| Renewable Direct-Sale Revenue | `Sell_Renew × best(DAM,GDAM,RTM_DA) × 250` |
| Battery Discharge Revenue | `Discharge × best(DAM,GDAM,RTM_DA) × 250` |
| Grid Charging Cost | `Charge_Grid × cheapest(DAM,GDAM,RTM_DA) × 250` |
| Degradation Cost | `(Charge_Renew + Charge_Grid + Discharge) × degradation_cost × 250` |
| **Total Net Revenue** | Sum of all above (costs subtracted) |

There is **no PPA revenue, shortfall penalty, or mandate term** in this model — 100% of revenue comes from market price arbitrage.
        """)
    with st.expander("⬇ Download CSV Templates", expanded=False):
        required_cols = list(REQUIRED_BASE_COLUMNS)
        if "Solar" in plant_config["plant_type"]:
            required_cols += ["SY_DA", "SY_ID"]
        if "Wind" in plant_config["plant_type"]:
            required_cols += ["WY_DA", "WY_ID"]
        blank_96 = pd.DataFrame([{
            "Block": i + 1, "Time": block_time_label(i),
            **{c: "" for c in required_cols},
        } for i in range(96)])
        st.download_button("⬇ Download blank 96-block input template (CSV)",
                            data=blank_96.to_csv(index=False).encode(),
                            file_name="batsimulator_template_blank.csv", mime="text/csv")
        st.download_button("⬇ Download demo 96-block sample data (CSV)",
                            data=demo_df.to_csv(index=False).encode(),
                            file_name="batsimulator_sample_data.csv", mime="text/csv")
