# BESS Dispatch Optimisation Demo

A PwC-styled Streamlit demonstration of a constrained, degradation-aware BESS dispatch optimiser for an illustrative Indian private-sector solar + storage asset.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Demo flow

1. Start with the five-step methodology: Frame, Forecast, Optimise, Validate, Execute + Learn.
2. Change battery power, duration, SOC limits, efficiency and degradation cost.
3. Change the scenario seed to simulate a different weather and price day.
4. Compare the constrained optimiser with the rule-based counterfactual.
5. Use the rolling-horizon chart and decision packet to explain a recommended action.
6. Download the resulting interval schedule as CSV.

## Data and scope

All operating data are synthetic and illustrative. The optimisation is functional and uses SciPy's HiGHS linear programming solver. This is a demonstration, not a production dispatch system or market forecast.
