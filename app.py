import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linprog

st.set_page_config(page_title="BESS Dispatch Optimisation", page_icon="🔋", layout="wide")

PWC = {
    "yellow": "#FFB600", "amber": "#EB8C00", "orange": "#D04A02",
    "red": "#E0301E", "rose": "#DB536A", "black": "#000000",
    "dark": "#2D2D2D", "grey": "#6D6E71", "light": "#F2F2F2", "white": "#FFFFFF"
}

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{background: #fff;}}
[data-testid="stSidebar"] {{background: #f6f6f6; border-right: 1px solid #ddd;}}
.block-container {{padding-top: 1.2rem; padding-bottom: 1.5rem;}}
h1, h2, h3 {{color: {PWC['black']}; font-family: Arial, sans-serif;}}
.pwc-top {{height: 7px; background: linear-gradient(90deg, {PWC['yellow']} 0 25%, {PWC['amber']} 25% 45%, {PWC['orange']} 45% 70%, {PWC['red']} 70% 100%); margin: -1.2rem -4rem 1rem -4rem;}}
.kicker {{color:{PWC['orange']}; font-size:0.72rem; font-weight:700; letter-spacing:1.5px;}}
.hero {{font-size:2.05rem; font-weight:700; line-height:1.1; margin:0.2rem 0;}}
.sub {{font-size:0.98rem; color:{PWC['grey']}; max-width:1050px;}}
.metric-card {{background:#fff; border:1px solid #ddd; border-top:4px solid {PWC['orange']}; padding:14px 16px; min-height:112px;}}
.metric-label {{font-size:0.72rem; color:{PWC['grey']}; font-weight:700; letter-spacing:0.8px;}}
.metric-value {{font-size:1.65rem; font-weight:700; color:{PWC['black']};}}
.metric-note {{font-size:0.75rem; color:{PWC['grey']};}}
.method {{display:flex; gap:8px; align-items:stretch; margin:14px 0 20px 0;}}
.method-step {{flex:1; padding:10px; border:1px solid #ddd; background:#fff; text-align:center;}}
.method-num {{width:26px;height:26px;border-radius:50%;background:{PWC['orange']};color:#fff;margin:auto;padding-top:3px;font-weight:700;}}
.method-title {{font-size:0.75rem;font-weight:700;margin-top:7px;}}
.method-copy {{font-size:0.68rem;color:{PWC['grey']};margin-top:4px;}}
.packet {{background:{PWC['dark']}; color:white; padding:18px; border-radius:4px;}}
.packet-label {{color:{PWC['yellow']}; font-size:0.7rem; font-weight:700; letter-spacing:1px;}}
.packet-value {{font-size:1.45rem; font-weight:700; margin-bottom:10px;}}
.small {{font-size:0.75rem; color:#d0d0d0;}}
.notice {{background:#fff8ef;border-left:5px solid {PWC['amber']};padding:10px 14px;font-size:0.82rem;}}
</style>
<div class="pwc-top"></div>
""", unsafe_allow_html=True)

@st.cache_data
def generate_data(seed: int, steps: int = 96):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-03", periods=steps, freq="15min")
    h = np.arange(steps) / 4
    solar_shape = np.maximum(0, np.sin(np.pi * (h - 6) / 12))
    solar = np.maximum(0, 82 * solar_shape * (0.86 + 0.10*np.sin(h/2.7)) + rng.normal(0, 2.1, steps))
    load = 46 + 7*np.sin(2*np.pi*(h-7)/24) + 15*np.exp(-0.5*((h-20)/2.1)**2) + rng.normal(0,1.4,steps)
    price = 4.1 + 0.7*np.sin(2*np.pi*(h-15)/24) + 2.7*np.exp(-0.5*((h-20)/1.8)**2) - 1.2*solar_shape + rng.normal(0,0.18,steps)
    price = np.clip(price, 1.8, 9.5)
    export_limit = np.full(steps, 68.0)
    export_limit[(h >= 11.5) & (h <= 14.5)] = 50.0
    curtailment = np.maximum(0, solar - export_limit)
    delivery = np.zeros(steps)
    delivery[(h >= 18) & (h < 22)] = 38.0
    return pd.DataFrame({"timestamp":idx,"hour":h,"solar_mw":solar,"load_mw":load,"price_inr_kwh":price,
                         "export_limit_mw":export_limit,"potential_curtailment_mw":curtailment,"contract_mw":delivery})

def optimise(df, power_mw, energy_mwh, initial_soc, min_soc, max_soc, rte, degradation, reserve_soc, risk_margin, contract_penalty):
    n=len(df); dt=0.25; eta=np.sqrt(rte)
    # vars: charge n, discharge n, soc n+1, shortfall n
    N=4*n+1
    ch=np.arange(0,n); dis=np.arange(n,2*n); soc=np.arange(2*n,3*n+1); short=np.arange(3*n+1,4*n+1)
    c=np.zeros(N)
    c[ch]=(df.price_inr_kwh.values*1000*dt/eta)+(degradation*1000*dt)
    c[dis]=-(df.price_inr_kwh.values*1000*dt*eta)+(degradation*1000*dt)
    c[short]=contract_penalty*1000*dt
    Aeq=[]; beq=[]
    row=np.zeros(N); row[soc[0]]=1; Aeq.append(row); beq.append(initial_soc/100*energy_mwh)
    for t in range(n):
        row=np.zeros(N); row[soc[t+1]]=1; row[soc[t]]=-1; row[ch[t]]=-eta*dt; row[dis[t]]=dt/eta
        Aeq.append(row); beq.append(0)
    Aub=[]; bub=[]
    # Contract: discharge + shortfall >= contract, represented negative <=
    for t in range(n):
        row=np.zeros(N); row[dis[t]]=-1; row[short[t]]=-1; Aub.append(row); bub.append(-df.contract_mw.iloc[t])
        # Grid charging limited by renewable surplus plus import headroom, conservative risk derate
        row=np.zeros(N); row[ch[t]]=1
        available=max(0, power_mw*(1-risk_margin/100))
        Aub.append(row); bub.append(available)
    bounds=[]
    for _ in range(n): bounds.append((0,power_mw))
    for _ in range(n): bounds.append((0,power_mw))
    effective_min=max(min_soc,reserve_soc)
    for t in range(n+1): bounds.append((effective_min/100*energy_mwh,max_soc/100*energy_mwh))
    for _ in range(n): bounds.append((0,None))
    res=linprog(c,A_ub=np.array(Aub),b_ub=np.array(bub),A_eq=np.array(Aeq),b_eq=np.array(beq),bounds=bounds,method="highs")
    if not res.success: return None,res.message
    x=res.x; out=df.copy(); out["charge_mw"]=x[ch]; out["discharge_mw"]=x[dis]; out["soc_pct"]=x[soc[1:]]/energy_mwh*100; out["shortfall_mw"]=x[short]
    out["net_dispatch_mw"]=out.discharge_mw-out.charge_mw
    out["gross_revenue_inr"]=(out.discharge_mw*eta-out.charge_mw/eta)*out.price_inr_kwh*1000*dt
    out["degradation_cost_inr"]=(out.charge_mw+out.discharge_mw)*degradation*1000*dt
    out["penalty_inr"]=out.shortfall_mw*contract_penalty*1000*dt
    out["net_value_inr"]=out.gross_revenue_inr-out.degradation_cost_inr-out.penalty_inr
    return out,None

def rule_based(df,power_mw,energy_mwh,initial_soc,min_soc,max_soc,rte,deg,reserve_soc,charge_threshold,discharge_threshold):
    dt=.25; eta=np.sqrt(rte); soc=initial_soc/100*energy_mwh; rows=[]; minimum=max(min_soc,reserve_soc)/100*energy_mwh; maximum=max_soc/100*energy_mwh
    for _,r in df.iterrows():
        ch=dis=0.0
        if r.price_inr_kwh<=charge_threshold: ch=min(power_mw,(maximum-soc)/(eta*dt))
        if r.price_inr_kwh>=discharge_threshold or r.contract_mw>0: dis=min(power_mw,(soc-minimum)*eta/dt)
        dis=min(dis,max(r.contract_mw,dis))
        soc=np.clip(soc+ch*eta*dt-dis/eta*dt,minimum,maximum)
        gross=(dis*eta-ch/eta)*r.price_inr_kwh*1000*dt; wear=(ch+dis)*deg*1000*dt
        rows.append((ch,dis,soc/energy_mwh*100,gross-wear))
    o=df.copy(); o[["charge_mw","discharge_mw","soc_pct","net_value_inr"]]=rows; o["net_dispatch_mw"]=o.discharge_mw-o.charge_mw
    return o

with st.sidebar:
    st.markdown("### Scenario controls")
    seed=st.slider("Weather and market scenario",1,50,17)
    st.markdown("#### Battery")
    power=st.slider("Rated power (MW)",20,200,100,5)
    duration=st.select_slider("Duration (hours)",options=[1,2,3,4],value=2)
    energy=power*duration
    initial_soc=st.slider("Initial SOC (%)",10,90,55)
    min_soc=st.slider("Minimum SOC (%)",0,40,10)
    max_soc=st.slider("Maximum SOC (%)",60,100,95)
    reserve_soc=st.slider("Reserve SOC (%)",0,40,15)
    rte=st.slider("Round-trip efficiency (%)",70,95,88)/100
    degradation=st.slider("Marginal degradation cost (₹/kWh)",0.0,2.5,0.65,0.05)
    st.markdown("#### Commercial and risk")
    penalty=st.slider("Contract shortfall cost (₹/kWh)",5.0,25.0,14.0,0.5)
    risk=st.slider("Uncertainty reserve margin (%)",0,25,8)
    charge_t=st.slider("Rule-based charge threshold (₹/kWh)",1.5,5.0,3.2,0.1)
    discharge_t=st.slider("Rule-based discharge threshold (₹/kWh)",4.0,9.0,6.0,0.1)
    horizon=st.slider("Visible horizon (hours)",4,24,24)

st.markdown('<div class="kicker">DISPATCH OPTIMISATION METHODOLOGY</div>',unsafe_allow_html=True)
st.markdown('<div class="hero">Convert uncertain forecasts into executable, degradation-aware dispatch decisions</div>',unsafe_allow_html=True)
st.markdown('<div class="sub">Interactive demonstration for an illustrative Indian private-sector solar + BESS asset. Adjust assumptions in the sidebar to see the optimiser re-frame the dispatch plan.</div>',unsafe_allow_html=True)
st.markdown('<div class="notice"><b>Demo data:</b> All load, price, renewable, contract and asset observations are synthetic and illustrative. They are designed to behave realistically but are not actual market or client data.</div>',unsafe_allow_html=True)

st.markdown("""<div class="method">
<div class="method-step"><div class="method-num">1</div><div class="method-title">FRAME</div><div class="method-copy">Objectives and guardrails</div></div>
<div class="method-step"><div class="method-num">2</div><div class="method-title">FORECAST</div><div class="method-copy">Price, RE and demand paths</div></div>
<div class="method-step"><div class="method-num">3</div><div class="method-title">OPTIMISE</div><div class="method-copy">Charge, discharge, hold</div></div>
<div class="method-step"><div class="method-num">4</div><div class="method-title">VALIDATE</div><div class="method-copy">SOC, power and warranty</div></div>
<div class="method-step"><div class="method-num">5</div><div class="method-title">EXECUTE + LEARN</div><div class="method-copy">Explain and reconcile</div></div>
</div>""",unsafe_allow_html=True)

df=generate_data(seed)
opt,err=optimise(df,power,energy,initial_soc,min_soc,max_soc,rte,degradation,reserve_soc,risk,penalty)
if err:
    st.error(f"Optimisation infeasible: {err}. Reduce reserve SOC or contract obligation.")
    st.stop()
base=rule_based(df,power,energy,initial_soc,min_soc,max_soc,rte,degradation,reserve_soc,charge_t,discharge_t)
view=int(horizon*4); optv=opt.iloc[:view]; basev=base.iloc[:view]

opt_value=opt.net_value_inr.sum()/1e5; base_value=base.net_value_inr.sum()/1e5; uplift=opt_value-base_value
throughput=(opt.charge_mw.sum()+opt.discharge_mw.sum())*.25
cycles=throughput/(2*energy)
contract_delivery=np.minimum(opt.discharge_mw,opt.contract_mw).sum()*.25
contract_required=opt.contract_mw.sum()*.25
compliance=100 if contract_required==0 else min(100,100*contract_delivery/contract_required)

cols=st.columns(5)
metrics=[("OPTIMISED VALUE",f"₹{opt_value:,.2f} lakh",f"₹{uplift:,.2f} lakh vs rule-based"),("ENDING SOC",f"{opt.soc_pct.iloc[-1]:.1f}%",f"Reserve floor {max(min_soc,reserve_soc)}%"),("CONTRACT DELIVERY",f"{compliance:.1f}%",f"{contract_delivery:.1f} of {contract_required:.1f} MWh"),("THROUGHPUT",f"{throughput:,.0f} MWh",f"{cycles:.2f} equivalent cycles"),("WEAR COST",f"₹{opt.degradation_cost_inr.sum()/1e5:,.2f} lakh",f"₹{degradation:.2f}/kWh marginal")]
for c,(lab,val,note) in zip(cols,metrics): c.markdown(f'<div class="metric-card"><div class="metric-label">{lab}</div><div class="metric-value">{val}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)

st.markdown("### Rolling-horizon dispatch")
fig=make_subplots(specs=[[{"secondary_y":True}]])
fig.add_trace(go.Scatter(x=optv.timestamp,y=optv.price_inr_kwh,name="Price ₹/kWh",line=dict(color=PWC['black'],width=2)),secondary_y=True)
fig.add_trace(go.Bar(x=optv.timestamp,y=-optv.charge_mw,name="Charge MW",marker_color=PWC['yellow'],opacity=.85),secondary_y=False)
fig.add_trace(go.Bar(x=optv.timestamp,y=optv.discharge_mw,name="Discharge MW",marker_color=PWC['orange'],opacity=.9),secondary_y=False)
fig.add_trace(go.Scatter(x=optv.timestamp,y=optv.contract_mw,name="Contract MW",line=dict(color=PWC['red'],dash="dot",width=2)),secondary_y=False)
fig.add_trace(go.Scatter(x=optv.timestamp,y=optv.soc_pct,name="SOC %",line=dict(color=PWC['rose'],width=3)),secondary_y=True)
fig.update_layout(height=460,barmode="relative",margin=dict(l=25,r=25,t=35,b=20),legend=dict(orientation="h",y=1.1),plot_bgcolor="white",hovermode="x unified")
fig.update_yaxes(title_text="Power (MW)",gridcolor="#e5e5e5",secondary_y=False)
fig.update_yaxes(title_text="Price / SOC",secondary_y=True)
st.plotly_chart(fig,use_container_width=True)

left,right=st.columns([1.55,1])
with left:
    st.markdown("### Forecast and constraint context")
    f2=make_subplots(specs=[[{"secondary_y":True}]])
    f2.add_trace(go.Scatter(x=optv.timestamp,y=optv.solar_mw,name="Solar forecast",fill="tozeroy",line=dict(color=PWC['yellow'])),secondary_y=False)
    f2.add_trace(go.Scatter(x=optv.timestamp,y=optv.load_mw,name="Site load",line=dict(color=PWC['dark'],width=2)),secondary_y=False)
    f2.add_trace(go.Scatter(x=optv.timestamp,y=optv.export_limit_mw,name="Grid export limit",line=dict(color=PWC['red'],dash="dash")),secondary_y=False)
    f2.add_trace(go.Scatter(x=optv.timestamp,y=optv.price_inr_kwh,name="Price",line=dict(color=PWC['orange'])),secondary_y=True)
    f2.update_layout(height=390,margin=dict(l=20,r=20,t=25,b=20),legend=dict(orientation="h",y=1.12),plot_bgcolor="white",hovermode="x unified")
    f2.update_yaxes(title_text="MW",gridcolor="#ececec",secondary_y=False); f2.update_yaxes(title_text="₹/kWh",secondary_y=True)
    st.plotly_chart(f2,use_container_width=True)
with right:
    now_idx=int(np.argmax(optv.discharge_mw.values)) if optv.discharge_mw.max()>0.1 else int(np.argmax(optv.charge_mw.values))
    row=optv.iloc[now_idx]
    action="DISCHARGE" if row.discharge_mw>0.1 else ("CHARGE" if row.charge_mw>0.1 else "HOLD")
    mw=max(row.discharge_mw,row.charge_mw)
    why=("Price and delivery value clear efficiency, degradation and risk thresholds." if action=="DISCHARGE" else "Low-price interval creates value while preserving the reserve floor." if action=="CHARGE" else "No action clears the lifecycle-value threshold.")
    bind="Reserve SOC / energy balance" if row.soc_pct<=max(min_soc,reserve_soc)+2 else "Power or price threshold"
    confidence="HIGH" if risk<=10 else "MODERATE"
    st.markdown("### Decision packet")
    st.markdown(f"""<div class="packet">
    <div class="packet-label">ACTION</div><div class="packet-value">{action}</div>
    <div class="small">{mw:.1f} MW | {row.timestamp.strftime('%d %b, %H:%M')} | Expected SOC {row.soc_pct:.1f}%</div><br>
    <div class="packet-label">EXPECTED INTERVAL VALUE</div><div class="packet-value">₹{row.net_value_inr/1000:,.1f}k</div>
    <div class="packet-label">CONFIDENCE</div><div class="packet-value">{confidence}</div>
    <div class="packet-label">WHY NOW</div><div class="small">{why}</div><br>
    <div class="packet-label">BINDING CONSTRAINT</div><div class="small">{bind}</div><br>
    <div class="packet-label">OPERATOR CONTROL</div><div class="small">Accept, amend or reject. Reason is retained for audit and model learning.</div>
    </div>""",unsafe_allow_html=True)

st.markdown("### Optimiser versus rule-based counterfactual")
comp=pd.DataFrame({"Method":["Rule-based","Optimiser"],"Net value (₹ lakh)":[base_value,opt_value],"Ending SOC (%)":[base.soc_pct.iloc[-1],opt.soc_pct.iloc[-1]],"Equivalent cycles":[(base.charge_mw.sum()+base.discharge_mw.sum())*.25/(2*energy),cycles]})
c1,c2=st.columns([1.25,1])
with c1:
    f3=go.Figure([go.Bar(x=comp.Method,y=comp["Net value (₹ lakh)"],marker_color=[PWC['grey'],PWC['orange']],text=comp["Net value (₹ lakh)"].map(lambda x:f"₹{x:.2f}L"),textposition="outside")])
    f3.update_layout(height=330,margin=dict(l=20,r=20,t=25,b=20),plot_bgcolor="white",yaxis_title="Net value (₹ lakh)")
    st.plotly_chart(f3,use_container_width=True)
with c2:
    st.dataframe(comp.set_index("Method").style.format({"Net value (₹ lakh)":"{:.2f}","Ending SOC (%)":"{:.1f}","Equivalent cycles":"{:.2f}"}),use_container_width=True)
    st.download_button("Download optimised schedule",opt.to_csv(index=False).encode(),"optimised_dispatch_schedule.csv","text/csv",use_container_width=True)

with st.expander("Methodology and model assumptions"):
    st.markdown("""
**Objective:** maximise energy-market and contracted-delivery value, less charging energy, efficiency losses, marginal degradation and contract shortfall cost.

**Constraints:** interval energy balance, charge/discharge power, minimum and maximum SOC, reserve SOC, round-trip efficiency and contract delivery. The demonstration uses a linear programme solved by SciPy HiGHS.

**Counterfactual:** a transparent threshold-based dispatcher charges below a user-defined price and discharges above a user-defined price or during a contracted delivery window.

**Important limitation:** this demonstration does not model cell-level electrochemistry, network power flow, taxes, exchange fees, DSM regulations, bid granularity, battery augmentation or an actual PPA/BESPA. Those would be configured during a client pilot.
    """)
