import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from data.loader import (
    get_spot, get_hist, get_risk_free,
    get_expirations, get_option_chain, days_to_expiry
)
from pricing.black_scholes import (
    bs_call, bs_put, implied_vol_call, implied_vol_put, greeks_call
)

st.set_page_config(page_title="Options Risk Dashboard", layout="wide")

# ---------- sidebar ----------
st.sidebar.title("Inputs")
ticker = st.sidebar.text_input("Ticker", value="AAPL").upper()

try:
    spot = get_spot(ticker)
except Exception as e:
    st.error(f"Failed to load {ticker}: {e}")
    st.stop()
st.sidebar.markdown(f"**Spot:** {spot:.2f}")

exp_days = st.sidebar.slider("Days to Expiry (for demo smile/greeks)", 7, 365, 30)
T = exp_days / 365.0

rate_mode = st.sidebar.selectbox("Risk‑free rate", ["Auto (UST)", "Manual"])
manual_rate = st.sidebar.number_input("Manual rate (%)", value=4.50) if rate_mode == "Manual" else None
r = get_risk_free(T, manual_rate)

width_pct = st.sidebar.slider("Strike range (% of spot)", 60, 160, 120)
num_strikes = st.sidebar.slider("Number of strikes", 7, 41, 17, step=2)

st.title("Options Risk Dashboard")
st.caption("Black‑Scholes pricing • Implied Vol • Greeks • Smile / Term Structure • Live Option Chain & IV Heatmap")
st.write(f"Using **{ticker}**, **T ≈ {exp_days}d**, risk‑free **r = {r:.4f}** (cont.)")

# ---------- synthetic smile+term (demo) ----------
low_k = spot * (100 - (width_pct-100)) / 100.0
high_k = spot * (100 + (width_pct-100)) / 100.0
strikes = np.round(np.linspace(low_k, high_k, num_strikes), 2)

iv_true = 0.20 + 0.25 * ((strikes/spot - 1.0) ** 2)
mids = [bs_call(spot, K, T, r, iv) for K, iv in zip(strikes, iv_true)]
iv_hat = [implied_vol_call(p, spot, K, T, r) for p, K in zip(mids, strikes)]
df_smile = pd.DataFrame({"Strike": strikes, "Moneyness": strikes/spot, "IV": iv_hat})

c1, c2 = st.columns(2, gap="large")
with c1:
    fig_smile = px.line(df_smile, x="Moneyness", y="IV", markers=True,
                        title=f"Implied Vol Smile (~{exp_days}D)",
                        labels={"Moneyness":"K/S", "IV":"Implied Vol"})
    fig_smile.update_layout(yaxis_tickformat=".1%")
    st.plotly_chart(fig_smile, use_container_width=True)

with c2:
    Ts = np.array([7,14,21,30,45,60,90,120,180,365])/365.0
    atm_K = spot
    atm_mids = [bs_call(spot, atm_K, t, get_risk_free(t, manual_rate if rate_mode=="Manual" else None), 0.22) for t in Ts]
    iv_term = [implied_vol_call(p, spot, atm_K, t, get_risk_free(t, manual_rate if rate_mode=="Manual" else None))
               for p, t in zip(atm_mids, Ts)]
    fig_term = px.line(x=(Ts*365), y=iv_term, markers=True,
                       labels={"x":"Days to Expiry", "y":"Implied Vol"},
                       title="Term Structure (ATM IV)")
    fig_term.update_layout(yaxis_tickformat=".1%")
    st.plotly_chart(fig_term, use_container_width=True)

st.divider()

# ---------- greeks & scenario ----------
st.subheader("Greeks & Scenario")
sel_K = st.slider("Selected strike", float(strikes.min()), float(strikes.max()), float(np.round(spot, 2)))
base_iv = float(np.interp(sel_K, strikes, iv_hat))
left, right = st.columns(2)
with left:
    dS = st.slider("Spot shock (%)", -10, 10, 0)
with right:
    dIV = st.slider("IV shock (Δ percentage points)", -20, 20, 0)

S_scn = spot * (1 + dS/100.0)
iv_scn = max(0.0001, base_iv + dIV/100.0)

delta, gamma, vega, theta, rho = greeks_call(spot, sel_K, T, r, base_iv)
st.markdown(
    f"**Base IV:** {base_iv:.3f}  &nbsp;|&nbsp; "
    f"**Δ** {delta:.3f} &nbsp; **Γ** {gamma:.6f} &nbsp; **Vega** {vega:.2f} &nbsp; "
    f"**Θ** {theta:.2f} &nbsp; **ρ** {rho:.2f}"
)

from pricing.black_scholes import bs_call as _bs_call
base_px = _bs_call(spot, sel_K, T, r, base_iv)
scn_px  = _bs_call(S_scn, sel_K, T, r, iv_scn)
st.write(f"**Call price:** Base {base_px:.2f} → Scenario {scn_px:.2f} (**Δ {scn_px - base_px:+.2f}**)")

st.divider()

# ---------- LIVE OPTION CHAIN + IV HEATMAP ----------
st.subheader("Market IV Heatmap (yfinance) & Option Chain")

use_live = st.checkbox("Fetch live option chain (yfinance)", value=True)
max_exps = st.slider("How many near expiries to include", 1, 8, 4)
strike_band = st.slider("Only strikes within ±% of spot", 10, 50, 20)

if use_live:
    @st.cache_data(ttl=900)
    def _get_exps_cached(tkr):
        return get_expirations(tkr)

    exps = _get_exps_cached(ticker)
    if not exps:
        st.info("No listed options found for this ticker.")
    else:
        chosen_exps = exps[:max_exps]
        all_rows = []

        progress = st.progress(0.0, text="Loading chains…")
        for i, ex in enumerate(chosen_exps, start=1):
            try:
                df_chain = get_option_chain(ticker, ex)
                # focus on calls for heatmap (cleaner); you can switch to puts by changing type
                calls = df_chain[df_chain["type"] == "call"].copy()

                # mid price: prefer (bid+ask)/2; fallback lastPrice
                calls["mid"] = (calls["bid"].fillna(0) + calls["ask"].fillna(0)) / 2.0
                calls.loc[calls["mid"] <= 0, "mid"] = calls["lastPrice"]

                # time to expiry & rate per tenor
                T_days = days_to_expiry(calls["expiry"].iloc[0])
                if T_days <= 0:
                    continue
                T_years = T_days / 365.0
                r_tenor = get_risk_free(T_years, manual_rate if rate_mode=="Manual" else None)

                # keep only strikes near spot
                lo = spot * (1 - strike_band/100)
                hi = spot * (1 + strike_band/100)
                calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)].copy()

                # compute IV (robust)
                ivs = []
                for _, row in calls.iterrows():
                    K = float(row["strike"])
                    price = float(row["mid"])
                    try:
                        iv = implied_vol_call(price, spot, K, T_years, r_tenor)
                    except Exception:
                        iv = np.nan
                    ivs.append(iv)

                calls["iv_calc"] = ivs
                calls["expiry_days"] = round(T_days)
                calls["moneyness"] = calls["strike"] / spot
                all_rows.append(calls[["strike","expiry_days","iv_calc","bid","ask","lastPrice","moneyness"]])

            except Exception as e:
                st.warning(f"Failed to load {ex}: {e}")

            progress.progress(i/len(chosen_exps))

        progress.empty()

        if all_rows:
            market_df = pd.concat(all_rows, ignore_index=True).dropna(subset=["iv_calc"])
            if not market_df.empty:
                pivot = market_df.pivot_table(index="strike", columns="expiry_days", values="iv_calc", aggfunc="mean")
                pivot = pivot.sort_index().sort_index(axis=1)

                fig_hm = px.imshow(
                    pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    color_continuous_scale="Viridis",
                    aspect="auto",
                    labels={"x":"Days to Expiry", "y":"Strike", "color":"IV"},
                    title="Implied Volatility Heatmap (Calls)"
                )
                fig_hm.update_layout(coloraxis_colorbar=dict(tickformat=".0%"))
                st.plotly_chart(fig_hm, use_container_width=True)

                with st.expander("Show option chain (calls)"):
                    st.dataframe(
                        market_df.sort_values(["expiry_days","strike"]),
                        use_container_width=True,
                        height=400
                    )
            else:
                st.info("Could not compute any implied vols from the chain.")
        else:
            st.info("No option data returned for selected expiries.")

with st.expander("Historical price (context)"):
    try:
        hist = get_hist(ticker, "1y")
        st.line_chart(hist)
    except Exception as e:
        st.info(f"Could not load history: {e}")

st.caption("Notes: IV is computed from market mids via Black‑Scholes. Rates pulled from ^IRX/^TNX and converted to continuous compounding.")