"""Observational Constraints page for the Alpha Ladder BH dashboard.

This is the MOST IMPORTANT page -- it shows why everything is unobservable.
Quantifies charge limits, effect scaling, PPN constraints, and the role of
the Cassini bound in ruling out the massless dilaton.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.formatting import fmt_decimal, fmt_percent
from app.components.charts import effects_scaling_chart, constraint_summary_chart

try:
    from observational_constraints import (
        estimate_wald_charge,
        schwinger_discharge_limit,
        dilaton_effects_at_realistic_q,
        constraint_summary_table,
        what_would_it_take,
        dark_charge_scenario,
    )
    _available = True
except ImportError:
    _available = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
a = 1.0 / math.sqrt(3.0)
G = 6.674298e-11
c = 2.99792458e8
M_sun = 1.989e30

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
params = render_sidebar()
M_geom = G * params["M_solar"] * M_sun / (c * c)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("Observational Constraints")
st.markdown(
    "Quantifying exactly why Gibbons-Maeda dilaton effects on black holes "
    "are undetectable by any conceivable instrument."
)

st.divider()

if not _available:
    st.warning(
        "Module `observational_constraints` not available. "
        "Cannot compute results."
    )
    st.stop()


# ---------------------------------------------------------------------------
# 1. Charge Limits
# ---------------------------------------------------------------------------
st.header("1. Charge Limits")

wald_data = estimate_wald_charge(params["M_solar"], 1.0)
schwinger_data = schwinger_discharge_limit(params["M_solar"])

c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        label="q_Wald",
        value=f"{wald_data['q_wald']:.2e}",
        delta="Equilibrium in B = 1 G",
        delta_color="off",
    )
with c2:
    st.metric(
        label="q_Schwinger",
        value=f"{schwinger_data['q_max_schwinger']:.2e}",
        delta="Pair production limit",
        delta_color="off",
    )
with c3:
    q_combined = min(wald_data["q_wald"], schwinger_data["q_max_schwinger"])
    st.metric(
        label="q_max (combined)",
        value=f"{q_combined:.2e}",
        delta="Effective upper bound",
        delta_color="off",
    )

with st.expander("Charge mechanism details"):
    st.markdown(
        f"**Wald mechanism** (Wald 1974): A rotating BH in an external "
        f"magnetic field B acquires equilibrium charge "
        f"Q_Wald = 2 B G M^2 / c^3.\n\n"
        f"- Q_Wald = {wald_data['Q_wald_coulombs']:.4e} C\n"
        f"- Q_ext = {wald_data['Q_ext_coulombs']:.4e} C\n"
        f"- q = Q/Q_ext = {wald_data['q_wald']:.4e}\n\n"
        f"**Schwinger discharge**: Above the critical field "
        f"E_cr = m_e^2 c^3 / (e hbar) = {schwinger_data['E_schwinger_V_per_m']:.4e} V/m, "
        f"electron-positron pairs are produced at the horizon.\n\n"
        f"- r_Schwarzschild = {schwinger_data['r_schwarzschild_m']:.4e} m\n"
        f"- Q_max = {schwinger_data['Q_schwinger_coulombs']:.4e} C\n"
        f"- q_max = {schwinger_data['q_max_schwinger']:.4e}"
    )

st.divider()


# ---------------------------------------------------------------------------
# 2. Effects Scaling (THE key chart)
# ---------------------------------------------------------------------------
st.header("2. Effects Scaling")

st.markdown(
    "All GM dilaton deviations from Schwarzschild scale as q\u00b2 at leading "
    "order. This log-log plot shows how observable deviations shrink as "
    "charge decreases toward realistic levels."
)

with st.spinner("Computing dilaton effects at multiple charge levels..."):
    effects_data = dilaton_effects_at_realistic_q()

fig = effects_scaling_chart(effects_data)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Scaling coefficients"):
    coeffs = effects_data["coefficients"]
    st.markdown(
        f"All deviations scale as delta ~ C * q\u00b2:\n\n"
        f"- C_shadow = {coeffs['C_shadow_pct_per_q2']:.4f} %/q\u00b2\n"
        f"- C_ISCO = {coeffs['C_isco_pct_per_q2']:.4f} %/q\u00b2\n"
        f"- C_eta = {coeffs['C_eta_pct_per_q2']:.4f} %/q\u00b2\n"
        f"- C_T_Hawking = {coeffs['C_T_pct_per_q2']:.4f} %/q\u00b2\n\n"
        f"At q ~ 10\u207b\u2079 (Wald limit), all deviations are < 10\u207b\u00b3\u2070 ppm."
    )

st.divider()


# ---------------------------------------------------------------------------
# 3. Cassini PPN Constraint (red card)
# ---------------------------------------------------------------------------
st.header("3. Cassini PPN Constraint")

a_sq = a * a
gamma_ppn_dilaton = (1.0 + a_sq) / (1.0 + 2.0 * a_sq)
gamma_deviation = gamma_ppn_dilaton - 1.0
cassini_precision = 2.3e-5
sigma_tension = abs(gamma_deviation) / cassini_precision

st.markdown(
    '<div class="bh-null">'
    "<p><strong>THE KILLER CONSTRAINT</strong></p>"
    "<p>The massless dilaton with a = 1/\u221a3 predicts a PPN parameter "
    "gamma that deviates from GR:</p>"
    f"<p>gamma_PPN = (1 + a\u00b2) / (1 + 2a\u00b2) = {gamma_ppn_dilaton:.6f}</p>"
    f"<p>Deviation from GR (gamma = 1): {gamma_deviation:.6f}</p>"
    f"<p>Cassini measured: gamma = 1 + (2.1 +/- 2.3) \u00d7 10\u207b\u2075</p>"
    f"<p><strong>Ruled out by {sigma_tension:.0f}\u03c3.</strong></p>"
    "<p>Even if black holes carried macroscopic charge, the massless dilaton "
    "is excluded by solar system tests. The dilaton MUST be massive (as "
    "predicted by Alpha Ladder flux stabilization), in which case it "
    "decouples and all BH solutions revert to standard GR.</p>"
    "</div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        label="\u03b3_PPN (dilaton)",
        value=f"{gamma_ppn_dilaton:.6f}",
    )
with c2:
    st.metric(
        label="Cassini precision",
        value="2.3 \u00d7 10\u207b\u2075",
    )
with c3:
    st.metric(
        label="Tension",
        value=f"{sigma_tension:.0f}\u03c3",
        delta="EXCLUDED",
        delta_color="inverse",
    )

st.divider()


# ---------------------------------------------------------------------------
# 4. Constraint Summary Table
# ---------------------------------------------------------------------------
st.header("4. Constraint Summary")

with st.spinner("Compiling constraint summary..."):
    summary_data = constraint_summary_table()

fig_summary = constraint_summary_chart(summary_data)
st.plotly_chart(fig_summary, use_container_width=True)

# Also show as formatted table
table_rows = []
for row in summary_data["rows"]:
    table_rows.append({
        "Source": row["source"],
        "Observable": row["observable"],
        "Precision": row["precision"],
        "q constraint": row["q_constraint"],
        "GM effect at q_max": row["gm_effect_at_qmax"],
    })

st.dataframe(table_rows, use_container_width=True, hide_index=True)

with st.expander("Summary interpretation"):
    st.markdown(summary_data["summary"])

st.divider()


# ---------------------------------------------------------------------------
# 5. Dark Sector Scenarios
# ---------------------------------------------------------------------------
st.header("5. Dark Sector Scenarios")

with st.spinner("Computing dark charge scenarios..."):
    dark_data = dark_charge_scenario()
    threshold_data = what_would_it_take()

tabs = st.tabs(["Dark Charge Accumulation", "What Would It Take?"])

with tabs[0]:
    st.markdown(
        "If dark matter carries a U(1)_dark gauge charge that does not "
        "couple to Standard Model fields, BHs could accumulate dark charge "
        "over cosmological timescales. This is speculative."
    )

    for label, result in dark_data["results"].items():
        with st.expander(label):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**N captured:** {result['N_captured']:.4e}")
                st.markdown(
                    f"**q (biased):** {result['q_biased']:.4e} "
                    f"(log10 = {result['log10_q_biased']:.1f})"
                )
            with c2:
                st.markdown(
                    f"**q (random walk):** {result['q_random']:.4e} "
                    f"(log10 = {result['log10_q_random']:.1f})"
                )
                st.markdown(
                    f"**Q_ext:** {result['Q_ext_C']:.4e} C"
                )

    st.markdown(
        '<div class="bh-caveat">'
        f"<p><strong>Caveat:</strong> {dark_data['caveat']}</p>"
        "</div>",
        unsafe_allow_html=True,
    )

with tabs[1]:
    st.markdown(
        "For each instrument, the minimum charge ratio q required "
        "for a detectable GM dilaton signal:"
    )

    det_rows = []
    for det in threshold_data["detectors"]:
        det_rows.append({
            "Detector": det["detector"],
            "Observable": det["observable"],
            "Precision (%)": f"{det['precision_pct']:.4f}",
            "q threshold": f"{det['q_threshold']:.4f}",
            "Orders above Wald": f"{det['orders_above_wald']:.0f}",
            "Likelihood": det["likelihood"],
        })

    st.dataframe(det_rows, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="bh-caveat">'
        f"<p>{threshold_data['conclusion']}</p>"
        "</div>",
        unsafe_allow_html=True,
    )
