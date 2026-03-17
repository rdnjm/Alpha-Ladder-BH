"""Quasinormal Modes page.

Computes QNM frequencies for the Gibbons-Maeda dilaton black hole using
3rd-order WKB approximation, compares with Schwarzschild and evaluates
detectability by LIGO, Einstein Telescope, and LISA.

The dilaton coupling a = 1/\u221a3, \u03c9 = 0.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.formatting import fmt_decimal, fmt_percent
from app.components.charts import qnm_spectrum_chart

try:
    from quasinormal_modes import (
        compute_qnm_spectrum,
        compare_with_ligo,
        summarize_qnm_analysis,
    )
    _available = True
except ImportError:
    _available = False

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
a = 1.0 / math.sqrt(3.0)
G = 6.674298e-11
c_light = 2.99792458e8
M_sun = 1.989e30

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
params = render_sidebar()
M_geom = G * params["M_solar"] * M_sun / (c_light * c_light)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("Quasinormal Modes")
st.markdown(
    "Damped oscillation frequencies of perturbed Gibbons-Maeda black holes, "
    "computed via 3rd-order WKB (Schutz-Will / Iyer-Will / Konoplya). "
    "The dilaton coupling a = 1/\u221a3 modifies the effective potential "
    "and shifts QNM frequencies relative to Schwarzschild."
)
st.divider()

if not _available:
    st.warning("Module `quasinormal_modes` not available. Install it and restart.")
    st.stop()


# ===================================================================
# 1. QNM Spectrum
# ===================================================================
st.header("1. QNM Spectrum")

@st.cache_data
def _cached_qnm_spectrum(m_solar, q):
    return compute_qnm_spectrum(M_solar=m_solar, qm_ratio=q)

spectrum = _cached_qnm_spectrum(params["M_solar"], params["q"])

# Pass spectrum data directly to chart (expects dict-of-dicts format)
fig = qnm_spectrum_chart(spectrum)
st.plotly_chart(fig, use_container_width=True)

# Mode table
st.markdown("**Mode frequencies (l = 2, 3, 4; n = 0 fundamental):**")
table_rows = []
for l_val in [2, 3, 4]:
    key = f"l={l_val}"
    mode = spectrum.get("modes", {}).get(key, {})
    if "error" in mode:
        table_rows.append({
            "l": l_val,
            "n": 0,
            "omega_R (Hz)": "error",
            "omega_I (Hz)": "error",
            "Q-factor": "N/A",
            "delta vs Schwarz (%)": "N/A",
        })
        continue

    gm = mode.get("GM", {})
    sch = mode.get("Schwarzschild", {})
    delta_R = mode.get("delta_omega_R_frac")

    f_Hz_gm = gm.get("f_Hz", 0.0)
    # Compute omega_I in Hz
    omega_I_M = gm.get("omega_I_M", 0.0)
    omega_I_phys = omega_I_M / M_geom if M_geom > 0 else 0.0
    f_I_Hz = omega_I_phys * c_light / (2.0 * math.pi)

    # Q-factor = omega_R / (2 * |omega_I|)
    q_factor = abs(gm.get("omega_R", 0.0)) / (2.0 * abs(gm.get("omega_I", 1e-30)))

    delta_pct = (delta_R * 100.0) if delta_R is not None else 0.0

    table_rows.append({
        "l": l_val,
        "n": 0,
        "omega_R (Hz)": f"{f_Hz_gm:.2f}",
        "omega_I (Hz)": f"{f_I_Hz:.2f}",
        "Q-factor": f"{q_factor:.2f}",
        "delta vs Schwarz (%)": f"{delta_pct:.4f}",
    })

if table_rows:
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

st.divider()


# ===================================================================
# 2. LIGO / ET / LISA Comparison
# ===================================================================
st.header("2. LIGO / ET / LISA Comparison")

@st.cache_data
def _cached_ligo(m_solar):
    return compare_with_ligo(M_solar=m_solar)

ligo_data = _cached_ligo(params["M_solar"])

# Scan results table
scan = ligo_data.get("scan_results", [])
if scan:
    st.markdown(f"**Charge scan for M = {params['M_solar']:.1f} M_sun (l = 2, n = 0):**")
    scan_table = []
    for entry in scan:
        if "error" in entry:
            continue
        q_val = entry.get("q", 0.0)
        f_gm = entry.get("f_Hz_GM")
        f_sch = entry.get("f_Hz_Schwarz")
        delta_f = entry.get("delta_f_frac")
        delta_f_hz = entry.get("delta_f_Hz")

        scan_table.append({
            "q": f"{q_val:.2f}",
            "f_GM (Hz)": f"{f_gm:.2f}" if f_gm else "N/A",
            "f_Schwarz (Hz)": f"{f_sch:.2f}" if f_sch else "N/A",
            "delta_f/f (%)": f"{delta_f * 100:.4f}" if delta_f else "N/A",
            "delta_f (Hz)": f"{delta_f_hz:.4f}" if delta_f_hz else "N/A",
        })
    if scan_table:
        st.dataframe(scan_table, use_container_width=True, hide_index=True)

# Detector comparison
st.markdown("**Detector capabilities:**")

ligo_ctx = ligo_data.get("ligo_context", {})
future = ligo_data.get("future_detectors", {})

detector_table = [
    {
        "Detector": "LIGO O4/O5",
        "Frequency band": "10 - 5000 Hz",
        "Resolution": f"~{ligo_ctx.get('typical_freq_resolution_Hz', 10):.0f} Hz",
        "Can detect shift?": "No (sub-percent shifts below resolution)",
    },
    {
        "Detector": "Einstein Telescope",
        "Frequency band": "1 - 10000 Hz",
        "Resolution": "~0.1 Hz (projected)",
        "Can detect shift?": "Marginal (requires q > 0.3 and high SNR)",
    },
    {
        "Detector": "LISA",
        "Frequency band": "0.1 mHz - 100 mHz",
        "Resolution": "~0.01 mHz at SNR > 100",
        "Can detect shift?": "Possible for massive BH mergers (10^5-10^7 M_sun)",
    },
]
st.dataframe(detector_table, use_container_width=True, hide_index=True)

st.markdown(
    '<div class="bh-null">'
    "<p><strong>Key message:</strong> The QNM frequency shift scales as "
    "delta_omega ~ q\u00b2 a\u00b2, so for q \u2248 0 (astrophysical BHs) the deviation "
    "from GR is undetectable. Even at q = 0.5 the shift is only a few percent, "
    "requiring next-generation detectors with sub-percent precision on "
    "ringdown frequencies.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


# ===================================================================
# 3. Summary
# ===================================================================
st.header("3. Summary and Assessment")

@st.cache_data
def _cached_summary():
    return summarize_qnm_analysis()

summary = _cached_summary()

# WKB validation
wkb_val = summary.get("wkb_validation", {}).get("schwarzschild_l2_n0", {})
if wkb_val:
    with st.expander("WKB method validation (Schwarzschild l=2, n=0)"):
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Exact omega_R * M",
                f"{wkb_val.get('exact_omega_R_M', 0):.4f}",
            )
            st.metric(
                "WKB omega_R * M",
                f"{wkb_val.get('wkb_omega_R_M', 0):.4f}",
            )
            err_R = wkb_val.get("error_omega_R")
            if err_R is not None:
                st.metric("Error (omega_R)", fmt_percent(err_R, 2))
        with c2:
            st.metric(
                "Exact omega_I * M",
                f"{wkb_val.get('exact_omega_I_M', 0):.4f}",
            )
            st.metric(
                "WKB omega_I * M",
                f"{wkb_val.get('wkb_omega_I_M', 0):.4f}",
            )
            err_I = wkb_val.get("error_omega_I")
            if err_I is not None:
                st.metric("Error (omega_I)", fmt_percent(err_I, 2))

# Key findings
key_findings = summary.get("key_findings", [])
if key_findings:
    st.markdown("**Key findings:**")
    for finding in key_findings:
        st.markdown(f"- {finding}")

# Honest assessment
honest = summary.get("honest_assessment", [])
if honest:
    st.markdown("**Honest assessment:**")
    for point in honest:
        st.markdown(f"- {point}")

# Theoretical significance
sig = summary.get("theoretical_significance", "")
if sig:
    st.markdown(
        '<div class="bh-positive">'
        f"<p><strong>Theoretical significance:</strong> {sig}</p>"
        "</div>",
        unsafe_allow_html=True,
    )
