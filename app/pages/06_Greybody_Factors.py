"""Greybody & Hawking Radiation page for the Alpha Ladder BH dashboard.

Presents Hawking temperature, emission spectra, greybody factors, total
power, and the dilaton emission channel for Gibbons-Maeda dilaton black
holes with the Alpha Ladder coupling a = 1/\u221a3.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.formatting import fmt_decimal, fmt_percent
from app.components.charts import hawking_spectrum_chart, greybody_scan_chart

try:
    from greybody_factors import (
        hawking_temperature,
        effective_potential_peak,
        greybody_factor,
        hawking_spectrum,
        total_power,
        greybody_scan,
        dilaton_emission_channel,
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
st.title("Greybody & Hawking Radiation")
st.markdown(
    "Hawking temperature, greybody factors, and emission spectra for the "
    "Gibbons-Maeda dilaton black hole with a = 1/\u221a3."
)

st.divider()

if not _available:
    st.warning(
        "Module `greybody_factors` not available. Cannot compute results."
    )
    st.stop()


# ---------------------------------------------------------------------------
# 1. Hawking Temperature
# ---------------------------------------------------------------------------
st.header("1. Hawking Temperature")

temp_data = hawking_temperature(M_geom, params["q"])

if "error" in temp_data:
    st.error(f"Temperature computation failed: {temp_data['error']}")
else:
    T_kelvin = temp_data["T_H_kelvin"]
    T_schwarz = temp_data["T_schwarz_kelvin"]
    ratio_T = temp_data["ratio_T_H_over_T_schwarz"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="T_H (K)",
            value=f"{T_kelvin:.4e}",
        )
    with c2:
        st.metric(
            label="T_Schwarzschild (K)",
            value=f"{T_schwarz:.4e}",
        )
    with c3:
        st.metric(
            label="T_H / T_Schwarz",
            value=f"{ratio_T:.6f}",
        )

    with st.expander("Temperature physics"):
        st.markdown(
            "The Hawking temperature for the GM metric is:\n\n"
            "    T_H = (1 - r_-/r_+)^gamma / (4 pi r_+)\n\n"
            "where \u03b3 = (1 \u2212 a\u00b2)/(1 + a\u00b2) = 1/2 for a = 1/\u221a3.\n\n"
            "**Key difference from Reissner-Nordstrom:** The extremal GM "
            "black hole (q -> 1) has FINITE temperature, unlike the RN "
            "extremal limit where T -> 0. This means GM black holes "
            "evaporate completely -- no remnant.\n\n"
            f"- r_+ = {temp_data['r_plus']:.6e} m\n"
            f"- r_- = {temp_data['r_minus']:.6e} m\n"
            f"- gamma = {temp_data['gamma']:.4f}"
        )

st.divider()


# ---------------------------------------------------------------------------
# 2. Hawking Spectrum
# ---------------------------------------------------------------------------
st.header("2. Hawking Spectrum")

with st.spinner("Computing Hawking spectrum..."):
    spectrum_data = hawking_spectrum(M_geom, params["q"])

if "error" in spectrum_data:
    st.error(f"Spectrum computation failed: {spectrum_data['error']}")
else:
    fig = hawking_spectrum_chart(spectrum_data)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("About this chart"):
        st.markdown(
            "The solid curve shows the actual Hawking emission rate "
            "dN/(dt domega) summed over angular momentum modes l = 0 to "
            f"{spectrum_data['l_max']}. The dashed curve shows the pure "
            "blackbody (greybody factor = 1) spectrum for comparison.\n\n"
            "The ratio (greybody suppression) is always less than 1, because "
            "the curved spacetime potential barrier partially reflects "
            "outgoing quanta back into the hole. Low-frequency emission is "
            "strongly suppressed; high-frequency quanta can tunnel through."
        )

st.divider()


# ---------------------------------------------------------------------------
# 3. Greybody Factors
# ---------------------------------------------------------------------------
st.header("3. Greybody Factors")

with st.spinner("Scanning greybody factors over charge..."):
    gb_scan_data = greybody_scan(l=2, M=M_geom)

fig_gb = greybody_scan_chart(gb_scan_data)
st.plotly_chart(fig_gb, use_container_width=True)

with st.expander("Greybody physics"):
    st.markdown(
        "The greybody factor Gamma_l(omega) is the transmission probability "
        "through the spacetime potential barrier for angular momentum mode l.\n\n"
        "It is computed via the WKB approximation:\n\n"
        "    Gamma_l(omega) = 1 / (1 + exp(2 pi (V_peak - omega^2) / |V''|^(1/2)))\n\n"
        "where V_peak is the maximum of the effective potential and V'' is "
        "its second derivative at the peak.\n\n"
        "The GM metric with \u03b3 = 1/2 produces modified potential barriers "
        "compared to Schwarzschild, affecting the greybody spectrum for "
        "charged black holes."
    )

st.divider()


# ---------------------------------------------------------------------------
# 4. Total Power
# ---------------------------------------------------------------------------
st.header("4. Total Power")

with st.spinner("Computing total Hawking power..."):
    power_data = total_power(M_geom, params["q"])

if "error" in power_data:
    st.error(f"Power computation failed: {power_data['error']}")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="P_GM (W)",
            value=f"{power_data['P_gm_watts']:.4e}",
        )
    with c2:
        st.metric(
            label="P_Schwarzschild (W)",
            value=f"{power_data['P_schwarz_watts']:.4e}",
        )
    with c3:
        st.metric(
            label="P_GM / P_Schwarz",
            value=f"{power_data['ratio']:.6f}",
        )

    with st.expander("Power and evaporation details"):
        st.markdown(
            f"- P_GM = {power_data['P_gm_watts']:.4e} W\n"
            f"- P_Schwarzschild = {power_data['P_schwarz_watts']:.4e} W\n"
            f"- Evaporation timescale (GM): {power_data['t_evap_gm_seconds']:.4e} s\n"
            f"- Evaporation timescale (Schwarz): "
            f"{power_data['t_evap_schwarz_seconds']:.4e} s\n"
            f"- Exact Schwarzschild t_evap: "
            f"{power_data['t_evap_schwarz_exact_seconds']:.4e} s\n\n"
            "For stellar-mass and supermassive black holes, the Hawking "
            "temperature is far below the CMB temperature, so net evaporation "
            "does not occur. Hawking radiation is significant only for "
            "primordial or micro black holes."
        )

st.divider()


# ---------------------------------------------------------------------------
# 5. Dilaton Channel
# ---------------------------------------------------------------------------
st.header("5. Dilaton Emission Channel")

with st.spinner("Computing dilaton channel..."):
    dilaton_data = dilaton_emission_channel(M_geom, params["q"])

if "error" in dilaton_data:
    st.error(f"Dilaton channel computation failed: {dilaton_data['error']}")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="Fraction of total (massless)",
            value=f"{dilaton_data['fraction_of_total']*100:.1f}%",
        )
    with c2:
        st.metric(
            label="m_phi / T_H",
            value=(
                f"{dilaton_data['m_phi_over_T_H']:.2e}"
                if dilaton_data['m_phi_over_T_H'] < 1e50
                else ">> 1"
            ),
        )
    with c3:
        st.metric(
            label="Suppression (massive)",
            value=(
                f"{dilaton_data['suppression_factor']:.2e}"
                if dilaton_data['suppression_factor'] > 0
                else "0 (complete)"
            ),
        )

    st.markdown(
        '<div class="bh-null">'
        "<p><strong>Dilaton channel is kinematically blocked.</strong></p>"
        "<p>The Alpha Ladder flux stabilization gives the dilaton a "
        "Planck-scale mass m_phi ~ 6.3 \u00d7 10\u00b2\u2079 eV. For any astrophysical "
        f"black hole (T_H ~ {dilaton_data['T_H_eV']:.2e} eV), the ratio "
        f"m_phi / T_H ~ {dilaton_data['m_phi_over_T_H']:.2e}, yielding a "
        "Boltzmann suppression factor of exactly zero to any practical "
        "precision. The extra scalar emission channel is completely shut "
        "off, and the Hawking spectrum reverts to the GR prediction.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("If the dilaton were massless..."):
        st.markdown(
            "A massless dilaton would provide an additional l = 0 scalar "
            "emission channel, contributing "
            f"{dilaton_data['fraction_of_total']*100:.1f}% of the total "
            "scalar Hawking power.\n\n"
            "However, a massless dilaton is independently ruled out by the "
            "Cassini PPN bound (gamma_PPN deviation of ~0.2 vs the measured "
            "precision of 2.3 \u00d7 10\u207b\u2075). The dilaton must be massive, and "
            "the Alpha Ladder framework itself predicts this via flux "
            "stabilization."
        )
