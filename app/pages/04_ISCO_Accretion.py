"""ISCO & Accretion page for the Alpha Ladder BH dashboard.

Presents innermost stable circular orbit analysis, accretion efficiency,
and luminosity comparisons for Gibbons-Maeda dilaton black holes with
the Alpha Ladder coupling a = 1/\u221a3.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.formatting import fmt_decimal, fmt_percent
from app.components.charts import isco_scan_chart

try:
    from isco_accretion import (
        find_isco,
        accretion_efficiency,
        isco_scan,
        luminosity_comparison,
        compare_gm_rn_kerr,
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
st.title("ISCO & Accretion")
st.markdown(
    "Innermost stable circular orbit and radiative efficiency for the "
    "Gibbons-Maeda dilaton black hole with a = 1/\u221a3."
)

st.divider()

if not _available:
    st.warning("Module `isco_accretion` not available. Cannot compute results.")
    st.stop()


# ---------------------------------------------------------------------------
# 1. ISCO Properties
# ---------------------------------------------------------------------------
st.header("1. ISCO Properties")

isco_data = find_isco(M_geom, params["q"])

if "error" in isco_data:
    st.error(f"ISCO computation failed: {isco_data['error']}")
else:
    # Schwarzschild reference
    eta_schwarz = 1.0 - math.sqrt(8.0 / 9.0)
    delta_r = isco_data["delta_r_vs_schwarz_percent"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="r_ISCO / M",
            value=f"{isco_data['r_isco_over_M']:.4f}",
            delta=f"Schwarz: 6.0000",
            delta_color="off",
        )
    with c2:
        st.metric(
            label="E_ISCO",
            value=f"{isco_data['E_isco']:.6f}",
        )
    with c3:
        st.metric(
            label="eta (%)",
            value=f"{isco_data['eta_percent']:.2f}%",
            delta=f"Schwarz: {eta_schwarz * 100:.2f}%",
            delta_color="off",
        )
    with c4:
        st.metric(
            label="delta vs Schwarzschild",
            value=f"{delta_r:+.2f}%",
            delta="in r_ISCO",
            delta_color="off",
        )

    with st.expander("Schwarzschild reference values"):
        st.markdown(
            f"- **r_ISCO** = 6 M\n"
            f"- **E_ISCO** = \u221a(8/9) = {math.sqrt(8.0/9.0):.6f}\n"
            f"- **eta** = 1 \u2212 \u221a(8/9) = {eta_schwarz*100:.4f}%\n"
            f"- At q = 0 the GM solution reduces identically to Schwarzschild."
        )

st.divider()


# ---------------------------------------------------------------------------
# 2. Efficiency Scan
# ---------------------------------------------------------------------------
st.header("2. Efficiency Scan")

with st.spinner("Computing ISCO scan over charge ratios..."):
    scan_data = isco_scan()
    comparison_data = compare_gm_rn_kerr()

fig = isco_scan_chart(scan_data, comparison_data)
st.plotly_chart(fig, use_container_width=True)

with st.expander("About this chart"):
    st.markdown(
        "**Left panel:** ISCO radius in units of M as a function of the "
        "charge ratio q = Q/Q_ext, comparing Gibbons-Maeda (a = 1/\u221a3) "
        "with Reissner-Nordstrom (a = 0).\n\n"
        "**Right panel:** Radiative accretion efficiency eta = 1 - E_ISCO "
        "as a function of charge ratio. Higher efficiency means more rest-mass "
        "energy is radiated as matter spirals to the ISCO.\n\n"
        "Both solutions converge to the Schwarzschild values (r_ISCO = 6M, "
        "eta = 5.72%) at q = 0."
    )

st.divider()


# ---------------------------------------------------------------------------
# 3. GM vs RN vs Schwarzschild
# ---------------------------------------------------------------------------
st.header("3. GM vs RN vs Schwarzschild")

# Filter out error entries and build table
table_rows = []
for row in comparison_data:
    if "error" in row:
        continue
    table_rows.append({
        "q": f"{row['q']:.2f}",
        "r_ISCO/M (GM)": f"{row['r_isco_gm_over_M']:.4f}",
        "r_ISCO/M (RN)": (
            f"{row['r_isco_rn_over_M']:.4f}"
            if row.get("r_isco_rn_over_M") is not None else "N/A"
        ),
        "eta_GM (%)": f"{row['eta_gm_percent']:.2f}",
        "eta_RN (%)": (
            f"{row['eta_rn_percent']:.2f}"
            if row.get("eta_rn_percent") is not None else "N/A"
        ),
        "r_GM / r_RN": (
            f"{row['r_ratio_gm_rn']:.6f}"
            if row.get("r_ratio_gm_rn") is not None else "1.0000"
        ),
    })

if table_rows:
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

st.markdown(
    '<div class="bh-positive">'
    "<p><strong>Key observation:</strong> At q = 0, all three solutions agree "
    "exactly: r_ISCO = 6M and eta = 5.72%. The dilaton field vanishes for "
    "uncharged black holes, so the Gibbons-Maeda solution reduces identically "
    "to Schwarzschild.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


# ---------------------------------------------------------------------------
# 4. Luminosity
# ---------------------------------------------------------------------------
st.header("4. Luminosity Comparison")

with st.spinner("Computing luminosity..."):
    lum = luminosity_comparison(params["M_solar"], 1.0e-8, params["q"])

if "error" in lum:
    st.error(f"Luminosity computation failed: {lum['error']}")
else:
    st.markdown(
        f"**Accretion rate:** 10^-8 M_sun/yr (typical X-ray binary)"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="L_GM (L_sun)",
            value=fmt_decimal(lum["L_gm_Lsun"]),
        )
    with c2:
        st.metric(
            label="L_Schwarzschild (L_sun)",
            value=fmt_decimal(lum["L_schwarz_Lsun"]),
        )
    with c3:
        st.metric(
            label="L_Eddington (L_sun)",
            value=fmt_decimal(lum["L_eddington_Lsun"]),
        )
    with c4:
        st.metric(
            label="delta_L",
            value=f"{lum['delta_L_percent']:+.4f}%",
            delta="GM vs Schwarzschild",
            delta_color="off",
        )

    with st.expander("Luminosity details"):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**GM dilaton:**")
            st.markdown(f"- eta = {lum['eta_gm']*100:.4f}%")
            st.markdown(f"- L = {lum['L_gm_W']:.4e} W")
        with cols[1]:
            st.markdown("**Schwarzschild (GR):**")
            st.markdown(f"- eta = {lum['eta_schwarz']*100:.4f}%")
            st.markdown(f"- L = {lum['L_schwarz_W']:.4e} W")

        if lum.get("eta_rn") is not None:
            st.markdown(f"**Reissner-Nordstrom:** eta = {lum['eta_rn']*100:.4f}%")

st.divider()

# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------
st.header("Honest Caveats")
st.markdown(
    "- Astrophysical black holes are expected to be electrically neutral to "
    "extremely high precision (q < 10\u207b\u2079). At these charge levels, the ISCO "
    "is indistinguishable from the Schwarzschild value 6M.\n"
    "- If the dilaton acquires Planck-scale mass from flux stabilization, it "
    "decouples entirely and BH orbits revert to pure GR.\n"
    "- Real accretion disks are thick, turbulent, and magnetized. The "
    "thin-disk ISCO efficiency is an idealization.\n"
    "- The value of this analysis is theoretical: mapping the observable "
    "consequences of the Alpha Ladder dilaton coupling."
)
