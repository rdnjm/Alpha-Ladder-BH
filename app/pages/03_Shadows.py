"""Shadows & EHT page.

Computes the photon sphere, shadow angular size, and EHT constraints for
Gibbons-Maeda dilaton black holes with a = 1/\u221a3. Compares shadow
shrinkage rate with Reissner-Nordstrom and provides an honest assessment
of observational prospects.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.formatting import fmt_decimal, fmt_percent
from app.components.charts import shadow_scan_chart, eht_constraint_chart

try:
    from shadows import (
        gm_horizons,
        photon_sphere,
        shadow_angular_size,
        shadow_scan,
        eht_constraints,
        compare_rn_vs_gm,
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
st.title("Shadows & EHT")
st.markdown(
    "Shadow radius predictions for the Gibbons-Maeda dilaton black hole "
    "(a = 1/\u221a3), comparison with Reissner-Nordstrom (a = 0), and "
    "constraints from Event Horizon Telescope observations of Sgr A* and M87*."
)
st.divider()

if not _available:
    st.warning("Module `shadows` not available. Install it and restart.")
    st.stop()


# ===================================================================
# 1. Photon Sphere
# ===================================================================
st.header("1. Photon Sphere")

ps = photon_sphere(M_geom, params["q"])

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("r_ph / M", fmt_decimal(ps["r_ph_over_M"], 4))
with c2:
    st.metric("b_c / M", fmt_decimal(ps["b_c_over_M"], 6))
with c3:
    st.metric("delta_b vs Schwarz (%)", fmt_decimal(ps["delta_b_percent"], 4))
with c4:
    st.metric("f(r_ph)", fmt_decimal(ps["f_at_rph"], 6))

with st.expander("Schwarzschild reference values"):
    st.markdown(
        f"- r_ph (Schwarz) = 3M = {3.0 * M_geom:.4e} m\n"
        f"- b_c (Schwarz) = 3 sqrt(3) M = {3.0 * math.sqrt(3.0) * M_geom:.4e} m\n"
        f"- r_ph / M (Schwarz) = 3.0\n"
        f"- b_c / M (Schwarz) = {3.0 * math.sqrt(3.0):.6f}"
    )

st.divider()


# ===================================================================
# 2. Shadow Size vs Charge
# ===================================================================
st.header("2. Shadow Size vs Charge")

tab_sgra, tab_m87 = st.tabs(["Sgr A*", "M87*"])

with tab_sgra:
    @st.cache_data
    def _cached_scan_sgra(m_solar):
        return shadow_scan(m_solar, None, D_kpc=8.127)

    scan_sgra = _cached_scan_sgra(4.0e6)

    @st.cache_data
    def _cached_comparison():
        return compare_rn_vs_gm()

    comparison = _cached_comparison()

    fig_scan = shadow_scan_chart(scan_sgra, comparison)
    st.plotly_chart(fig_scan, use_container_width=True, key="shadow_sgra")

    st.markdown(
        '<div class="bh-result">'
        "<p><strong>Key observation:</strong> The Gibbons-Maeda shadow shrinks "
        "approximately 34% faster with charge than the Reissner-Nordstrom shadow. "
        "This is because the dilaton modifies the areal radius R(r)\u00b2 = "
        "r\u00b2 (1 \u2212 r\u208b/r)^(1 \u2212 \u03b3), which affects the critical impact "
        "parameter b_c more strongly than in the a = 0 case.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Comparison data table
    with st.expander("GM vs RN shadow data"):
        comp_table = []
        for row in comparison:
            comp_table.append({
                "q": f"{row['q']:.2f}",
                "delta_GM (%)": f"{row['delta_gm_percent']:.4f}",
                "delta_RN (%)": f"{row['delta_rn_percent']:.4f}",
                "GM / RN": (
                    f"{row['gm_vs_rn_ratio']:.6f}"
                    if row["gm_vs_rn_ratio"] is not None else "N/A"
                ),
            })
        st.dataframe(comp_table, use_container_width=True, hide_index=True)

with tab_m87:
    @st.cache_data
    def _cached_scan_m87(m_solar):
        return shadow_scan(m_solar, 16.8)

    scan_m87 = _cached_scan_m87(6.5e9)

    fig_m87 = shadow_scan_chart(scan_m87, comparison)
    st.plotly_chart(fig_m87, use_container_width=True, key="shadow_m87")

    st.markdown(
        f"**M87* Schwarzschild shadow:** "
        f"{scan_m87.get('theta_schwarz_uas', 0.0):.2f} uas"
    )

st.divider()


# ===================================================================
# 3. EHT Constraints
# ===================================================================
st.header("3. EHT Constraints")

@st.cache_data
def _cached_eht():
    return eht_constraints()

eht_data = _cached_eht()

fig_eht = eht_constraint_chart(eht_data)
st.plotly_chart(fig_eht, use_container_width=True)

# Detailed constraint table
for key, label in [("sgra", "Sgr A*"), ("m87", "M87*")]:
    src = eht_data.get(key, {})
    if not src:
        continue

    with st.expander(f"{label} constraint details"):
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Schwarzschild shadow (uas)",
                f"{src.get('theta_schwarz_uas', 0):.2f}",
            )
            st.metric(
                "Observed shadow (uas)",
                f"{src.get('theta_observed', 0):.1f} +/- {src.get('sigma', 0):.1f}",
            )
        with c2:
            st.metric(
                "q_max (1-sigma)",
                f"{src.get('q_max_1sigma', 0):.3f}",
            )
            st.metric(
                "q_max (2-sigma)",
                f"{src.get('q_max_2sigma', 0):.3f}",
            )

        st.caption(f"Reference: {src.get('ref', '')}")

st.markdown(
    '<div class="bh-caveat">'
    "<p><strong>Note on uncertainties:</strong> Current EHT shadow measurements "
    "have 7-15% uncertainties, far too large to distinguish Gibbons-Maeda "
    "from Reissner-Nordstrom or even to rule out moderate charge ratios. "
    "Next-generation EHT (ngEHT) may reach ~1% precision, which would begin "
    "to probe q ~ 0.3-0.5 if charge were somehow sustained.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


# ===================================================================
# 4. Honest Assessment
# ===================================================================
st.header("4. Honest Assessment")

st.markdown(
    """
**Why shadows cannot distinguish GM from GR in practice:**

- **Astrophysical black holes are neutral.** Selective charge accretion and
  Schwinger pair production discharge any macroscopic electric charge on
  timescales much shorter than the black hole age. For all observed black
  holes, q = Q/Q_ext < 10\u207b\u2079, making dilaton shadow deviations unobservably
  small.

- **The dilaton may be massive.** If flux stabilization gives the dilaton a
  Planck-scale mass (as the Alpha Ladder framework predicts), it decouples
  at astrophysical scales and black hole solutions revert exactly to GR.
  The entire GM shadow analysis then becomes academic.

- **EHT uncertainties are too large.** With 7-15% measurement uncertainties,
  current observations cannot distinguish GM from RN even at moderate charge.
  The mass-charge degeneracy further limits constraining power.

- **The value is theoretical.** This analysis maps out the observable
  consequences of the Alpha Ladder's specific dilaton coupling and provides
  concrete, falsifiable predictions should charged black hole solutions prove
  relevant in other contexts (primordial black holes, dark sector charges,
  or magnetic monopoles).
"""
)

st.markdown(
    '<div class="bh-null">'
    "<p><strong>Bottom line:</strong> The Gibbons-Maeda shadow analysis "
    "produces a null result for current observations. All deviations from GR "
    "scale as q\u00b2, and astrophysical black holes have q \u2248 0. The prediction "
    "is concrete and falsifiable in principle, but observational verification "
    "requires both next-generation instruments AND charged black holes -- "
    "neither of which is available in the near term.</p>"
    "</div>",
    unsafe_allow_html=True,
)
