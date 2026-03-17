"""Gibbons-Maeda Foundation page.

Displays the exact charged dilaton black hole solution from the Alpha Ladder
S\u00b2 Kaluza-Klein reduction (\u03c9 = 0, a = 1/\u221a3), including horizon
structure, Hawking temperature, entropy, dilaton hair, and astrophysical
black hole properties.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.formatting import fmt_decimal, fmt_percent
from app.components.charts import temperature_profile_chart, entropy_profile_chart

try:
    from gibbons_maeda import (
        compute_dilaton_coupling,
        compute_hawking_temperature,
        compute_temperature_profile,
        compute_entropy_correction,
        compute_entropy_profile,
        compute_no_hair_violation,
        compute_alpha_ladder_black_holes,
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
st.title("Gibbons-Maeda Foundation")
st.markdown(
    "The exact charged dilaton black hole from the Alpha Ladder "
    "\u03c9 = 0 Kaluza-Klein reduction on S\u00b2. The dilaton coupling "
    "a = 1/\u221a3 is fixed by the reduction, not a free parameter."
)
st.divider()

if not _available:
    st.warning("Module `gibbons_maeda` not available. Install it and restart.")
    st.stop()


# ===================================================================
# 1. Dilaton Coupling
# ===================================================================
st.header("1. Dilaton Coupling")

@st.cache_data
def _cached_coupling():
    return compute_dilaton_coupling()

coupling = _cached_coupling()
a_val = coupling["a_from_BD"]
a_sq = coupling["a_squared_from_BD"]
omega = coupling["omega"]
gamma = (1.0 - a_sq) / (1.0 + a_sq)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("a (from BD)", fmt_decimal(a_val, 6))
with c2:
    st.metric("a\u00b2", fmt_decimal(a_sq, 6))
with c3:
    st.metric("\u03c9 (BD)", str(omega))
with c4:
    st.metric("\u03b3", fmt_decimal(gamma, 6))

with st.expander("Derivation details"):
    st.markdown(
        f"""
**Brans-Dicke route:** a\u00b2 = 1 / (2\u03c9 + 3) = 1/3 for \u03c9 = 0.

**KK route:** a\u00b2 = n / (2(n+2)) = 1/4 for n = 2.

The discrepancy is a known normalization subtlety. We adopt the BD value
a = 1/\u221a3 since \u03c9 = 0 is the defining feature of the Alpha Ladder
reduction.

**Comparison of theories:**
"""
    )
    for name, val in coupling["comparison"].items():
        st.markdown(f"- **{name}**: a = {val:.4f}")

st.divider()


# ===================================================================
# 2. Horizon Structure
# ===================================================================
st.header("2. Horizon Structure")

temp_data = compute_hawking_temperature(a, M_geom, params["q"])

if "error" in temp_data:
    st.error(f"Horizon computation failed: {temp_data['error']}")
else:
    r_plus = temp_data["r_plus"]
    r_minus = temp_data["r_minus"]
    r_ratio = temp_data["r_ratio"]
    gam = temp_data["gamma"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("r+ (m)", fmt_decimal(r_plus, 4))
    with c2:
        st.metric("r- (m)", fmt_decimal(r_minus, 4))
    with c3:
        st.metric("r\u208b / r\u208a", fmt_decimal(r_ratio, 6))
    with c4:
        st.metric("\u03b3", fmt_decimal(gam, 6))

    st.markdown(
        '<div class="bh-theory">'
        "<p><strong>GM horizons vs RN:</strong> "
        "In Reissner-Nordstrom (a = 0), extremal black holes have r\u208a = r\u208b "
        "(degenerate horizon). In the Gibbons-Maeda solution with a = 1/\u221a3, "
        "the inner horizon is r\u208b = a\u00b2 q\u00b2 M\u00b2 / ((1 + a\u00b2) r\u208a), which is always "
        "less than r\u208a even at extremality. This non-degeneracy is the root cause "
        "of the finite extremal temperature.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()


# ===================================================================
# 3. Temperature Profile
# ===================================================================
st.header("3. Temperature Profile")

@st.cache_data
def _cached_temp_profile(_a, _M):
    return compute_temperature_profile(_a, _M)

profile_data = _cached_temp_profile(a, M_geom)

# Filter out error entries for charting
valid_profile = [d for d in profile_data if "error" not in d]
if valid_profile:
    fig = temperature_profile_chart(valid_profile)
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    '<div class="bh-result">'
    "<p><strong>Key result:</strong> Extremal Gibbons-Maeda black holes have "
    "FINITE nonzero Hawking temperature, unlike Reissner-Nordstrom where "
    "T_ext = 0. For a = 1/\u221a3, the extremal temperature ratio is "
    "T_ext / T_Schwarz = (3/4)^(1/2) * 2 ~ 1.732. "
    "This means GM black holes evaporate completely rather than approaching "
    "a zero-temperature remnant.</p>"
    "</div>",
    unsafe_allow_html=True,
)

with st.expander("Temperature data table"):
    table_rows = []
    for entry in valid_profile:
        q = entry.get("qm_ratio", 0.0)
        t_ratio = entry.get("T_ratio", 0.0)
        r_rat = entry.get("r_ratio", 0.0)
        table_rows.append({
            "q = Q/Q_ext": f"{q:.4f}",
            "T_H / T_Schwarz": f"{t_ratio:.6f}" if t_ratio else "N/A",
            "r\u208b / r\u208a": f"{r_rat:.6f}",
        })
    if table_rows:
        st.table(table_rows)

st.divider()


# ===================================================================
# 4. Entropy
# ===================================================================
st.header("4. Entropy")

@st.cache_data
def _cached_entropy_profile(_a, _M):
    return compute_entropy_profile(_a, _M)

entropy_data = _cached_entropy_profile(a, M_geom)
valid_entropy = [d for d in entropy_data if "error" not in d]

if valid_entropy:
    fig_ent = entropy_profile_chart(valid_entropy)
    st.plotly_chart(fig_ent, use_container_width=True)

st.markdown(
    '<div class="bh-theory">'
    "<p><strong>Entropy formula:</strong> "
    "S = \u03c0 r\u208a\u00b2 \u221a(1 \u2212 r\u208b/r\u208a) / G</p>"
    "<p>For a = 1/\u221a3, the entropy exponent is 2a\u00b2/(1+a\u00b2) = 1/2. "
    "The effective horizon area is A_eff = 4\u03c0 r\u208a\u00b2 (1 \u2212 r\u208b/r\u208a)^(1/2), "
    "which is smaller than the Schwarzschild area 4\u03c0 r\u208a\u00b2. Charged "
    "dilaton black holes carry LESS entropy than uncharged black holes "
    "of the same outer horizon radius.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


# ===================================================================
# 5. Dilaton Hair
# ===================================================================
st.header("5. Dilaton Hair")

hair = compute_no_hair_violation(a, M_geom, params["q"])

if "error" in hair:
    st.error(f"Dilaton hair computation failed: {hair['error']}")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Dilaton charge D (m)", fmt_decimal(hair["dilaton_charge_D"], 4))
    with c2:
        st.metric("e^{-2a phi} at horizon", fmt_decimal(hair["eff_gauge_at_horizon"], 6))

    st.markdown("**Dilaton profile phi(r) at various radii:**")

    profile_rows = []
    for loc, val in hair["dilaton_profile"].items():
        profile_rows.append({
            "Location": loc,
            "phi(r)": fmt_decimal(val, 6) if val != float("inf") else "inf",
        })
    st.table(profile_rows)

    st.markdown(
        '<div class="bh-theory">'
        f"<p><strong>Note:</strong> {hair['note']}</p>"
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()


# ===================================================================
# 6. Astrophysical BH Table
# ===================================================================
st.header("6. Astrophysical Black Holes")

@st.cache_data
def _cached_alpha_bhs():
    return compute_alpha_ladder_black_holes()

bh_data = _cached_alpha_bhs()

table_rows = []
for label in ["1 solar mass", "10 solar masses", "Sgr A* (4e6 solar)"]:
    data = bh_data.get(label)
    if data is None:
        continue

    # Extract T at q=0 from temperature profile
    t_q0_str = "N/A"
    for entry in data.get("temperature_profile", []):
        if entry.get("qm_ratio") == 0.0 and "T_Kelvin" in entry:
            t_q0_str = f"{entry['T_Kelvin']:.4e} K"
            break

    table_rows.append({
        "Object": label,
        "M (kg)": f"{data['M_kg']:.3e}",
        "r_Schwarz (m)": f"{data['r_schwarzschild_m']:.4e}",
        "T_Hawking (q=0)": t_q0_str,
        "T_extremal (K)": f"{data['T_extremal_K']:.4e}",
    })

if table_rows:
    st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True,
    )

if "key_finding" in bh_data:
    with st.expander("Key finding"):
        st.markdown(bh_data["key_finding"])
