"""Overview page for the Alpha Ladder BH dashboard.

Presents the key results and navigation guide for the black hole
phenomenology side quest.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.formatting import fmt_decimal, fmt_sigma

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
params = render_sidebar()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("Alpha Ladder \u2014 Black Hole Phenomenology")
st.markdown(
    "A side quest from the Alpha Ladder framework -- exploring what the "
    "\u03c9 = 0 Kaluza-Klein reduction predicts for black hole observables."
)

st.divider()

# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        label="Dilaton coupling",
        value="a = 1/\u221a3",
        delta="Fixed by \u03c9 = 0",
    )

with c2:
    st.metric(
        label="Extremal T\u2095",
        value="Finite",
        delta="Unlike RN (T=0)",
    )

with c3:
    st.metric(
        label="Cassini PPN",
        value="20,000\u03c3",
        delta="Massless dilaton excluded",
        delta_color="inverse",
    )

with c4:
    st.metric(
        label="Observable?",
        value="No",
        delta="q < 10\u207b\u2079 for real BHs",
        delta_color="off",
    )

st.divider()

# ---------------------------------------------------------------------------
# Key result card
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="bh-null">'
    "<p><strong>Key result:</strong> "
    "All Gibbons-Maeda dilaton effects on black hole observables scale as "
    "q<sup>2</sup> where q = Q / Q<sub>ext</sub>. "
    "Astrophysical black holes have q &lt; 10<sup>-9</sup> (Wald mechanism + "
    "Schwinger discharge), making all deviations from GR undetectable. "
    "Additionally, if the dilaton acquires Planck-scale mass from flux "
    "stabilization, it decouples entirely and BH solutions revert to "
    "standard GR.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Navigation guide
# ---------------------------------------------------------------------------
st.header("Page Guide")

nav_items = [
    ("01", "Gibbons-Maeda Solution", "Exact charged dilaton BH metric, thermodynamics, horizons"),
    ("02", "Quasinormal Modes", "QNM frequencies via WKB, LIGO/ET/LISA detectability"),
    ("03", "Shadows & EHT", "Shadow radius, EHT angular size, Sgr A* and M87* bounds"),
    ("04", "ISCO & Accretion", "Innermost stable orbit, radiative efficiency, luminosity"),
    ("05", "Observational Constraints", "PPN \u03b3, Cassini bound, charge limits, effects scaling"),
    ("06", "Greybody & Hawking", "Greybody factors, Hawking spectrum, dilaton emission channel"),
    ("07", "The Verdict", "Synthesis of all results, three null-result arguments, what IS testable"),
]

# 2-column grid, 4 rows
for i in range(0, len(nav_items), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        idx = i + j
        if idx < len(nav_items):
            num, title, desc = nav_items[idx]
            col.markdown(
                f'<div class="bh-nav">'
                f'<strong>{num} \u2014 {title}</strong><br>'
                f'{desc}'
                f'</div>',
                unsafe_allow_html=True,
            )
