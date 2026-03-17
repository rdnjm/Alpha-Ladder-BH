"""Shared sidebar component for the Alpha Ladder BH dashboard.

Provides the mass selector, charge ratio slider, dilaton coupling info,
and injects global CSS for consistent dark-theme styling.
"""

from __future__ import annotations

import math

import streamlit as st

# ---------------------------------------------------------------------------
# Mass presets: label -> solar masses
# ---------------------------------------------------------------------------
_MASS_PRESETS: dict[str, float] = {
    "1 M_sun": 1.0,
    "10 M_sun": 10.0,
    "30 M_sun": 30.0,
    "Sgr A* (4e6 M_sun)": 4.0e6,
    "M87* (6.5e9 M_sun)": 6.5e9,
}

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Mono:wght@400;500&display=swap');

/* ---- Base font family everywhere ---- */
html, body, [class*="css"] {
    font-family: 'Fira Mono', monospace;
}

/* ---- Main content area ---- */
.main .block-container {
    font-size: 1.1rem !important;
    line-height: 1.5;
    background: #0e1117;
}

/* ---- Headings via testid ---- */
div[data-testid="stHeading"] h1 {
    font-size: 2.0rem !important;
}
div[data-testid="stHeading"] h2 {
    font-size: 1.6rem !important;
}
div[data-testid="stHeading"] h3 {
    font-size: 1.3rem !important;
}

/* ---- Markdown containers via testid ---- */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span {
    font-family: 'Fira Mono', monospace !important;
    font-size: 1.1rem !important;
    line-height: 1.55;
}

/* ---- Sidebar via testid ---- */
section[data-testid="stSidebar"] {
    font-size: 1.05rem !important;
    line-height: 1.6;
}
section[data-testid="stSidebar"] h2 { font-size: 1.3rem !important; }
section[data-testid="stSidebar"] h3 { font-size: 1.1rem !important; }

/* ---- Metrics via testid ---- */
div[data-testid="stMetric"] {
    background: #1a1d23;
    border: 1px solid #2e3440;
    border-radius: 8px;
    padding: 1rem;
}
div[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 1.0rem !important;
}

/* ---- Code / pre ---- */
code, pre {
    font-size: 1.05rem !important;
}

/* ---- Tab styling ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Fira Mono', monospace;
    font-size: 0.95rem !important;
}

/* ---- Semantic left-border accent cards ---- */
.bh-theory {
    background: #1a1d23;
    border-left: 4px solid #60a5fa;
    padding: 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}
.bh-result {
    background: #1a1d23;
    border-left: 4px solid #a78bfa;
    padding: 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}
.bh-null {
    background: #1a1d23;
    border-left: 4px solid #f87171;
    padding: 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}
.bh-positive {
    background: #1a1d23;
    border-left: 4px solid #34d399;
    padding: 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}
.bh-caveat {
    background: #1a1d23;
    border-left: 4px solid #f59e0b;
    padding: 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}
.bh-nav {
    background: #1a1d23;
    border: 1px solid #3b4252;
    padding: 1.2rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}
</style>
"""


def render_sidebar() -> dict:
    """Render the shared sidebar and return user-selected parameters.

    Returns
    -------
    dict
        ``{"M_solar": float, "q": float}`` where *M_solar* is the black
        hole mass in solar masses and *q* is the charge-to-extremal ratio.
    """
    # Inject global CSS once per page render
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## Alpha Ladder BH")
        st.caption("A side quest from the Alpha Ladder framework")

        st.divider()

        # -- Mass selector --
        st.markdown("### Black Hole Mass")
        mass_label = st.selectbox(
            "Mass preset",
            options=list(_MASS_PRESETS.keys()),
            index=0,
            label_visibility="collapsed",
        )
        m_solar = _MASS_PRESETS[mass_label]

        st.divider()

        # -- Charge ratio slider --
        st.markdown("### Charge Ratio  q = Q/Q\u2091\u2093\u209c")
        q = st.slider(
            "Charge ratio",
            min_value=0.0,
            max_value=0.99,
            value=0.5,
            step=0.01,
            key="q_slider",
            label_visibility="collapsed",
        )

        st.divider()

        # -- Dilaton coupling (read-only) --
        st.markdown("### Dilaton Coupling")
        a_val = 1.0 / math.sqrt(3)
        st.info(
            f"a = 1/\u221a3 = {a_val:.4f}\n\n"
            "Fixed by the \u03c9 = 0 Kaluza-Klein reduction."
        )

    return {"M_solar": m_solar, "q": q}
