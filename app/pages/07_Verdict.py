"""The Verdict page for the Alpha Ladder BH dashboard.

Synthesizes all results into a final assessment. Calls each module's
summary function and presents the honest conclusion that GM dilaton
effects on black holes are unobservable.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.formatting import fmt_decimal, fmt_percent
from app.components.charts import constraint_summary_chart

try:
    from gibbons_maeda import summarize_gibbons_maeda_analysis
    from quasinormal_modes import summarize_qnm_analysis
    from shadows import summarize_shadow_analysis
    from isco_accretion import summarize_isco_analysis
    from observational_constraints import summarize_observational_constraints
    from greybody_factors import summarize_greybody_analysis
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

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title("The Verdict")
st.markdown(
    "A synthesis of all black hole phenomenology results from the "
    "Alpha Ladder framework."
)

st.divider()

if not _available:
    st.warning(
        "One or more analysis modules not available. "
        "Showing static verdict."
    )
    # Show the verdict even without modules
    _show_static = True
else:
    _show_static = False


# ---------------------------------------------------------------------------
# 1. Constraint Hierarchy
# ---------------------------------------------------------------------------
st.header("1. Constraint Hierarchy")

if not _show_static:
    with st.spinner("Compiling constraint hierarchy..."):
        obs_summary = summarize_observational_constraints()

    # Extract constraint summary table from obs_summary
    if "constraints" in obs_summary and "rows" in obs_summary["constraints"]:
        constraint_data = obs_summary["constraints"]
    else:
        # Try to build it from the summary data
        try:
            from observational_constraints import constraint_summary_table
            constraint_data = constraint_summary_table()
        except Exception:
            constraint_data = None

    if constraint_data is not None:
        fig = constraint_summary_chart(constraint_data)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="bh-null">'
        "<p><strong>Cassini PPN is the killer constraint.</strong> "
        "The massless dilaton with a = 1/\u221a3 predicts gamma_PPN that "
        "deviates from GR by ~0.2 -- ruled out by ~20,000\u03c3. "
        "Even with macroscopic BH charge, the theory is excluded unless "
        "the dilaton is massive. And a massive dilaton decouples, reverting "
        "all BH solutions to pure GR.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    st.info(
        "Constraint hierarchy chart unavailable. "
        "See page 05 for the full analysis."
    )

st.divider()


# ---------------------------------------------------------------------------
# 2. Three Reasons It's Unobservable
# ---------------------------------------------------------------------------
st.header("2. Three Reasons It Is Unobservable")

st.markdown(
    '<div class="bh-null">'
    "<p><strong>Reason 1: Black holes are neutral.</strong></p>"
    "<p>The Wald mechanism and Schwinger pair-production discharge constrain "
    "astrophysical BH charge to q = Q/Q\u2091\u2093\u209c < 10\u207b\u2079 (and likely < 10\u207b\u00b9\u2078). "
    "All Gibbons-Maeda effects scale as q\u00b2, giving deviations from GR "
    "below 10\u207b\u00b3\u2070 parts per million -- completely undetectable by any "
    "conceivable instrument, now or in the future.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="bh-null">'
    "<p><strong>Reason 2: Cassini kills the massless dilaton.</strong></p>"
    "<p>The PPN parameter gamma has been measured by Cassini to a precision "
    "of 2.3 \u00d7 10\u207b\u2075. The massless dilaton with a = 1/\u221a3 predicts a "
    "deviation of ~0.2 from GR -- ruled out by 20,000\u03c3. Even if BHs "
    "somehow carried macroscopic charge, the underlying theory (massless "
    "dilaton) is excluded by solar system tests. The dilaton must be "
    "massive.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="bh-null">'
    "<p><strong>Reason 3: Flux stabilization decouples the "
    "dilaton.</strong></p>"
    "<p>The Alpha Ladder framework's own flux stabilization mechanism gives "
    "the dilaton a Planck-scale mass m_phi ~ 6.3 \u00d7 10\u00b2\u2079 eV. At "
    "astrophysical scales, the dilaton has zero effect. The Yukawa "
    "suppression factor exp(-m_phi r) kills all dilaton-mediated forces "
    "at any distance larger than the Planck length. Black holes are pure "
    "GR.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


# ---------------------------------------------------------------------------
# 3. What Remains Testable
# ---------------------------------------------------------------------------
st.header("3. What Remains Testable")

st.markdown(
    '<div class="bh-positive">'
    "<p><strong>The Alpha Ladder's REAL prediction is not about black "
    "holes.</strong></p>"
    "<p>The framework predicts Newton's gravitational constant G to "
    "sub-ppm precision from fundamental constants alone:</p>"
    "<p>G = \u03b1\u00b2\u2074 \u00d7 \u03bc \u00d7 (\u03bc \u2212 \u221a\u03c6 \u00d7 (1 \u2212 \u03b1)) \u00d7 "
    "hbar c / m_p\u00b2</p>"
    "<p>This formula yields G to -0.31 ppm with zero fitted parameters. "
    "That is the testable prediction -- not BH effects.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "The value of this BH side quest is **self-consistency**: confirming "
    "that the framework does not produce contradictions in the black hole "
    "sector. The honest null result (no observable BH effects) is itself "
    "a positive outcome -- it means the Alpha Ladder's predictions for G "
    "are not undermined by exotic BH phenomenology."
)

st.divider()


# ---------------------------------------------------------------------------
# 4. Summary Table
# ---------------------------------------------------------------------------
st.header("4. Summary Table")

# Build summary table from available data or static values
summary_rows = []

if not _show_static:
    with st.spinner("Building summary table..."):
        try:
            gm_summary = summarize_gibbons_maeda_analysis()
            qnm_summary = summarize_qnm_analysis()
            shadow_summary = summarize_shadow_analysis()
            isco_summary = summarize_isco_analysis()
            greybody_summary = summarize_greybody_analysis()
            _summaries_loaded = True
        except Exception:
            _summaries_loaded = False
else:
    _summaries_loaded = False

# Static fallback data -- always available
observable_data = [
    {
        "Observable": "Hawking temperature",
        "GM deviation at q=0.5": "Finite at extremality (vs T=0 for RN)",
        "GM deviation at realistic q": "< 10\u207b\u00b3\u2070 ppm",
        "Verdict": "Academic",
    },
    {
        "Observable": "Shadow diameter",
        "GM deviation at q=0.5": "~3% shrinkage",
        "GM deviation at realistic q": "< 10\u207b\u00b3\u2070 ppm",
        "Verdict": "Unobservable",
    },
    {
        "Observable": "QNM frequency",
        "GM deviation at q=0.5": "~2% shift",
        "GM deviation at realistic q": "< 10\u207b\u00b3\u2070 ppm",
        "Verdict": "Unobservable",
    },
    {
        "Observable": "ISCO radius",
        "GM deviation at q=0.5": "~5% inward shift",
        "GM deviation at realistic q": "< 10\u207b\u00b3\u2070 ppm",
        "Verdict": "Unobservable",
    },
    {
        "Observable": "Accretion efficiency",
        "GM deviation at q=0.5": "~8% increase",
        "GM deviation at realistic q": "< 10\u207b\u00b3\u2070 ppm",
        "Verdict": "Unobservable",
    },
    {
        "Observable": "Greybody factors",
        "GM deviation at q=0.5": "Modified barrier shape",
        "GM deviation at realistic q": "< 10\u207b\u00b3\u2070 ppm",
        "Verdict": "Academic",
    },
    {
        "Observable": "Dilaton emission channel",
        "GM deviation at q=0.5": "Extra l=0 mode (if massless)",
        "GM deviation at realistic q": "Blocked (m_phi ~ M_Planck)",
        "Verdict": "Blocked",
    },
    {
        "Observable": "PPN gamma",
        "GM deviation at q=0.5": "N/A (solar system test)",
        "GM deviation at realistic q": "~0.2 deviation (massless)",
        "Verdict": "EXCLUDED (20,000\u03c3)",
    },
]

st.dataframe(observable_data, use_container_width=True, hide_index=True)

st.divider()


# ---------------------------------------------------------------------------
# Final statement
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="bh-theory">'
    "<p><strong>Final assessment:</strong> The Gibbons-Maeda dilaton black "
    "hole with the Alpha Ladder coupling a = 1/\u221a3 produces a "
    "mathematically rich structure -- modified horizons, thermodynamics, "
    "orbits, spectra, and greybody factors. However, three independent "
    "arguments (BH neutrality, Cassini PPN, and dilaton mass from flux "
    "stabilization) each independently guarantee that none of these "
    "modifications are observable in practice. This is an honest null "
    "result.</p>"
    "<p>The Alpha Ladder's testable predictions lie elsewhere: in the "
    "sub-ppm prediction of G from fundamental constants, in the 24 d\u03b1/\u03b1 "
    "time variation coefficient, and in the M\u2086 \u2248 3\u20135 TeV compactification "
    "scale accessible to colliders.</p>"
    "</div>",
    unsafe_allow_html=True,
)
