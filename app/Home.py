"""Entry point for the Alpha Ladder BH Streamlit dashboard."""

import streamlit as st

st.set_page_config(
    page_title="Alpha Ladder BH",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/00_Overview.py", title="Overview", icon=None),
    st.Page("pages/01_Gibbons_Maeda.py", title="Gibbons-Maeda", icon=None),
    st.Page("pages/02_Quasinormal_Modes.py", title="Quasinormal Modes", icon=None),
    st.Page("pages/03_Shadows.py", title="Shadows & EHT", icon=None),
    st.Page("pages/04_ISCO_Accretion.py", title="ISCO & Accretion", icon=None),
    st.Page("pages/05_Observational_Constraints.py", title="Observational Constraints", icon=None),
    st.Page("pages/06_Greybody_Factors.py", title="Greybody & Hawking", icon=None),
    st.Page("pages/07_Verdict.py", title="The Verdict", icon=None),
]

nav = st.navigation(pages)
nav.run()
