"""Main entrypoint for Streamlit UI.

The page simply sets the theme and then routes to either the login page or the
dashboard based on session state.
"""
import os
import streamlit as st
from ui_components import login, dashboard, utils

# initialize session variables if missing
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# page configuration and basic styling
st.set_page_config(page_title="Vision Memory Assistant", layout="wide")
THEME_BG = "#0f1117"
THEME_PANEL = "#151a22"
THEME_TEXT = "#e6e6e6"
THEME_ACCENT = "#00ADB5"
st.markdown(f"""
<style>
    .stApp {{ background-color: {THEME_BG}; color: {THEME_TEXT}; }}
    .block-container {{ padding-top: 0.5rem; }}
    .panel {{ background-color: {THEME_PANEL}; padding: 1rem; border-radius: 12px; }}
    .title {{ font-size: 26px; font-weight: 700; margin-bottom: 0.25rem; margin-top: 1rem; }}
    .sub {{ opacity: 0.8; margin-bottom: 1rem; }}
    div.stButton > button {{
        background-color: {THEME_ACCENT}; color: white; border-radius: 10px;
        border: 0px; padding: 0.55rem 1rem;
    }}
    input, textarea {{ background-color: #0c0f14 !important; color: {THEME_TEXT} !important; }}
</style>
""", unsafe_allow_html=True)

# routing logic
if st.session_state.user_id is None:
    login.login_page()
else:
    dashboard.dashboard_page()