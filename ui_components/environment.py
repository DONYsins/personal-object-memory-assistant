"""Environment management UI (create, delete)"""
import os
import json
import time
import streamlit as st
from logger import log_info, log_error
from ui_components import utils


def environment_controls(uid):
    st.markdown("#### Add Environment")
    env_label = st.text_input("Environment name (e.g., Bedroom)")
    if st.button("Create Environment"):
        log_info(f"[UI] Button: Create Environment - {env_label}")
        res = utils.api_post("/add_environment", {"user_id": uid, "environment_label": env_label})
        if res.get("status") == "success":
            st.success("Created environment.")
            st.rerun()
        else:
            st.error(res.get("message", "Failed to create environment"))

# helper to remove environment entirely from UI

def delete_environment(uid, env_id):
    utils.api_post("/delete_environment", {"user_id": uid, "environment_id": env_id})
