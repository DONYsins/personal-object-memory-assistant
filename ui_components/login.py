"""
Login / Registration Page

WHY THIS PAGE:
  Users need to create accounts and log in to access their personal object tracking.
  Each user has their own database records, FAISS indices, and event history.

FLOW:
  1. New users: click "Register" tab, create username + password
  2. Existing users: click "Login" tab, enter credentials
  3. On success: user_id stored in session, redirected to dashboard
"""
import streamlit as st
from ui_components import utils
from logger import log_info, log_error


def login_page():
    # Page header
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="title">Login</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Sign in to your profile</div>', unsafe_allow_html=True)

    # Two tabs: existing users login, new users register
    tabs = st.tabs(["Login", "Register"])
    
    # ============================================================
    # TAB 1: LOGIN
    # ============================================================
    with tabs[0]:
        st.markdown("**Enter your credentials to log in:**")
        username = st.text_input("Username", key="login_u", help="Your username")
        password = st.text_input("Password", type="password", key="login_p", help="Your password")
        
        if st.button("Login", use_container_width=False):
            # WHY: Validate username + password against database
            # Backend checks bcrypt hash for security
            try:
                res = utils.api_post("/login", {"username": username, "password": password})
            except Exception as e:
                st.error(f"❌ Login request failed: {e}")
                log_error(f"[UI] Login request failed: {e}")
                return

            if res.get("status") == "success":
                # Success: store user_id in session and redirect
                st.session_state.user_id = res["user_id"]
                st.success("✓ Login successful!")
                log_info(f"[UI] Login success user_id={st.session_state.user_id}")
                st.rerun()
            else:
                # Failure: show error message
                msg = res.get("message", "Login failed")
                st.error(f"❌ {msg}")
                log_info(f"[UI] Login failed for '{username}': {msg}")

    # ============================================================
    # TAB 2: REGISTER (New Users)
    # ============================================================
    with tabs[1]:
        st.markdown("**Create a new account:**")
        st.info("💡 Choose a username and password you'll remember")
        
        new_username = st.text_input("New Username", key="reg_u", help="Pick a unique username")
        new_password = st.text_input("New Password", type="password", key="reg_p", help="Minimum recommended: 8 characters")
        
        if st.button("Register", use_container_width=False):
            # Backend stores bcrypt hash (not plaintext) for security
            if not new_username or not new_password:
                st.error("❌ Please enter both username and password")
                return
            
            if len(new_password) < 4:
                st.error("❌ Password too short (minimum 4 characters)")
                return
            
            try:
                res = utils.api_post("/register_user", {"username": new_username, "password": new_password})
            except Exception as e:
                st.error(f"❌ Register request failed: {e}")
                log_error(f"[UI] Register request failed: {e}")
                return

            if res.get("status") == "success":
                # Success: user can now log in
                st.success("✓ Registration successful!")
                st.info("👉 Now switch to the 'Login' tab and sign in with your new credentials")
                log_info(f"[UI] Registered successfully: {new_username}")
            else:
                # Failure: show error (e.g., username already exists)
                msg = res.get("message", "Registration failed")
                st.error(f"❌ {msg}")
                log_info(f"[UI] Register failed for '{new_username}': {msg}")