"""Dashboard page assembling all components."""
import streamlit as st
from ui_components import tracking, query, environment, utils
from logger import log_info

from constants import PERSONAL_CLASSES, BACKEND_FPS, LANDMARK_CLASSES, CLEANUP_OPTIONS_HOURS


def dashboard_page():
    uid = st.session_state.user_id
    
    # Get username from cache or fetch it
    if "username" not in st.session_state:
        user_data = utils.api_get("/get_user_info", {"user_id": uid})
        st.session_state.username = user_data.get("username", f"User {uid}")
    
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub">Current user: {st.session_state.username}</div>', unsafe_allow_html=True)

    # Use fixed FPS from constants
    fps = BACKEND_FPS

    # Top bar controls
    colA, colB = st.columns([3, 1])
    with colA:
        ip_cam_url = st.text_input("IP Camera URL", value=utils.IP_CAM_URL_DEFAULT)
    with colB:
        if st.button("Logout"):
            tracking.stop_ingest_if_running()
            st.session_state.user_id = None
            st.session_state.session_id = None
            st.session_state.mode = None
            log_info("Logout")
            st.rerun()

    # fetch list of objects/environments
    data = utils.api_get("/dashboard", {"user_id": uid})
    objects = data.get("objects", [])
    envs = data.get("environments", [])

    # SINGLE COLUMN LAYOUT
    
    # 1. TRACKING
    tracking.tracking_controls(uid, ip_cam_url, fps)
    st.divider()

    # 2. QUERY
    query.query_controls(uid)
    st.divider()

    # 3. PERSONAL OBJECTS
    st.markdown("### Personal Objects")
    if objects:
        for o in objects:
            col1, col2 = st.columns([4, 1])
            with col1:
                # display only type and user label
                st.write(f"• [{o['generic_type']}] **{o['user_label']}**")
            with col2:
                if st.button("Delete", key=f"del_obj_{o['user_object_id']}"):
                    log_info(f"[UI] Button: Delete personal object '{o['user_label']}'")
                    utils.api_post("/delete_personal_object", {"user_id": uid, "user_object_id": o['user_object_id']})
                    st.success("Deleted")
                    st.rerun()
    else:
        st.info("No personal objects yet. Add one below.")

    # Add Personal Object
    st.markdown("#### Add Personal Object")

    # PERSONAL_CLASSES from constants automatically appear
    gen_type = st.selectbox("Object type", PERSONAL_CLASSES, key="add_gen")
    user_label = st.text_input("Your label (e.g., Black Wallet)", key="add_label")
    if st.button("Create Object"):
        log_info(f"[UI] Button: Create Object - type={gen_type}, label={user_label}")
        res = utils.api_post("/add_personal_object", {"user_id": uid, "generic_type": gen_type, "user_label": user_label})
        if res.get("status") == "success":
            st.success(f"Created object id={res['user_object_id']}. Now enroll it below.")
            st.rerun()
        else:
            st.error(res.get("message", "Failed to create object")) 

    # Enroll Object (Live)
    st.markdown("#### Enroll Object (Live)")
    if objects:
        obj_choice = st.selectbox(
            "Choose object to enroll",
            objects,
            format_func=lambda x: f"{x['user_label']} (type={x['generic_type']})",
            key="enroll_obj_choice"
        )
        colE1, colE2 = st.columns(2)
        with colE1:
            if st.button("Start Enroll Object"):
                log_info(f"[UI] Button: Start Enroll Object - {obj_choice['user_label']}")
                res = utils.api_post("/start_session", {
                    "user_id": uid,
                    "mode": "ENROLL_OBJECT",
                    "user_object_id": obj_choice["user_object_id"]
                })
                if res.get("status") == "success":
                    tracking.stop_ingest_if_running()
                    st.session_state.session_id = res["session_id"]
                    st.session_state.mode = "ENROLL_OBJECT"
                    st.session_state.ingest_proc = tracking.launch_ingest(res["session_id"], ip_cam_url, fps=fps)
                    st.success("Enrolling... show the object to camera.")
                else:
                    st.error(res.get("message", "Failed to start enroll"))
        with colE2:
            if st.button("Stop Enroll Object"):
                log_info(f"[UI] Button: Stop Enroll Object")
                sid = st.session_state.session_id
                if sid:
                    utils.api_post("/stop_session", {"session_id": sid})
                tracking.stop_ingest_if_running()
                st.session_state.session_id = None
                st.session_state.mode = None
                st.success("Enroll stopped.")
        
        
        # Check if enroll auto-stopped (e.g., camera disconnected)
        if st.session_state.get("session_id"):
            utils.auto_stop_if_session_ended()
    else:
        st.info("Create an object first, then enroll it.")

    st.divider()

    # 4. ENVIRONMENTS
    st.markdown("### Environments")
    if envs:
        for e in envs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"• **{e['environment_label']}**")
            with col2:
                if st.button("Delete", key=f"del_env_{e['environment_id']}"):
                    log_info(f"[UI] Button: Delete Environment - {e['environment_label']}")
                    utils.api_post("/delete_environment", {"user_id": uid, "environment_id": e['environment_id']})
                    st.success("Deleted")
                    st.rerun()
    else:
        st.info("No environments yet. Add one below.")

    st.divider()

    # Add Environment and Enroll Environment
    environment.environment_controls(uid)

    st.divider()

    # Manage Environment Landmarks
    st.markdown("### Landmarks")
    if envs:
        env_choice = st.selectbox(
            "Select Environment",
            envs,
            format_func=lambda x: x['environment_label'],
            key="left_landmark_env"
        )
        
        # Display existing landmarks
        saved = utils.api_get("/get_environment_landmarks", {"user_id": uid, "environment_id": env_choice["environment_id"]})
        if saved.get("landmarks"):
            st.markdown("**Existing Landmarks:**")
            for lm in saved.get("landmarks", []):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"• {lm['landmark_class']} → **{lm.get('user_label') or '(no label)'}**")
                with col2:
                    # multiple labels for same landmark_class without key collision
                    user_label_safe = (lm.get('user_label') or 'nolabel').replace(" ", "_")
                    if st.button("Delete", key=f"del_lm_{env_choice['environment_id']}_{user_label_safe}"):
                        log_info(f"[UI] Button: Delete Landmark - {lm['landmark_class']}")
                        r2 = utils.api_post("/delete_environment_landmark", {"user_id": uid, "environment_id": env_choice["environment_id"], "user_label": lm.get('user_label', '')})
                        if r2.get("status") == "success":
                            st.success("Deleted")
                            st.rerun()
                        else:
                            st.error(r2.get("message", "Failed"))
        else:
            st.info("No landmarks defined yet.")
        
        # Add new landmark form
        st.markdown("**Add New Landmark:**")
        
        landmark_class = st.selectbox(
            "YOLO Landmark Class",
            LANDMARK_CLASSES,
            key="add_landmark_class"
        )
        user_landmark_label = st.text_input("Your Label (e.g., 'Master Bed')", key="add_landmark_label")
        if st.button("Add Landmark"):
            log_info(f"[UI] Button: Add Landmark - {landmark_class} as {user_landmark_label}")
            if user_landmark_label:
                res = utils.api_post("/add_environment_landmark", {
                    "user_id": uid,
                    "environment_id": env_choice["environment_id"],
                    "landmark_class": landmark_class,
                    "user_label": user_landmark_label
                })
                if res.get("status") == "success":
                    st.success(f"Added '{user_landmark_label}' to {env_choice['environment_label']}")
                    st.rerun()
                else:
                    st.error(res.get("message", "Failed to add landmark"))
            else:
                st.error("Please enter a label")
    else:
        st.info("Create an environment first.")

    st.divider()

    # ENROLL LANDMARK (LIVE) - Similar to ENROLL_OBJECT
    st.markdown("#### Enroll Landmark (Live)")
    if envs:
        env_choice_enroll = st.selectbox(
            "Select Environment for Landmark Enrollment",
            envs,
            format_func=lambda x: x['environment_label'],
            key="enroll_landmark_env"
        )
        
        saved_lm = utils.api_get("/get_environment_landmarks", {"user_id": uid, "environment_id": env_choice_enroll["environment_id"]})
        landmarks_list = saved_lm.get("landmarks", [])
        
        if landmarks_list:
            lm_choice = st.selectbox(
                "Choose landmark to enroll",
                landmarks_list,
                format_func=lambda x: f"{x['user_label']} (type={x['landmark_class']})",
                key="enroll_lm_choice"
            )
            colL1, colL2 = st.columns(2)
            with colL1:
                if st.button("Start Enroll Landmark"):
                    log_info(f"Button: Start Enroll Landmark - {lm_choice['user_label']}")
                    res = utils.api_post("/start_session", {
                        "user_id": uid,
                        "mode": "ENROLL_LANDMARK",
                        "environment_id": env_choice_enroll["environment_id"],
                        "landmark_id": lm_choice["environment_landmark_id"]
                    })
                    if res.get("status") == "success":
                        tracking.stop_ingest_if_running()
                        st.session_state.session_id = res["session_id"]
                        st.session_state.mode = "ENROLL_LANDMARK"
                        st.session_state.ingest_proc = tracking.launch_ingest(res["session_id"], ip_cam_url, fps=fps)
                        st.success("Enrolling landmark... show the landmark to camera.")
                    else:
                        st.error(res.get("message", "Failed to start enroll"))
            with colL2:
                if st.button("Stop Enroll Landmark"):
                    log_info(f"Button: Stop Enroll Landmark")
                    sid = st.session_state.session_id
                    if sid:
                        utils.api_post("/stop_session", {"session_id": sid})
                    tracking.stop_ingest_if_running()
                    st.session_state.session_id = None
                    st.session_state.mode = None
                    st.success("Landmark enrollment stopped.")
            
            # Check if enroll auto-stopped
            if st.session_state.get("session_id"):
                utils.auto_stop_if_session_ended()
        else:
            st.info("Add a landmark first, then enroll it.")
    else:
        st.info("Create an environment first.")

    st.divider()

    # TEST ENVIRONMENT (LIVE) - Live detection with landmark overlays
    st.markdown("#### Test Environment (Live)")
    if envs:
        env_choice_test = st.selectbox(
            "Select Environment to Test",
            envs,
            format_func=lambda x: x['environment_label'],
            key="test_env_choice"
        )
        
        colT1, colT2 = st.columns(2)
        with colT1:
            if st.button("Start Test Environment"):
                log_info(f"Button: Start Test Environment - {env_choice_test['environment_label']}")
                res = utils.api_post("/start_session", {
                    "user_id": uid,
                    "mode": "TEST_ENVIRONMENT",
                    "environment_id": env_choice_test["environment_id"]
                })
                if res.get("status") == "success":
                    tracking.stop_ingest_if_running()
                    st.session_state.session_id = res["session_id"]
                    st.session_state.mode = "TEST_ENVIRONMENT"
                    st.session_state.ingest_proc = tracking.launch_ingest(res["session_id"], ip_cam_url, fps=fps)
                    st.success("Testing environment... viewing live detections.")
                else:
                    st.error(res.get("message", "Failed to start test"))
        with colT2:
            if st.button("Stop Test Environment"):
                log_info(f"Button: Stop Test Environment")
                sid = st.session_state.session_id
                if sid:
                    utils.api_post("/stop_session", {"session_id": sid})
                tracking.stop_ingest_if_running()
                st.session_state.session_id = None
                st.session_state.mode = None
                st.success("Test stopped.")
    else:
        st.info("Create an environment first.")

    # 5. DATA CLEANUP
    st.markdown("### Data Cleanup")
    st.markdown("**Delete events older than:**")
    
    # WHY: Clean old data to free disk space
    # User-friendly dropdown: shows time in hours instead of minutes
    # Each option is a human-readable time period
    cleanup_label = st.selectbox(
        "Time period",
        list(CLEANUP_OPTIONS_HOURS.keys()),
        help="Select how old events must be to be deleted",
        key="cleanup_time_select"
    )
    cleanup_minutes = CLEANUP_OPTIONS_HOURS[cleanup_label]
    
    # Confirmation button
    if st.button("🗑️ Run Cleanup"):
        log_info(f"Button: Run Cleanup - older than {cleanup_label}")
        # WHY: Confirm before deleting - data loss is permanent
        with st.spinner(f"Deleting events older than {cleanup_label}..."):
            res = utils.api_post("/manual_cleanup", {"user_id": uid, "older_than_minutes": cleanup_minutes})
            deleted_count = res.get('deleted', 0)
            st.success(f"✓ Deleted {deleted_count} events older than {cleanup_label}")
