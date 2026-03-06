"""Query UI: text and voice based search, results display."""
import os
import streamlit as st
from ui_components import utils
from logger import log_info, log_error
import speech_recognition as sr


def listen_and_transcribe() -> str:
    """Record from microphone and transcribe via Google Web Speech API. Requires internet."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Reduce background noise
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)  # 5 sec max recording
    return recognizer.recognize_google(audio)  # Google Web Speech API


def show_results(data, obj):
    if not data or not obj:
        return
    results = data.get("results", [])
    if not results:
        st.warning("No saved sightings found.")
        return

    st.subheader(f"Results for: {obj}")
    idx = st.radio(
        "Select sighting",
        options=list(range(len(results))),
        format_func=lambda i: f"{utils.ordinal(i)} -- {utils.pretty_time(results[i]['timestamp'])}",
        key="sighting_radio"
    )
    r = results[idx]
    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"**{utils.ordinal(idx)}**")
        st.caption(f"Location: {r.get('location_text', 'unknown')}")
    with right:
        st.markdown(f"**{utils.pretty_time(r['timestamp'])}**")
    path = (r.get("image_path") or "").strip()
    if path and os.path.exists(path):
        st.image(path, width="stretch")
    elif path:
        st.error("Image file not found on disk.")
        st.code(path)
        log_error("Image missing on disk: %s", path)


def query_controls(uid):
    """Render the query section and handle text/voice search."""
    st.markdown("### Query (Last Seen)")

    # Voice button populates the text box; user then edits and hits Search
    c1, c2 = st.columns([1, 1])
    with c2:
        if st.button("Voice Input", key="voice_btn"):
            log_info("[QUERY] Voice Input clicked")
            with st.spinner("Listening... speak now. (Requires internet)"):
                try:
                    text = listen_and_transcribe()
                    log_info("[QUERY] Voice transcription: '%s'", text)
                    st.session_state["query_input"] = text
                    st.rerun()
                except sr.WaitTimeoutError:
                    st.error("No speech detected. Please try again.")
                    log_error("[QUERY] Voice: timeout - no speech detected")
                except sr.UnknownValueError:
                    st.error("Could not understand the audio. Please speak clearly.")
                    log_error("[QUERY] Voice: could not understand audio")
                except sr.RequestError as e:
                    st.error("Voice input requires an internet connection.")
                    log_error("[QUERY] Voice: Google API error - %s", e)
                except Exception as e:
                    st.error(f"Voice error: {e}")
                    log_error("[QUERY] Voice error: %s", e)
                    
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = "Where did I last see my watch?"

    query_text = st.text_input(
        "Query",
        label_visibility="collapsed",
        key="query_input"
    )

    k = st.selectbox("Number of results", options=list(range(1, 11)), index=2)

    with c1:
        search_clicked = st.button("Search", key="search_btn")

    if search_clicked:
        if not query_text.strip():
            st.warning("Type a question first.")
        else:
            # Match query text against user's registered object labels (substring check)
            objs = utils.api_get("/dashboard", {"user_id": uid}).get("objects", [])
            labels = [o['user_label'] for o in objs]
            obj = None
            for lab in labels:
                if lab.lower() in query_text.lower():  # Case-insensitive substring match
                    obj = lab
                    break
            # Fallback: generic class keywords
            if not obj:
                t = query_text.lower()
                if "watch" in t:
                    obj = "Watch"
                elif "wallet" in t:
                    obj = "Wallet"
                elif "key" in t:
                    obj = "Bike Key"
            if not obj:
                st.error("Could not find a matching object. Mention the label name (e.g. 'black wallet').")
            else:
                data = utils.api_post("/query", {"user_id": uid, "user_label": obj, "k": k})
                log_info("[Query] label='%s' k=%d", obj, k)
                st.session_state["last_data"] = data
                st.session_state["last_obj"] = obj

    show_results(st.session_state.get("last_data"), st.session_state.get("last_obj"))