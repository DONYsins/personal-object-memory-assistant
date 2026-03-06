@echo off
setlocal enabledelayedexpansion

call conda activate itemmem

REM ----- start backend -----
start "BACKEND" cmd /k "uvicorn backend_api:app --reload"

timeout /t 12 >nul


REM ----- start UI -----
start "UI" cmd /k "streamlit run ui_streamlit.py"

endlocal
