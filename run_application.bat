@echo off
setlocal enabledelayedexpansion

call conda activate itemmem

REM ----- start backend -----
start "BACKEND" cmd /k "uvicorn backend_api:app --host 127.0.0.1 --port 8000"

timeout /t 2 >nul

REM ----- start ingest -----
start "INGEST" cmd /k "python ingest_ipcam.py"

timeout /t 2 >nul

REM ----- start UI -----
start "UI" cmd /k "streamlit run ui_streamlit.py"

endlocal
