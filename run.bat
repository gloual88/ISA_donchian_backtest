@echo off
REM ISA trend-strategy dashboard launcher (ASCII-only, codepage-safe)
cd /d "%~dp0"
"%~dp0..\pykrx_venv\Scripts\python.exe" -m streamlit run app.py --server.port 8520
pause
