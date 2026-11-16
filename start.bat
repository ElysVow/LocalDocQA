@echo off
setlocal
cd /d "%~dp0"
title LocalDocQA
if not exist ".venv\Scripts\python.exe" (
    echo [LocalDocQA] .venv not found. See README section "Quick start".
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
echo.
echo [LocalDocQA] server stopped. Press any key to close.
pause >nul