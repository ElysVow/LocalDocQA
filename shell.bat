@echo off
cd /d "%~dp0"
title LocalDocQA CLI
call ".venv\Scripts\activate.bat"
echo.
echo LocalDocQA CLI ready. Examples:
echo   localdocqa stats
echo   localdocqa list -c demo
echo   localdocqa ask "experiment report requirements?" -c demo
echo   localdocqa ingest examples\sample_docs -c demo -f
echo.
cmd /k