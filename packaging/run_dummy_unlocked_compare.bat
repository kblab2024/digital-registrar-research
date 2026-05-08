@echo off
setlocal
cd /d "%~dp0"

rem --- Skip Streamlit's one-time "please enter your email" prompt on first launch.
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
    (
        echo [general]
        echo email = ""
    ) > "%USERPROFILE%\.streamlit\credentials.toml"
)

set "REGISTRAR_ANNOTATE_BASE_DIR=%~dp0dummy"
set "PYTHONPATH=%~dp0app"
set "PYTHONDONTWRITEBYTECODE=1"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

echo ================================================================
echo   Digital Registrar - Compare / Consensus (dummy, unlocked)
echo ================================================================
echo   Data root : dummy\
echo   Mode      : full UI (mode/annotator/dataset pickers visible)
echo   URL       : http://localhost:8501
echo.
echo   Use this launcher to explore the full Compare/Consensus UI
echo   against the dummy skeleton tree (no real data).
echo   Keep this window open; close or Ctrl+C to stop.
echo ================================================================
echo.

"%~dp0python\python.exe" -m streamlit run ^
    "%~dp0app\digital_registrar_research\annotation\compare_app_canonical.py" ^
    --server.address=localhost ^
    --server.port=8501

echo.
echo Streamlit has stopped.
pause
endlocal
