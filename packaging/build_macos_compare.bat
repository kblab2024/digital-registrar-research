@echo off
rem Build the macOS (Apple Silicon, Mac mini M1/M2/M3+) bundle for the
rem Compare / Consensus tool. Locked to NHC vs KPC, with_preann,
rem consensus mode -> gold.
rem Output: packaging\dist\digital-registrar-compare-macos.tar.gz
setlocal
set "PLATFORM=macos"
set "ANNOTATOR_SET=compare"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0_build_common.ps1"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo Build failed with exit code %RC%.
    pause
    exit /b %RC%
)
echo.
pause
endlocal
