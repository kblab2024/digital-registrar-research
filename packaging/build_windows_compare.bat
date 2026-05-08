@echo off
rem Build the Windows bundle for the Compare / Consensus tool.
rem Locked to NHC vs KPC, with_preann, consensus mode -> gold.
rem Output: packaging\dist\digital-registrar-compare-windows.zip
setlocal
set "PLATFORM=windows"
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
