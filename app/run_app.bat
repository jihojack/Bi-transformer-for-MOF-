@echo off
setlocal
title MOF Adsorption Predictor
cd /d "%~dp0.."
set "MOF_VENV=%~dp0..\.venv\Scripts\python.exe"
if exist "%MOF_VENV%" (
  "%MOF_VENV%" -m server.app
) else (
  py -3.11 -m server.app
)
if errorlevel 1 (
  echo.
  echo App execution failed. Review the error message above.
  pause
)
