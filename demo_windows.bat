@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup_cpu.bat first.
  exit /b 1
)
set "demo_output=results\Demo_%RANDOM%_%RANDOM%"
call process_meeting.bat "demo\mixed.wav" --transcribe --out "%demo_output%"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" verify_demo.py "%demo_output%"
if errorlevel 1 exit /b 1
start "" "%demo_output%\listen.html"
