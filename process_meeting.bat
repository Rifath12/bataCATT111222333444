@echo off
setlocal
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0process_meeting.py" %*
) else (
  py -3.12 "%~dp0process_meeting.py" %*
)
