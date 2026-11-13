@echo off
setlocal
cd /d "%~dp0"

rem ---- 1. Find a supported Python (3.12 preferred, then 3.11, then 3.10). Never 3.13 or newer.
set "PYV="
for %%V in (3.12 3.11 3.10) do (
  if not defined PYV (
    py -%%V -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PYV=%%V"
  )
)
if not defined PYV (
  echo Python 3.10, 3.11 or 3.12 was not found.
  echo Install Python 3.12 from python.org, then run this file again.
  echo Python 3.13 and newer cannot be used because MediaPipe does not support them.
  pause
  exit /b 1
)

rem ---- 2. If an old environment was built with an unsupported Python, delete it.
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo Existing environment uses an unsupported Python. Rebuilding it.
    rmdir /s /q ".venv"
  )
)

rem ---- 3. Create the environment with the chosen version.
if not exist ".venv\Scripts\python.exe" (
  echo Creating environment with Python %PYV%.
  py -%PYV% -m venv ".venv"
  if errorlevel 1 goto fail
)

rem ---- 4. Install packages and prove MediaPipe works. Done once, then skipped.
if not exist ".venv\.ready" (
  echo Installing packages. This needs internet and takes a few minutes.
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 goto fail
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto fail
  echo Checking the pose model.
  ".venv\Scripts\python.exe" -c "import mediapipe as mp; mp.solutions.pose.Pose(model_complexity=1).close(); print('Pose model OK')"
  if errorlevel 1 goto fail
  echo ready> ".venv\.ready"
)

rem ---- 5. Launch.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
exit /b 0

:fail
echo.
echo Setup failed. The environment was removed so the next run starts clean.
echo Check your internet connection, then run this file again.
if exist ".venv" rmdir /s /q ".venv"
pause
exit /b 1
