@echo off
title TISSAIA V14 COMMANDER
color 0a

echo [WARLORD] INITIALIZING TISSAIA V14...
echo -------------------------------------

:: Check API Key
if "%GOOGLE_API_KEY%"=="" (
    echo [CRITICAL] GOOGLE_API_KEY missing in Environment Variables!
    echo Please set it and restart.
    pause
    exit
)

:: Start Backend
echo [1/2] Launching Docker Core...
cd Tissaia_Project
start "TISSAIA BACKEND" docker compose up
cd ..

:: Wait for Server
echo Waiting for Neural Link...
timeout /t 10 >nul

:: Start Frontend
echo [2/2] Launching Necro UI...
cd tissaia_ui
flutter run -d windows

echo [INFO] System Active.
pause