@echo off
echo.
echo ========================================
echo   Opening Emotion Classification App
echo ========================================
echo.
echo Opening frontend application...
start "" "%~dp0frontend\index.html"
echo.
echo Application opened in your default browser!
echo Backend is running at: http://localhost:5000
echo.
pause
