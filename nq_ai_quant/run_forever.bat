@echo off
REM ===========================================================================
REM  Leave this running. It searches, evolves and reports on its own.
REM  If Python crashes for any reason it restarts after 30 seconds and picks up
REM  exactly where it left off - nothing already tried is ever tried again.
REM
REM  Double-click this file, or run it from a terminal.
REM  Close the window (or Ctrl+C twice) to stop.
REM ===========================================================================
cd /d "%~dp0"
title NQ strategy search

:loop
echo.
echo [%date% %time%] starting search...
python run_search.py --config config.yaml
echo [%date% %time%] process exited with code %errorlevel%.
if %errorlevel%==0 (
    echo Clean exit. Stopping.
    goto end
)
echo Restarting in 30 seconds. Press Ctrl+C to stop for good.
timeout /t 30 /nobreak >nul
goto loop

:end
echo.
echo Leaderboard: %~dp0results\leaderboard.html
pause
