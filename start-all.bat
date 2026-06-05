@echo off
echo === AIimg Prompt Studio - Start All Services ===

echo [1/3] Starting ComfyUI...
start "ComfyUI" /D "D:\ComfyUI-portable\ComfyUI_windows_portable" cmd /c "set PYTHONUTF8=1 && set PYTHONIOENCODING=utf-8 && python_embeded\python.exe ComfyUI\main.py --windows-standalone-build"

echo [2/3] Waiting 5s for ComfyUI to initialize...
timeout /t 5 /nobreak >nul

echo [3/3] Starting Backend...
start "AIimg Backend" /D "C:\Users\a0929\Desktop\AIimg-prompt-studio\services\backend" cmd /c "C:\Users\a0929\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe run uvicorn app.main:app --host 127.0.0.1 --port 8000"

echo.
echo [4/4] Starting Frontend...
start "AIimg Frontend" /D "C:\Users\a0929\Desktop\AIimg-prompt-studio\apps\desktop" cmd /c "C:\Program Files\nodejs\npm.cmd run dev"

echo.
echo All services starting. Open http://127.0.0.1:1420 in your browser.
pause
