@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

REM ============================================================
REM  AnimeFinder 服务管理脚本 (Windows)
REM  用法:
REM    service.bat start     启动服务(后台)
REM    service.bat stop      停止服务
REM    service.bat restart   重启服务
REM    service.bat status    查看运行状态
REM    service.bat           不带参数 = start
REM
REM  说明: 服务默认监听 8000 端口, 日志写入 backend\server.log,
REM        PID 写入 backend\server.pid
REM ============================================================

cd /d "%~dp0"
set "ROOT=%~dp0"
set "BACKEND=%ROOT%backend"
set "PIDFILE=%BACKEND%\server.pid"
set "LOGFILE=%BACKEND%\server.log"
set "PORT=8000"

REM ---- 挑选可用的 Python 解释器 ----
set "PYEXE="
REM 1) 优先 anaconda (依赖最全)
if exist "E:\anaconda3\python.exe" (
    set "PYEXE=E:\anaconda3\python.exe"
)
REM 2) 受管 venv
if not defined PYEXE (
    if exist "%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe" (
        set "PYEXE=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
    )
)
REM 3) PATH 中的 python
if not defined PYEXE (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if not defined PYEXE set "PYEXE=%%i"
    )
)
if not defined PYEXE (
    echo [错误] 未找到 Python, 请先安装 Python 3.10+ 并加入 PATH
    exit /b 1
)

set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=start"

if /i "%ACTION%"=="start"   goto :start
if /i "%ACTION%"=="stop"    goto :stop
if /i "%ACTION%"=="restart" goto :restart
if /i "%ACTION%"=="status"  goto :status
echo [错误] 未知参数: %ACTION%
echo 可用参数: start / stop / restart / status
exit /b 1

REM ==================== 启动 ====================
:start
call :check_running
if "!RUNNING!"=="1" (
    echo [提示] 服务已在运行 (PID !OLDPID!, 端口 %PORT%)
    echo        如需重启请执行: service.bat restart
    goto :eof
)

echo [1/2] 检查依赖...
"%PYEXE%" -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo       缺少依赖, 正在安装 (首次较慢, 请耐心等待)...
    "%PYEXE%" -m pip install -r "%ROOT%requirements.txt" >nul 2>&1
    if errorlevel 1 (
        echo [错误] 依赖安装失败, 请手动执行:
        echo        "%PYEXE%" -m pip install -r "%ROOT%requirements.txt"
        exit /b 1
    )
    echo       依赖安装完成
) else (
    echo       依赖正常
)

echo [2/2] 启动服务 (端口 %PORT%)...
cd /d "%BACKEND%"
start "AnimeFinder-Server" /min "%PYEXE%" -m uvicorn main:app --host 127.0.0.1 --port %PORT% > "%LOGFILE%" 2>&1

REM 等待服务起来并记录 PID
set "NEWPID="
for /l %%i in (1,1,15) do (
    if not defined NEWPID (
        timeout /t 1 /nobreak >nul
        for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
            if not defined NEWPID set "NEWPID=%%p"
        )
    )
)

if defined NEWPID (
    echo !NEWPID!> "%PIDFILE%"
    echo.
    echo [成功] 服务已启动
    echo        访问地址: http://localhost:%PORT%
    echo        接口文档: http://localhost:%PORT%/docs
    echo        PID: !NEWPID!
    echo        日志: %LOGFILE%
) else (
    echo [警告] 服务可能未正常启动, 请查看日志: %LOGFILE%
    type "%LOGFILE%"
)
goto :eof

REM ==================== 停止 ====================
:stop
call :check_running
if "!RUNNING!"=="0" (
    echo [提示] 服务未在运行
    goto :eof
)

echo 正在停止服务 (PID !OLDPID%)...
taskkill /PID !OLDPID! /F >nul 2>&1
if errorlevel 1 (
    echo       PID 方式失败, 改用端口方式停止...
    for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
        taskkill /PID %%p /F >nul 2>&1
    )
)
timeout /t 1 /nobreak >nul

call :check_running
if "!RUNNING!"=="0" (
    if exist "%PIDFILE%" del "%PIDFILE%" >nul 2>&1
    echo [成功] 服务已停止, 端口 %PORT% 已释放
) else (
    echo [警告] 停止失败, 请手动结束进程 PID !OLDPID!
)
goto :eof

REM ==================== 重启 ====================
:restart
call :stop
timeout /t 1 /nobreak >nul
goto :start

REM ==================== 状态 ====================
:status
call :check_running
if "!RUNNING!"=="1" (
    echo [状态] 运行中
    echo        PID: !OLDPID!
    echo        端口: %PORT%
    echo        地址: http://localhost:%PORT%
) else (
    echo [状态] 未运行
)
goto :eof

REM ==================== 辅助: 检测运行状态 ====================
:check_running
set "RUNNING=0"
set "OLDPID="
REM 优先按 PID 文件判断
if exist "%PIDFILE%" (
    set /p FILEPID=<"%PIDFILE%"
    if defined FILEPID (
        tasklist /FI "PID eq !FILEPID!" 2>nul | findstr /i "python" >nul
        if not errorlevel 1 (
            set "RUNNING=1"
            set "OLDPID=!FILEPID!"
            goto :eof
        )
    )
)
REM 回退: 按端口判断
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
    if not defined OLDPID (
        set "RUNNING=1"
        set "OLDPID=%%p"
    )
)
goto :eof
