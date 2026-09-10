#!/usr/bin/env bash
# ============================================================
#  AnimeFinder 服务管理脚本 (Linux / macOS / Git Bash)
#  用法:
#    ./service.sh start     启动服务(后台)
#    ./service.sh stop      停止服务
#    ./service.sh restart   重启服务
#    ./service.sh status    查看运行状态
#    ./service.sh           不带参数 = start
#
#  说明: 服务默认监听 8000 端口, 日志写入 backend/server.log,
#        PID 写入 backend/server.pid
# ============================================================

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
PIDFILE="$BACKEND/server.pid"
LOGFILE="$BACKEND/server.log"
PORT="${PORT:-8000}"

# ---- 挑选可用的 Python 解释器 ----
pick_python() {
    local candidates=(
        "E:/anaconda3/python.exe"
        "$HOME/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
        "$HOME/.workbuddy/binaries/python/envs/default/bin/python"
        "python3"
        "python"
    )
    for c in "${candidates[@]}"; do
        if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then
            if "$c" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >/dev/null 2>&1; then
                echo "$c"
                return 0
            fi
        fi
    done
    return 1
}

PYEXE="$(pick_python)" || {
    echo "[错误] 未找到 Python 3.10+, 请先安装并加入 PATH"
    exit 1
}

# ---- 检测服务是否运行 ----
# 输出: RUNNING=1/0  OLDPID=xxx
check_running() {
    RUNNING=0
    OLDPID=""
    if [ -f "$PIDFILE" ]; then
        local fp
        fp="$(cat "$PIDFILE" 2>/dev/null | tr -d '[:space:]')"
        if [ -n "$fp" ] && kill -0 "$fp" 2>/dev/null; then
            RUNNING=1
            OLDPID="$fp"
            return 0
        fi
    fi
    # 回退: 按端口查找
    local p
    p="$(netstat -ano 2>/dev/null | grep ":$PORT " | grep LISTENING | awk '{print $NF}' | head -1)"
    if [ -n "$p" ]; then
        RUNNING=1
        OLDPID="$p"
    fi
    return 0
}

do_start() {
    check_running
    if [ "$RUNNING" = "1" ]; then
        echo "[提示] 服务已在运行 (PID $OLDPID, 端口 $PORT)"
        echo "        如需重启请执行: ./service.sh restart"
        return 0
    fi

    echo "[1/2] 检查依赖..."
    if ! "$PYEXE" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
        echo "      缺少依赖, 正在安装 (首次较慢)..."
        "$PYEXE" -m pip install -r "$ROOT/requirements.txt" || {
            echo "[错误] 依赖安装失败, 请手动执行:"
            echo "       $PYEXE -m pip install -r $ROOT/requirements.txt"
            return 1
        }
        echo "      依赖安装完成"
    else
        echo "      依赖正常"
    fi

    echo "[2/2] 启动服务 (端口 $PORT)..."
    cd "$BACKEND" || return 1
    nohup "$PYEXE" -m uvicorn main:app --host 127.0.0.1 --port "$PORT" > "$LOGFILE" 2>&1 &
    local newpid=$!

    # 等待端口就绪
    local i=0
    while [ $i -lt 15 ]; do
        sleep 1
        if netstat -ano 2>/dev/null | grep ":$PORT " | grep -q LISTENING; then
            break
        fi
        i=$((i + 1))
    done

    if netstat -ano 2>/dev/null | grep ":$PORT " | grep -q LISTENING; then
        echo "$newpid" > "$PIDFILE"
        echo ""
        echo "[成功] 服务已启动"
        echo "       访问地址: http://localhost:$PORT"
        echo "       接口文档: http://localhost:$PORT/docs"
        echo "       PID: $newpid"
        echo "       日志: $LOGFILE"
    else
        echo "[警告] 服务可能未正常启动, 请查看日志: $LOGFILE"
        tail -20 "$LOGFILE" 2>/dev/null
        return 1
    fi
}

do_stop() {
    check_running
    if [ "$RUNNING" = "0" ]; then
        echo "[提示] 服务未在运行"
        return 0
    fi

    echo "正在停止服务 (PID $OLDPID)..."
    kill -9 "$OLDPID" 2>/dev/null || taskkill //PID "$OLDPID" //F >/dev/null 2>&1
    sleep 1

    check_running
    if [ "$RUNNING" = "0" ]; then
        rm -f "$PIDFILE"
        echo "[成功] 服务已停止, 端口 $PORT 已释放"
    else
        echo "[警告] 停止失败, 请手动结束进程 PID $OLDPID"
        return 1
    fi
}

do_status() {
    check_running
    if [ "$RUNNING" = "1" ]; then
        echo "[状态] 运行中"
        echo "       PID: $OLDPID"
        echo "       端口: $PORT"
        echo "       地址: http://localhost:$PORT"
    else
        echo "[状态] 未运行"
    fi
}

case "${1:-start}" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; sleep 1; do_start ;;
    status)  do_status ;;
    *)
        echo "[错误] 未知参数: $1"
        echo "可用参数: start / stop / restart / status"
        exit 1
        ;;
esac
