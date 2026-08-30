#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="${STATE_DIR:-/workspace/qwen-vllm}"
PID_FILE="${STATE_DIR}/server.pid"
LOG_FILE="${STATE_DIR}/server.log"

mkdir -p "$STATE_DIR"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(<"$PID_FILE")"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "vLLM is already running with PID ${existing_pid}."
    exit 0
  fi
fi

nohup /opt/qwen/start.sh >>"$LOG_FILE" 2>&1 &
server_pid=$!
echo "$server_pid" >"$PID_FILE"
echo "vLLM started in the background with PID ${server_pid}."
echo "Logs: ${LOG_FILE}"
