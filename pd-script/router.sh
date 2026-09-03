#!/bin/bash

# Usage: bash router.sh [node-name-or-ip] [prefill-port] [decode-port] [router-port]
prefill_node="${1:-node20}"
decode_node="${2:-node22}"
prefill_port="${3:-10015}"
decode_port="${4:-10016}"
router_port="${5:-30000}"
bootstrap_port="${SGLANG_DISAGGREGATION_BOOTSTRAP_PORT:-8998}"
prom_port="${SGLANG_ROUTER_PROMETHEUS_PORT:-29000}"

if [[ -z "$prefill_node" ]]; then
    p_ip="$(hostname -I | awk '{print $1}')"
else
    case "$prefill_node" in
        node20) p_ip="13.13.7.20" ;;
        node22) p_ip="13.13.7.22" ;;
        node26) p_ip="10.16.1.26" ;;
        sglang2) p_ip="10.16.1.33" ;;
        node104) p_ip="10.16.1.104" ;;
        node110) p_ip="10.16.1.110" ;;
        *.*.*.*) p_ip="$prefill_node" ;;
        *)
            echo "Invalid host identifier: $prefill_node"
            exit 1
            ;;
    esac
fi

if [[ -z "$decode_node" ]]; then
    d_ip="$(hostname -I | awk '{print $1}')"
else
    case "$decode_node" in
        node20) d_ip="10.16.1.20" ;;
        node22) d_ip="10.16.1.22" ;;
        node26) d_ip="10.16.1.26" ;;
        sglang2) d_ip="10.16.1.33" ;;
        node104) d_ip="10.16.1.104" ;;
        node110) d_ip="10.16.1.110" ;;
        *.*.*.*) d_ip="$decode_node" ;;
        *)
            echo "Invalid host identifier: $decode_node"
            exit 1
            ;;
    esac
fi

# 本机 ip_local_port_range = 1024-65535 且没有 ip_local_reserved_ports，
# 内核会把 29000 / 30000 这类端口随机分配给其它进程的出站连接做源端口，
# 撞上时 router 绑 Prometheus exporter 会 panic (EADDRINUSE)。启动前先探测。
port_free() {
    python3 -c "
import socket, sys
s = socket.socket()
try:
    s.bind(('0.0.0.0', int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
" "$1"
}

if ! port_free "$router_port"; then
    echo "ERROR: router port ${router_port} is already in use (very likely taken as an ephemeral"
    echo "       source port by another process). Free it, or pass a different port as \$5."
    echo "       Check with: ss -tanp | grep ':${router_port}'"
    exit 1
fi

if ! port_free "$prom_port"; then
    for try in $(seq "$prom_port" $((prom_port + 200))); do
        if port_free "$try"; then
            echo "WARN: prometheus port ${prom_port} is occupied, falling back to ${try}"
            prom_port="$try"
            break
        fi
    done
    if ! port_free "$prom_port"; then
        echo "ERROR: no free prometheus port found near ${prom_port}"
        exit 1
    fi
fi

echo "Starting PD router: prefill=${p_ip}:${prefill_port} (bootstrap ${bootstrap_port}), decode=${d_ip}:${decode_port}, router=:${router_port}, prometheus=:${prom_port}"
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "http://${p_ip}:${prefill_port}" "${bootstrap_port}" \
    --decode "http://${d_ip}:${decode_port}" \
    --host 0.0.0.0 \
    --port "${router_port}" \
    --prometheus-port "${prom_port}"
