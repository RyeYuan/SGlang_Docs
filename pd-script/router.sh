#!/bin/bash

# Usage: bash router.sh [node-name-or-ip] [prefill-port] [decode-port] [router-port]
prefill_node="${1:-node20}"
decode_node="${2:-node22}"
prefill_port="${3:-10015}"
decode_port="${4:-10016}"
router_port="${5:-30000}"
bootstrap_port="${SGLANG_DISAGGREGATION_BOOTSTRAP_PORT:-8998}"
prom_port="${SGLANG_ROUTER_PROMETHEUS_PORT:-29000}"

# HTTP/bootstrap addresses. Use the 10.16.1.x management network: it is the
# only one every host shares. 13.13.x does not exist on nmz104/105/107/110, so
# a 13.13.7.x prefill URL cannot be reached from a decode there (the RDMA NIC
# itself is chosen by run_dpsk-v41.sh, see IB_RAIL / PD_PEER).
resolve_node_ip() {
    case "$1" in
        "") hostname -I | awk '{print $1}' ;;
        node7|nmz7|node15|nmz15|node18|nmz18|node20|nmz20|node21|nmz21|node22|nmz22|\
        node26|nmz26|node28|nmz28|node104|nmz104|node105|nmz105|node107|nmz107|node110|nmz110)
            echo "10.16.1.${1//[!0-9]/}" ;;
        sglang2) echo 10.16.1.33 ;;
        *.*.*.*) echo "$1" ;;
        *)
            echo "Invalid host identifier: $1" >&2
            return 1
            ;;
    esac
}

p_ip=$(resolve_node_ip "$prefill_node") || exit 1
d_ip=$(resolve_node_ip "$decode_node") || exit 1

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

# Publish the exact endpoints used by this router. The profiling benchmark reads
# this shared state instead of maintaining a second, easily stale host mapping.
router_state_file="${PD_ROUTER_STATE_FILE:-/home/scripts/SGlang_Docs/pd-script/.router_endpoints.env}"
router_state_tmp="${router_state_file}.tmp.$$"
{
    printf 'ROUTER_PREFILL_URL=%q\n' "http://${p_ip}:${prefill_port}"
    printf 'ROUTER_DECODE_URL=%q\n' "http://${d_ip}:${decode_port}"
    printf 'ROUTER_PORT=%q\n' "$router_port"
} > "$router_state_tmp"
mv -f "$router_state_tmp" "$router_state_file"

python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "http://${p_ip}:${prefill_port}" "${bootstrap_port}" \
    --decode "http://${d_ip}:${decode_port}" \
    --host 0.0.0.0 \
    --port "${router_port}" \
    --prometheus-port "${prom_port}"
