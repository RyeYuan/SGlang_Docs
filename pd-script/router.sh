#!/bin/bash

# Usage: bash router.sh [node-name-or-ip] [prefill-port] [decode-port] [router-port]
prefill_node="${1:-node20}"
decode_node="${2:-node22}"
prefill_port="${3:-10015}"
decode_port="${4:-10016}"
router_port="${5:-30000}"
bootstrap_port="${SGLANG_DISAGGREGATION_BOOTSTRAP_PORT:-8998}"

if [[ -z "$prefill_node" ]]; then
    p_ip="$(hostname -I | awk '{print $1}')"
else
    case "$prefill_node" in
        node20) p_ip="13.13.7.20" ;;
        node22) p_ip="13.13.7.22" ;;
        node26) p_ip="13.13.7.26" ;;
        sglang2) p_ip="10.16.1.33" ;;
        node36) p_ip="10.16.1.36" ;;
        node39) p_ip="10.16.1.39" ;;
        node42) p_ip="10.16.1.42" ;;
        node43) p_ip="10.16.1.43" ;;
        node46) p_ip="10.16.1.46" ;;
        node51) p_ip="10.16.1.51" ;;
        node54) p_ip="10.16.1.54" ;;
        node55) p_ip="10.16.1.55" ;;
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
        node26) d_ip="13.13.7.26" ;;
        sglang2) d_ip="10.16.1.33" ;;
        node36) d_ip="10.16.1.36" ;;
        node39) d_ip="10.16.1.39" ;;
        node42) d_ip="10.16.1.42" ;;
        node43) d_ip="10.16.1.43" ;;
        node46) d_ip="10.16.1.46" ;;
        node51) d_ip="10.16.1.51" ;;
        node54) d_ip="10.16.1.54" ;;
        node55) d_ip="10.16.1.55" ;;
        *.*.*.*) d_ip="$decode_node" ;;
        *)
            echo "Invalid host identifier: $decode_node"
            exit 1
            ;;
    esac
fi

echo "Starting PD router: prefill=${p_ip}:${prefill_port} (bootstrap ${bootstrap_port}), decode=${d_ip}:${decode_port}, router=:${router_port}"
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "http://${p_ip}:${prefill_port}" "${bootstrap_port}" \
    --decode "http://${d_ip}:${decode_port}" \
    --host 0.0.0.0 \
    --port "${router_port}"
