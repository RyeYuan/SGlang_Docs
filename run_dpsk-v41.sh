#!/bin/bash

# DeepSeek-V4.1 launcher, modelled after run_dpsk-v4.sh:
#   - echoes every environment variable it exports and the final CLI line
#   - PD_MODE=none|prefill|decode picks TP / P-side CP+EP / D-side DP+EP args
#   - PD_OPEN=1 adds the disaggregation transport on top of that topology;
#     PD_OPEN=0 (default) runs the same topology standalone on one node
#   - HCU_NUM>8 switches to a multi-node launch (NNODES / NODE_RANK)
# source /home/unset_proxy.sh  # uncomment if a proxy breaks the PD bootstrap server

set -o pipefail

export PYTHONPATH=/home/proj_dpsk-v4/sglang-das/python:${PYTHONPATH:-}

readonly DEEPEP_CONFIG=/home/proj_dpsk-v4/configs/deepep_IntraConfig.json
default_ib_devices=mlx5_0
case "$(hostname -s)" in
    nmz20|nmz22) default_ib_devices=mlx5_2 ;;
esac
readonly IB_DEVICES=${IB_DEVICES:-$default_ib_devices}
readonly DEFAULT_MODEL_PATH=/module/DeepSeek-V4.1-Flash-Channel-FP8

die() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  bash run_dpsk-v41.sh [PORT] [MODEL_PATH] [RANK_ID] [HOST]

Positional arguments:
  PORT        Service port. Default: 30000.
  MODEL_PATH  Model checkpoint. Default: DeepSeek-V4.1-Flash-Channel-FP8.
  RANK_ID     Node rank for multi-node launch; ignored on one node.
  HOST        Optional distributed-init host alias. Supported: node18, node20,
              node22, node26, node104, node107, node110, sglang2.
              Default: first local IP address.

Core selectors:
  HCU_NUM=8                 Number of HCUs / TP size (default: 8).
  PD_MODE=none|prefill|decode
                            none    Pure TP + DSpark (default, original profile).
                            prefill CP + EP prefill topology.
                            decode  DP + EP + DP attention/LM head topology.
  PD_OPEN=0|1               Attach the PD disaggregation transport to the
                            PD_MODE=prefill/decode topology (default: 0).
                            1  Real PD separation: mooncake/UCX transfer, its
                               own --dist-init-addr port, bootstrap server.
                            0  Same topology, standalone: the P or D profile
                               runs on its own with no transfer arguments.
                            Requires PD_MODE=prefill or PD_MODE=decode.
  MOE_MODE=none|triton|deepep|megamoe
                            MoE runtime for the P/D paths (default: deepep when
                            PD_MODE is set, none otherwise). P/D paths require
                            deepep or megamoe.
  PC_ENABLE=0|1             Radix cache (default: 0). PD_OPEN=1 decode with
                            PC_ENABLE=1 enables the DSV4 decode radix path.
  DSPARK_BLOCK_SIZE=N       DSpark block size (default: 5).
  DSPARK_DRAFT_MODEL_PATH=P Draft checkpoint (default: MODEL_PATH).
  MEM_FRACTION_STATIC=F     KV cache memory fraction (default: 0.8).
  MAX_PREFILL_TOKENS=N      Optional prefill token cap on the P side.
  MAX_RUNNING_REQUESTS=N    Scheduler concurrency cap (default: 32).
  IB_DEVICES=LIST           PD/distributed RDMA devices (default: mlx5_2 on
                            nmz20/nmz22; mlx5_0 elsewhere).
  NNODES=N                  Multi-node count (default: 2).
  NODE_RANK=N               Overrides positional RANK_ID.
  DIST_INIT_PORT=N          Generic multi-node init port (default: 6245).
  PD_PREFILL_DIST_PORT=N    PD P-side init port (default: 6144).
  PD_DECODE_DIST_PORT=N     PD D-side init port (default: 6244).
  DECODE_DIST_INIT_PORT=N   Standalone decode init port (default: PORT + 733).
  DRY_RUN=1                 Print the resolved command without starting.

Distributed init-address precedence (exactly one --dist-init-addr is emitted):
  PD P side (PD_OPEN=1)            HOST:PD_PREFILL_DIST_PORT
  PD D side (PD_OPEN=1)            HOST:PD_DECODE_DIST_PORT
  Multi-node, PD_OPEN=0            HOST:DIST_INIT_PORT
  Single-node decode, PD_OPEN=0    127.0.0.1:DECODE_DIST_INIT_PORT
  Single-node TP (PD_MODE=none) does not set an explicit init address.

Examples:
  # Pure TP + DSpark (original profile)
  bash run_dpsk-v41.sh 30000 /module/DeepSeek-V4.1-Flash-Channel-FP8

  # PD P side
  PD_OPEN=1 PD_MODE=prefill bash run_dpsk-v41.sh 30000 /module/DeepSeek-V4.1-Flash-Channel-FP8

  # PD D side with radix cache
  PD_OPEN=1 PD_MODE=decode PC_ENABLE=1 \
    bash run_dpsk-v41.sh 30001 /module/DeepSeek-V4.1-Flash-Channel-FP8

  # Standalone P/D topologies: PD_OPEN defaults to 0, so the same parallel
  # setup runs as a single service with no disaggregation transport.
  PD_MODE=prefill bash run_dpsk-v41.sh 30000 /module/DeepSeek-V4.1-Flash-Channel-FP8
  PD_MODE=decode bash run_dpsk-v41.sh 30001 /module/DeepSeek-V4.1-Flash-Channel-FP8

  # Generic 2-node launch (run once per node with RANK_ID 0/1)
  HCU_NUM=16 NNODES=2 bash run_dpsk-v41.sh 30000 /module/DeepSeek-V4.1-Flash-Channel-FP8 0 node26
  HCU_NUM=16 NNODES=2 bash run_dpsk-v41.sh 30000 /module/DeepSeek-V4.1-Flash-Channel-FP8 1 node26

  # Two-node PD: P uses 6144, D uses 6244 by default.
  # Repeat each command with RANK_ID=1 on the worker node.
  HCU_NUM=16 NNODES=2 PD_OPEN=1 PD_MODE=prefill \
    bash run_dpsk-v41.sh 30000 /module/DeepSeek-V4.1-Flash-Channel-FP8 0 node26
  HCU_NUM=16 NNODES=2 PD_OPEN=1 PD_MODE=decode \
    bash run_dpsk-v41.sh 30001 /module/DeepSeek-V4.1-Flash-Channel-FP8 0 node26

  # Inspect the generated command without starting a service
  DRY_RUN=1 PD_OPEN=1 PD_MODE=prefill bash run_dpsk-v41.sh 30000 /module/DeepSeek-V4.1-Flash-Channel-FP8
EOF
}

resolve_ip() {
    local host_arg=$1 detected_ip
    if [[ -z "$host_arg" ]]; then
        detected_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
        if [[ -z "$detected_ip" ]]; then
            detected_ip=$(hostname -i 2>/dev/null | awk '{print $1}')
        fi
        [[ -n "$detected_ip" ]] || die "Could not determine the local IP address"
        echo "$detected_ip"
        return
    fi

    case "$host_arg" in
        node18) echo 13.13.2.18 ;;
        node20) echo 13.13.2.20 ;;
        node22) echo 13.13.2.22 ;;
        node26) echo 13.13.2.26 ;;
        node104) echo 12.12.12.104 ;;
        node105) echo 12.12.12.105 ;;
        node107) echo 12.12.12.107 ;;
        node110) echo 12.12.12.110 ;;
        sglang2) echo 10.16.1.33 ;;
        *) die "Invalid HOST=$host_arg (expected: node18|node20|node22|node26|node104|node107|node110|sglang2)" ;;
    esac
}

resolve_network_interface() {
    case "$1" in
        nmz26|nmz20|nmz22|nmz18|nmz15|nmz21|nmz7|nmz28) echo ens66f1np1 ;;
        nmz105|nmz107|nmz104|nmz110) echo ens65f0np0 ;;
        sglang5) echo eth0 ;;
        sglang8) echo enp113s0f0np0 ;;
        sglang6) echo eth10 ;;
        *) echo ens19f0 ;;
    esac
}

require_file() {
    [[ -f "$1" ]] || die "$2: $1"
}

append_deepep_args() {
    DEFAULT_ARGS+=(
        --moe-a2a-backend deepep
        --deepep-config "$DEEPEP_CONFIG"
    )
}

append_pd_args() {
    local mode=$1
    DEFAULT_ARGS+=(
        --disaggregation-ib-device "$IB_DEVICES"
        --disaggregation-mode "$mode"
        --disaggregation-transfer-backend mooncake
        --disaggregation-bootstrap-port "$pd_bootstrap_port"
    )
}

resolve_dist_init_addr() {
    # Sole source of --dist-init-addr. A PD pair needs stable, distinct ports
    # so both services can coexist on one host; a generic multi-node group
    # uses its own port; a standalone decode keeps the historical
    # port-derived local address; plain single-node TP sets none at all.
    # Keep the if/elif shape: a bare case/AND-list would return 1 (and kill
    # the script via `|| exit`) when no branch matches.
    if [[ "$pd_open" == 1 ]]; then
        case "$pd_mode" in
            prefill) echo "$ip:$pd_prefill_dist_port" ;;
            decode) echo "$ip:$pd_decode_dist_port" ;;
            *) die "PD_OPEN=1 requires PD_MODE=prefill or PD_MODE=decode" ;;
        esac
    elif (( is_multi_node )); then
        echo "$ip:$multi_node_dist_port"
    elif [[ "$pd_mode" == decode ]]; then
        echo "127.0.0.1:$local_decode_dist_port"
    fi
}

append_distributed_init_args() {
    if (( is_multi_node )); then
        DEFAULT_ARGS+=(--nnodes "$nnodes" --node-rank "$node_rank")
    fi
    [[ -z "$dist_init_addr" ]] || DEFAULT_ARGS+=(--dist-init-addr "$dist_init_addr")
}

# P side: CP + EP prefill. The DSpark draft only injects target hidden/KV
# states and must not initialize the decode-side DP/LM-head path, so the
# draft runs through the standalone triton path while the CP/EP target uses
# the selected A2A backend.
append_prefill_args() {
    DEFAULT_ARGS+=(
        --enable-prefill-cp
        --cp-strategy interleave
        --dp 1
        --attn-cp-size "$tp_size"
    )
    case "$moe_mode" in
        none|triton)
            DEFAULT_ARGS+=(
                --moe-runner-backend triton
                --speculative-moe-a2a-backend none
                --speculative-moe-runner-backend triton
            )
            ;;
        aiter)
            DEFAULT_ARGS+=(
                --moe-runner-backend aiter
                --speculative-moe-a2a-backend none
                --speculative-moe-runner-backend triton
            )
            ;;            
        deepep)
            append_deepep_args
            DEFAULT_ARGS+=(
                --deepep-mode normal 
                --moe-runner-backend deep_gemm
                --speculative-moe-a2a-backend deepep
                --speculative-moe-runner-backend triton
            )
            ;;
        megamoe)
            DEFAULT_ARGS+=(--moe-a2a-backend megamoe)
            ;;
        humming)
            append_deepep_args
            DEFAULT_ARGS+=(
                --moe-runner-backend humming
                --deepep-dispatcher-output-dtype fp8
                )
            ;;
    esac
    [[ -z "${MAX_PREFILL_TOKENS:-}" ]] || DEFAULT_ARGS+=(--max-prefill-tokens "$MAX_PREFILL_TOKENS")
    [[ "$pd_open" != 1 ]] || append_pd_args prefill
}

# D side: DP + EP decode with DP attention and a separate LM head.
append_decode_args() {
    DEFAULT_ARGS+=(
        --dp "$dp_size"
        --enable-dp-attention
        --enable-dp-lm-head
        --ep "$tp_size"
    )
    case "$moe_mode" in
        none|triton)
            DEFAULT_ARGS+=(
                --moe-runner-backend triton
                --speculative-moe-a2a-backend none
                --speculative-moe-runner-backend triton
            )
            ;;
        aiter)
            DEFAULT_ARGS+=(
                --moe-runner-backend aiter
                --speculative-moe-a2a-backend none
                --speculative-moe-runner-backend triton
            )
            ;;            
        deepep)
            append_deepep_args
            DEFAULT_ARGS+=(
                --deepep-mode auto
                --moe-runner-backend deep_gemm
                --speculative-moe-a2a-backend none
                --speculative-moe-runner-backend triton
            )
            ;;
        megamoe)
            DEFAULT_ARGS+=(--moe-a2a-backend megamoe)
            ;;
        humming)
            DEFAULT_ARGS+=(
                --moe-runner-backend humming
                --deepep-dispatcher-output-dtype fp8
                )
            ;;            
    esac
    [[ "$pd_open" != 1 ]] || append_pd_args decode
}

print_command() {
    local index=0 arg next
    printf '%s\n' "sglang serve \\"
    while [[ $index -lt ${#FINAL_ARGS[@]} ]]; do
        arg="${FINAL_ARGS[$index]}"
        if [[ "$arg" == --* ]]; then
            next=$((index + 1))
            if [[ $next -lt ${#FINAL_ARGS[@]} && "${FINAL_ARGS[$next]}" != --* ]]; then
                printf ' %s %s \\\n' "$arg" "${FINAL_ARGS[$next]}"
                index=$((index + 2))
            else
                printf ' %s \\\n' "$arg"
                index=$((index + 1))
            fi
        else
            printf ' %s \\\n' "$arg"
            index=$((index + 1))
        fi
    done
}

print_launch_profile() {
    echo "---- Launch Profile ------"
    echo "host=$the_host ip=$ip port=$port model=$model_path"
    echo "HCU_NUM=$tp_size PD_MODE=$pd_mode PD_OPEN=$pd_open MOE_MODE=$moe_mode PC_ENABLE=$pc_enable"
    echo "DSPARK_BLOCK_SIZE=$dspark_block_size MEM_FRACTION_STATIC=$mem_fraction_static MAX_RUNNING_REQUESTS=$max_running_requests"
    echo "NNODES=$nnodes NODE_RANK=$node_rank DIST_INIT_ADDR=${dist_init_addr:-auto}"
}

if [[ "${1:-}" == -h || "${1:-}" == --help || "${HELP:-0}" == 1 ]]; then
    usage
    exit 0
fi

port=${1:-30000}
model_path=${2:-$DEFAULT_MODEL_PATH}
rank_id=${3:-0}  # Default multi-node rank; NODE_RANK can override it.
host_arg=${4:-}
ip=$(resolve_ip "$host_arg") || exit $?
the_host=$(hostname)
net_ifname=$(resolve_network_interface "$the_host")

if [[ "$the_host" == nmz26 ]]; then
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libibverbs:${LD_LIBRARY_PATH}
fi

tp_size=${HCU_NUM:-8}
dp_size=$tp_size
nnodes=${NNODES:-2}
node_rank=${NODE_RANK:-$rank_id}
pd_mode=${PD_MODE:-none}
pd_open=${PD_OPEN:-0}
moe_mode=${MOE_MODE:-none}
if [[ "$pd_mode" != none && -z "${MOE_MODE+x}" ]]; then
    moe_mode=deepep
fi
pc_enable=${PC_ENABLE:-0}
dspark_block_size=${DSPARK_BLOCK_SIZE:-5}
dspark_draft_model_path=${DSPARK_DRAFT_MODEL_PATH:-$model_path}
mem_fraction_static=${MEM_FRACTION_STATIC:-0.8}
max_running_requests=${MAX_RUNNING_REQUESTS:-32}
pd_bootstrap_port=${SGLANG_DISAGGREGATION_BOOTSTRAP_PORT:-8998}
pd_prefill_dist_port=${PD_PREFILL_DIST_PORT:-6144}
pd_decode_dist_port=${PD_DECODE_DIST_PORT:-6244}
multi_node_dist_port=${DIST_INIT_PORT:-6245}
local_decode_dist_port=${DECODE_DIST_INIT_PORT:-$((port + 733))}

[[ "$tp_size" =~ ^[0-9]+$ ]] && (( tp_size > 0 )) || die "HCU_NUM must be a positive integer (got: $tp_size)"
[[ "$dspark_block_size" =~ ^[0-9]+$ ]] && (( dspark_block_size > 0 )) || die "DSPARK_BLOCK_SIZE must be a positive integer (got: $dspark_block_size)"
[[ "$max_running_requests" =~ ^[0-9]+$ ]] && (( max_running_requests > 0 )) || die "MAX_RUNNING_REQUESTS must be a positive integer (got: $max_running_requests)"
case "$pd_mode" in none|prefill|decode) ;; *) die "Invalid PD_MODE=$pd_mode (expected: none|prefill|decode)" ;; esac
case "$moe_mode" in none|triton|aiter|deepep|megamoe|humming) ;; *) die "Invalid MOE_MODE=$moe_mode (expected: none|triton|aiter|deepep|megamoe|humming)" ;; esac
case "$pc_enable" in 0|1) ;; *) die "PC_ENABLE must be 0 or 1" ;; esac
case "$pd_open" in 0|1) ;; *) die "PD_OPEN must be 0 or 1" ;; esac
if [[ "$pd_open" == 1 && "$pd_mode" == none ]]; then
    die "PD_OPEN=1 requires PD_MODE=prefill or PD_MODE=decode"
fi
[[ "$moe_mode" != deepep ]] || require_file "$DEEPEP_CONFIG" "DEEPEP_CONFIG does not exist"

is_multi_node=0
if (( tp_size > 8 )); then
    is_multi_node=1
    [[ "$nnodes" =~ ^[0-9]+$ ]] && (( nnodes >= 2 )) || die "NNODES must be an integer >= 2 (got: $nnodes)"
    [[ "$node_rank" =~ ^[0-9]+$ ]] && (( node_rank < nnodes )) || die "NODE_RANK/RANK_ID must be in [0, $((nnodes - 1))] (got: $node_rank)"
fi

dist_init_addr=$(resolve_dist_init_addr) || exit $?

env_vars=(
    "NCCL_SOCKET_IFNAME=$net_ifname"
    "GLOO_SOCKET_IFNAME=$net_ifname"
    # HCU compatibility switches needed by the current DeepSeek V4 path.
    "SGLANG_TORCH_PROFILER_DIR=/home/proj_dpsk-v4/profile"
    "SGLANG_OPT_SWIGLU_CLAMP_FUSION=false"
    "SGLANG_USE_AITER_AG=0"
    "SGLANG_USE_LIGHTOP=1"
    "SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1"
    "SGLANG_USE_DEEPGEMM_MOE=1"
    "SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1"
    "SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1"
    "SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1"
    "SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1"
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256"
    # opt args
    "SGLANG_USE_LIGHTOP_PAGED_MQA_LOGITS_FP4=1" # default as 1, need to confirm
)

# Cross-node NCCL fabric selection. Required by ANY multi-node run, not just
# PD: this host exposes mlx5_0..mlx5_9, and with no explicit list RCCL can
# select mlx5_0/mlx5_1, which are not on the inter-node fabric. The prefill/
# decode topologies need it even standalone, since their EP paths still talk
# over ROCSHMEM.
if (( is_multi_node )) || [[ "$pd_mode" != none ]]; then
    env_vars+=(
        "NCCL_IB_HCA=$IB_DEVICES"
        "ROCSHMEM_MAX_NUM_CONTEXTS=48"
    )
fi
# Disaggregation transport only. A PD_OPEN=0 prefill/decode run has no peer
# to transfer to, so the mooncake/UCX settings would be dead configuration.
if [[ "$pd_open" == 1 ]]; then
    env_vars+=(
        "MC_ENABLE_DEST_DEVICE_AFFINITY=1"
        "UCX_NET_DEVICES=$IB_DEVICES"
        "MC_ALLOWED_IBV_DEVICES=$IB_DEVICES"
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200"
        "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1"
    )
fi
if [[ "$pc_enable" == 1 && "$pd_open" == 1 && "$pd_mode" == decode ]]; then
    # DSV4 uses compressed KV plus SWA. Decode-side prefix reuse therefore
    # requires the unified radix tree and the guarded DSV4 implementation.
    env_vars+=(
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE=1"
        "SGLANG_EXPERIMENTAL_DSV4_DECODE_RADIX_CACHE=1"
    )
fi
case "${moe_mode:-}" in
    triton|none)
        env_vars+=("SGLANG_ROCM_USE_AITER_MOE=0")
        ;;
    aiter)
        env_vars+=("SGLANG_USE_FP8_W8A8_MOE=0" "SGLANG_ROCM_USE_AITER_MOE=0")
        ;;
    deepep)
        env_vars+=("SGLANG_USE_FP8_W8A8_MOE=1")
        ;;
    megamoe)
        env_vars+=(
            "SGLANG_HCU_MEGA_MOE_RUNTIME=megamoe"
            "SGLANG_DSV4_CHANNEL_FP8_SCALE=1"
        )
        ;;
    humming)
        env_vars+=(
            'SGLANG_HUMMING_INPUT_QUANT_CONFIG={"dtype": "float8e4m3", "group_size": 128}'
        )
        ;;
esac

if [[ "$pd_mode" == decode ]]; then
    env_vars+=(
        "ROCBLAS_TENSILE_LIBPATH=/home/proj_dpsk-v4/configs/dpsk-v41-dp8ep8-lib"
    )
fi

echo "---- Current Env Variables Setup ------"
echo "export PYTHONPATH=$PYTHONPATH"
[[ "$the_host" != nmz26 ]] || echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
for kv in "${env_vars[@]}"; do
    export "$kv"
    echo "export $kv"
done

DEFAULT_ARGS=(
    --model-path "$model_path"
    --tp-size "$tp_size"
    --host 0.0.0.0
    --port "$port"
    --trust-remote-code
    --dist-timeout 10000
    --watchdog-timeout 3600
    --cuda-graph-backend-prefill disabled
    # --cuda-graph-backend-decode disabled
    --skip-server-warmup
    --mem-fraction-static "$mem_fraction_static"
    --max-running-requests "$max_running_requests"
    --cuda-graph-max-bs-decode 32
    --speculative-algorithm DSPARK
    --speculative-dspark-block-size "$dspark_block_size"
    --reasoning-parser auto
    --tool-call-parser auto
    # opt args
    # --enable-dsa-cache-layer-split # enable it only prefill CP + DSA 模型 + PD 的 P 端 + KV 显存确实是瓶颈
    # --enable-cp-decode-attn-tp # work only on IFB mode
)
append_distributed_init_args

# The original standalone profile runs the MoE through the triton runner;
# the P/D paths select explicit target and draft MoE backends per MOE_MODE.
if [[ "$pd_mode" == none ]]; then
    DEFAULT_ARGS+=(--moe-runner-backend $moe_mode)
    [[ "$moe_mode" != aiter ]] || DEFAULT_ARGS+=(--speculative-moe-runner-backend triton)
fi

case "$pd_mode" in
    prefill) append_prefill_args ;;
    decode) append_decode_args ;;
esac

[[ "$pc_enable" != 0 ]] || DEFAULT_ARGS+=(--disable-radix-cache)
if [[ "$pc_enable" == 1 && "$pd_open" == 1 && "$pd_mode" == decode ]]; then
    DEFAULT_ARGS+=(--disaggregation-decode-enable-radix-cache)
fi

FINAL_ARGS=("${DEFAULT_ARGS[@]}")
print_launch_profile
echo "---- Current Running Cmd ------"
print_command

postfix="${the_host}_${pd_mode}"
# A standalone prefill/decode service is not the same job as the PD half on
# the same host, so keep their logs apart. PD_OPEN=1 keeps the historical
# names, which existing launcher scripts and collected logs already use.
[[ "$pd_mode" == none || "$pd_open" == 1 ]] || postfix="${postfix}_nodisagg"
printf '%s\n' " 2>&1 | tee running_dpsk-v41_${postfix}.log"
printf '%s\n' "--------------------------------"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "DRY_RUN=1: command validated; service was not started."
    exit 0
fi

sglang serve "${FINAL_ARGS[@]}" 2>&1 | tee "running_dpsk-v41_${postfix}.log"
