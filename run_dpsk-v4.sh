#!/bin/bash

# Prefill workers register to the local bootstrap server with Python requests.
# Proxy variables inherited from the terminal would route 0.0.0.0:8998 through
# localhost:18086 and leave the bootstrap server with zero registered workers.
# source /home/unset_proxy.sh

set -o pipefail

readonly DEEPEP_CONFIG=/home/proj_dpsk-v4/configs/deepep_IntraConfig.json
readonly IB_DEVICES=${IB_DEVICES:-mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9}
readonly DEFAULT_MODEL_PATH=/parastor/home/public_user/wanglong/DeepSeek-V4-Flash-FP8-Channel

die() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  bash run_dpsk-v4.sh [PORT] [MODEL_PATH] [RANK_ID] [HOST]

Positional arguments:
  PORT        Service port. Default: 30000.
  MODEL_PATH  Model checkpoint. Default: DeepSeek-V4-Flash-FP8-Channel.
  RANK_ID     Node rank for multi-node launch; compatibility placeholder on one node.
  HOST        Optional distributed-init host alias. Supported: node18, node20,
              node22, node26, sglang2. Default: first local IP address.

Core selectors:
  HCU_NUM=8                 Number of HCUs / TP size (default: 8).
  PD_MODE=none|prefill|decode
                            Select TP, prefill CP, or decode DP topology.
  PD_OPEN=0|1               Add PD disaggregation transport arguments.
  PC_ENABLE=0|1             Enable radix cache. In PD decode mode this also
                            enables the experimental DSV4 decode radix path.
  MOE_MODE=deepep|megamoe   Target MoE runtime selection (default: deepep).
  MTP_MODE=none|mtp|dspark  Speculative mode (default: none).
  IS_INT8=0|1               Enable SlimQuant INT8 loading.
  IS_FP8=0|1                Enable FP8 HCU optimization flags (default: 1).
  WEIGHT_LOAD_THREADS=N     Safetensors loader threads per local rank (default: 64).
  WEIGHT_LOAD_MULTITHREAD=0|1
                            Enable the multithreaded safetensors iterator (default: 1).
  WEIGHT_LOADER_PREFETCH=0|1
                            Prefetch disjoint checkpoint shards into each node's page cache.
  WEIGHT_LOADER_PREFETCH_THREADS=N
                            Prefetch threads per local rank (default: 4).
  DSPARK_MOE_MODE=none|deepep
                            DSpark MoE runtime selection (default: none).
  DSPARK_VARIANT=static|compact-sps|compact-sps-sts
                            DSpark verify variant (default: static).
  PP_SIZE=N                   Prefill pipeline-parallel size. Standard prefill
                            defaults to 2 for the PP2+CP8 two-node profile;
                            set PP_SIZE=1 to keep the single-node CP profile.
  NNODES=N                    Multi-node count when HCU_NUM > 8 or prefill PP > 1
                            (default: 2).
  NODE_RANK=N                 Overrides positional RANK_ID.
  DIST_INIT_PORT=N            Generic multi-node init port (default: 6145).
  PD_PREFILL_DIST_PORT=N      PD P-side init port (default: 6144).
  PD_DECODE_DIST_PORT=N       PD D-side init port (default: 6244).
  DECODE_DIST_INIT_PORT=N     Single-node decode init port (default: PORT + 733).

Distributed init-address precedence (exactly one --dist-init-addr is emitted):
  PD P side     HOST:PD_PREFILL_DIST_PORT
  PD D side     HOST:PD_DECODE_DIST_PORT
  Other multi-node runs  HOST:DIST_INIT_PORT
  Single-node decode     127.0.0.1:DECODE_DIST_INIT_PORT
  Single-node TP/prefill without PD does not set an explicit init address.

Parallel and test combinations:
  1. Pure TP
     PD_MODE=none, MTP_MODE=none
  2. PP2 + CP8 + EP prefill (two nodes when HCU_NUM=8)
     PD_MODE=prefill, MTP_MODE=none, MOE_MODE=deepep
  3. PP2 + CP8 + MegaMoE prefill (two nodes when HCU_NUM=8)
     PD_MODE=prefill, MTP_MODE=none, MOE_MODE=megamoe
  4. DP + EP + MTP decode
     PD_MODE=decode, MTP_MODE=mtp, MOE_MODE=deepep
  5. DP + MegaMoE + MTP decode
     PD_MODE=decode, MTP_MODE=mtp, MOE_MODE=megamoe
  6. DSpark standalone target/draft validation
     MTP_MODE=dspark, PD_MODE=none
  7. DSpark PD prefill: CP + EP target on the P side
     PD_OPEN=1, PD_MODE=prefill, MTP_MODE=dspark
  8. DSpark PD decode: DP + EP target on the D side
     PD_OPEN=1, PD_MODE=decode, MTP_MODE=dspark

Prefill PP note:
  Standard prefill defaults to PP2+CP8 and therefore requires two nodes with
  HCU_NUM=8. Set PP_SIZE=1 to run the single-node CP profile. PP2 prefill is
  supported only with MTP_MODE=none; speculative/DSpark variants stay on PP1.

DSpark variants (DSPARK_VARIANT, default: static):
  static              Static verify baseline.
  compact-sps         Compact verify with an SPS table.
  compact-sps-sts     Compact verify with SPS and STS tables.
  compact-nosps       Compact verify without SPS.
  compact-align       Compact SPS, aligned to graph tiers.
  cap-accept-sps      Cap-accept verify with SPS.
  g3 | g8             Override DSpark block size to 3 or 8.
  sps-record          Record SPS calibration data.

Common DSpark overrides:
  DSPARK_SPS_TABLE_PATH=/path/to/sps.json
  DSPARK_CONFIDENCE_STS_PATH=/path/to/sts.json
  DSPARK_PD_DRAFT_MOE_MODE=none|deepep  (PD decode draft, default: none)
  DSPARK_MEM_FRACTION_STATIC=FLOAT       (default: 0.957; PD decode + draft DeepEP: 0.90)
  DSPARK_BLOCK_SIZE=N
  DSPARK_MAX_RUNNING_REQUESTS=N
  DSPARK_CONTEXT_LENGTH=N
  DSPARK_ENABLE_METRICS=0|1

Examples:
  # Pure TP
  bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel

  # Single-node CP + EP prefill
  PD_MODE=prefill PP_SIZE=1 \
    bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel

  # PP2 + CP8 prefill across two nodes (run once per node with rank 0/1)
  PD_MODE=prefill HCU_NUM=8 NNODES=2 NODE_RANK=0 \
    bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel 0 node26
  PD_MODE=prefill HCU_NUM=8 NNODES=2 NODE_RANK=1 \
    bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel 1 node26

  # Single-node CP + MegaMoE prefill
  PD_MODE=prefill PP_SIZE=1 MOE_MODE=megamoe \
    bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel

  # DP + EP + MTP decode
  PD_MODE=decode MTP_MODE=mtp bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel

  # DP + MegaMoE + MTP decode
  PD_MODE=decode MTP_MODE=mtp MOE_MODE=megamoe \
    bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-FP8-Channel

  # Pure DSpark standalone target/draft validation
  MTP_MODE=dspark bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel

  # DeepEP DSpark standalone target/draft validation
  MTP_MODE=dspark DSPARK_MOE_MODE=deepep \
    bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel

  # DSpark compact SPS + STS standalone validation
  MTP_MODE=dspark DSPARK_VARIANT=compact-sps-sts bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel

  # Record SPS calibration data
  MTP_MODE=dspark DSPARK_VARIANT=sps-record \
    DSPARK_STS_COLLECT_PATH=/tmp/dspark_sts.json \
    bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel

  # DSpark PD P side, HCU 0-3
  HIP_VISIBLE_DEVICES=0,1,2,3 HCU_NUM=4 PD_OPEN=1 PD_MODE=prefill \
    MTP_MODE=dspark DSPARK_MOE_MODE=deepep \
    bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel

  # DSpark PD D side, HCU 4-7
  PC_ENABLE=1 HIP_VISIBLE_DEVICES=4,5,6,7 HCU_NUM=4 PD_OPEN=1 PD_MODE=decode \
    MTP_MODE=dspark DSPARK_MOE_MODE=deepep \
    bash run_dpsk-v4.sh 10016 /module/DeepSeek-V4-Flash-0731-FP8-Channel

  # INT8 pure TP
  IS_INT8=1 IS_FP8=0 bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-Channel-INT8-w8a8

  # Generic 2-node launch: use the same master HOST on both nodes.
  HCU_NUM=16 NNODES=2 bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel 0 node26
  HCU_NUM=16 NNODES=2 bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel 1 node26

  # Two-node DSpark PD: P uses 6144, D uses 6244 by default.
  # Repeat each command with final RANK_ID=1 on the worker node.
  HCU_NUM=16 NNODES=2 PD_OPEN=1 PD_MODE=prefill MTP_MODE=dspark \
    bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel 0 node26
  HCU_NUM=16 NNODES=2 PD_OPEN=1 PD_MODE=decode MTP_MODE=dspark \
    bash run_dpsk-v4.sh 10016 /module/DeepSeek-V4-Flash-0731-FP8-Channel 0 node26

  # Inspect the generated command without starting a service
  DRY_RUN=1 MTP_MODE=dspark DSPARK_VARIANT=compact-sps-sts \
    bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel
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
        node110) echo 12.12.12.110 ;;
        sglang2) echo 10.16.1.33 ;;
        *) die "Invalid HOST=$host_arg (expected: node18|node20|node22|node26|sglang2)" ;;
    esac
}

resolve_network_interface() {
    case "$1" in
        nmz26|nmz20|nmz22|nmz18|nmz15) echo ens66f1np1 ;;
        nmz104|nmz110) echo ens65f0np0;;
        sglang5) echo eth0 ;;
        sglang8) echo enp113s0f0np0 ;;
        sglang6) echo eth10 ;;
        *) echo ens19f0 ;;
    esac
}

append_deepep_args() {
    DEFAULT_ARGS+=(
        --moe-a2a-backend deepep
        --deepep-mode "$deepep_mode"
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
    # Keep this as the sole source of --dist-init-addr. PD P/D need stable,
    # distinct ports so both services can coexist on one host; a generic
    # multi-node group uses its own port; single-node decode keeps the
    # historical port-derived local address.
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

require_file() {
    [[ -f "$1" ]] || die "$2: $1"
}

append_dspark_variant_args() {
    case "$dspark_variant" in
        compact-sps|compact-sps-sts|compact-align|cap-accept-sps)
            require_file "$dspark_sps_table_path" \
                "DSPARK_VARIANT=$dspark_variant requires an SPS table"
            DEFAULT_ARGS+=(--speculative-dspark-sps-table-path "$dspark_sps_table_path")
            ;;
    esac

    case "$dspark_variant" in
        compact-align)
            DEFAULT_ARGS+=(--speculative-dspark-align-verify-tokens-to-graph-tier)
            ;;
        g3)
            DEFAULT_ARGS+=(--speculative-dspark-block-size 3)
            ;;
        g8)
            DEFAULT_ARGS+=(--speculative-dspark-block-size 8)
            ;;
    esac

    if [[ -n "${DSPARK_BLOCK_SIZE:-}" ]]; then
        [[ "$dspark_variant" != g3 && "$dspark_variant" != g8 ]] || \
            die "DSPARK_BLOCK_SIZE cannot be combined with DSPARK_VARIANT=$dspark_variant"
        DEFAULT_ARGS+=(--speculative-dspark-block-size "$DSPARK_BLOCK_SIZE")
    fi

    if [[ -n "$dspark_confidence_sts_path" ]]; then
        require_file "$dspark_confidence_sts_path" \
            "DSPARK_CONFIDENCE_STS_PATH does not exist"
        DEFAULT_ARGS+=(--speculative-dspark-confidence-sts-path "$dspark_confidence_sts_path")
    fi

    if [[ "$dspark_variant" == compact-sps-sts ]]; then
        echo "DSpark SPS+STS test arm: SPS=$dspark_sps_table_path STS=$dspark_confidence_sts_path"
    fi
}

append_dspark_args() {
    DEFAULT_ARGS+=(
        --speculative-algorithm DSPARK
        --speculative-draft-model-path "$dspark_draft_model_path"
        --speculative-num-steps 1
        --speculative-eagle-topk 1
        --max-running-requests "${DSPARK_MAX_RUNNING_REQUESTS:-32}"
        --context-length "${DSPARK_CONTEXT_LENGTH:-32768}"
    )

    # PD prefill uses target CP+DeepEP/DeepGEMM, but the draft only injects
    # target hidden KV and must not initialize the decode-side DP/LM-head path.
    if [[ "$pd_open" == 1 && "$pd_mode" == prefill ]]; then
        DEFAULT_ARGS+=(
            --disable-cuda-graph
            --max-total-tokens "${DSPARK_PD_PREFILL_MAX_TOTAL_TOKENS:-131072}"
            --speculative-moe-a2a-backend none
        )
    else
        if [[ "$pd_mode" != prefill ]]; then
            DEFAULT_ARGS+=(
                --dp "$dp_size"
                --enable-dp-attention
                --enable-dp-lm-head
            )
        fi
        DEFAULT_ARGS+=(
            --ep "$tp_size"
        )
        if [[ "$dspark_moe_mode" == deepep ]]; then
            append_deepep_args
            if [[ "$pd_open" == 1 && "$pd_mode" == decode && "$dspark_pd_draft_moe_mode" == none ]]; then
                # Match H20: keep one DeepEP/DeepGEMM target runtime and
                # execute the DSpark draft through the standalone path.
                DEFAULT_ARGS+=(
                    --speculative-moe-a2a-backend none
                    --speculative-moe-runner-backend triton
                )
            else
                DEFAULT_ARGS+=(
                    --speculative-moe-a2a-backend deepep
                    --speculative-moe-runner-backend deep_gemm
                )
            fi
            DEFAULT_ARGS+=(--moe-runner-backend deep_gemm)
        else
            DEFAULT_ARGS+=(--moe-a2a-backend none)
        fi
    fi

    append_dspark_variant_args

    case "${DSPARK_ENABLE_METRICS:-1}" in
        1|true|TRUE|yes|YES) DEFAULT_ARGS+=(--enable-metrics) ;;
        0|false|FALSE|no|NO) ;;
        *) die "Invalid DSPARK_ENABLE_METRICS=${DSPARK_ENABLE_METRICS} (expected boolean)" ;;
    esac
}

append_mtp_args() {
    DEFAULT_ARGS+=(
        --speculative-algorithm EAGLE
        --speculative-num-steps 3
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4
        --max-running-requests "${DSPARK_MAX_RUNNING_REQUESTS:-32}"
        --context-length "${DSPARK_CONTEXT_LENGTH:-4096}"
    )
}

append_prefill_parallel_args() {
    if [[ "$moe_mode" == megamoe ]]; then
        DEFAULT_ARGS+=(
            --enable-prefill-cp
            --cp-strategy interleave
            --moe-a2a-backend megamoe
        )
    else
        DEFAULT_ARGS+=(
            --enable-prefill-cp
            --cp-strategy interleave
            --dp 1
            --attn-cp-size "$tp_size"
            --enable-dp-attention
        )
        append_deepep_args
        [[ "$mtp_mode" != dspark ]] || DEFAULT_ARGS+=(--moe-runner-backend deep_gemm)
    fi

    if (( pp_size > 1 )); then
        DEFAULT_ARGS+=(
            --pp-size "$pp_size"
            --disable-overlap-schedule
        )
    fi

    [[ "$pd_open" != 1 ]] || append_pd_args prefill
}

append_decode_parallel_args() {
    if [[ "$mtp_mode" != dspark ]]; then
        if [[ "$moe_mode" == megamoe ]]; then
            DEFAULT_ARGS+=(--moe-a2a-backend megamoe)
        else
            append_deepep_args
        fi

        DEFAULT_ARGS+=(
            --dp "$dp_size"
            --enable-dp-attention
        )

        # MTP_MODE=mtp already supplied the EAGLE configuration above.
        [[ "$mtp_mode" != none ]] || DEFAULT_ARGS+=(
            --speculative-algo EAGLE
            --speculative-num-draft-tokens 4
            --speculative-eagle-topk 1
            --speculative-num-steps 3
        )
    fi

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
    echo "HCU_NUM=$tp_size PP_SIZE=$pp_size PD_MODE=$pd_mode PD_OPEN=$pd_open MOE_MODE=$moe_mode"
    echo "MTP_MODE=$mtp_mode DSPARK_VARIANT=$dspark_variant DSPARK_MOE_MODE=$dspark_moe_mode"
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

pc_enable=${PC_ENABLE:-0}
is_int4=${IS_INT4:-0}
is_int8=${IS_INT8:-0}
is_fp8=${IS_FP8:-1}
weight_load_threads=${WEIGHT_LOAD_THREADS:-64}
weight_load_multithread=${WEIGHT_LOAD_MULTITHREAD:-1}
weight_loader_prefetch=${WEIGHT_LOADER_PREFETCH:-0}
weight_loader_prefetch_threads=${WEIGHT_LOADER_PREFETCH_THREADS:-4}
deepep_mode=${DEEPEP_MODE:-auto}
pd_mode=${PD_MODE:-none}
pd_open=${PD_OPEN:-0}
moe_mode=${MOE_MODE:-deepep}
mtp_mode=${MTP_MODE:-none}
dspark_moe_mode=${DSPARK_MOE_MODE:-none}
dspark_variant=${DSPARK_VARIANT:-static}
dspark_sps_table_path=${DSPARK_SPS_TABLE_PATH:-/home/proj_sglang_open/dspark/sps_table_hcu.json}
dspark_confidence_sts_path=${DSPARK_CONFIDENCE_STS_PATH:-}
dspark_default_sts_path=/home/proj_sglang_open/dspark/sts_hcu/dspark_sts.json
dspark_draft_model_path=${DSPARK_DRAFT_MODEL_PATH:-$model_path}
dspark_pd_draft_moe_mode=${DSPARK_PD_DRAFT_MOE_MODE:-none}
pd_bootstrap_port=${SGLANG_DISAGGREGATION_BOOTSTRAP_PORT:-8998}
pd_prefill_dist_port=${PD_PREFILL_DIST_PORT:-6144}
pd_decode_dist_port=${PD_DECODE_DIST_PORT:-6244}
multi_node_dist_port=${DIST_INIT_PORT:-6245}
nnodes=${NNODES:-2}
node_rank=${NODE_RANK:-$rank_id}
local_decode_dist_port=${DECODE_DIST_INIT_PORT:-$((port + 733))}
tp_size=${HCU_NUM:-8}
dp_size=$tp_size
pp_size=${PP_SIZE:-1}
[[ "$tp_size" =~ ^[0-9]+$ ]] || die "HCU_NUM must be a positive integer (got: $tp_size)"
(( tp_size > 0 )) || die "HCU_NUM must be a positive integer (got: $tp_size)"
[[ "$pp_size" =~ ^[0-9]+$ ]] || die "PP_SIZE must be a positive integer (got: $pp_size)"
(( pp_size > 0 )) || die "PP_SIZE must be a positive integer (got: $pp_size)"

# The standard (non-speculative) prefill profile is PP2+CP8. Keep the
# speculative/DSpark prefill arms at PP1 because current SGLang validation
# rejects pipeline parallelism together with speculative decoding.
if [[ "$pd_mode" == prefill && "$mtp_mode" == none && -z "${PP_SIZE+x}" ]]; then
    pp_size=2
fi
[[ "$pd_mode" == prefill || "$pp_size" == 1 ]] || \
    die "PP_SIZE>1 is only supported for PD_MODE=prefill"
[[ "$pp_size" == 1 || "$mtp_mode" == none ]] || \
    die "PP_SIZE>1 requires MTP_MODE=none (PP is incompatible with speculative decoding)"

is_multi_node=0
if (( tp_size > 8 )) || [[ "$pd_mode" == prefill && "$pp_size" -gt 1 ]]; then
    is_multi_node=1
    [[ "$nnodes" =~ ^[0-9]+$ ]] && (( nnodes >= 2 )) || die "NNODES must be an integer >= 2 (got: $nnodes)"
    [[ "$node_rank" =~ ^[0-9]+$ ]] && (( node_rank < nnodes )) || die "NODE_RANK/RANK_ID must be in [0, $((nnodes - 1))] (got: $node_rank)"
fi

case "$pd_mode" in none|prefill|decode) ;; *) die "Invalid PD_MODE=$pd_mode (expected: none|prefill|decode)" ;; esac
case "$mtp_mode" in none|dspark|mtp) ;; *) die "Invalid MTP_MODE=$mtp_mode (expected: none|dspark|mtp)" ;; esac
case "$weight_load_multithread" in 0|1) ;; *) die "WEIGHT_LOAD_MULTITHREAD must be 0 or 1" ;; esac
case "$weight_loader_prefetch" in 0|1) ;; *) die "WEIGHT_LOADER_PREFETCH must be 0 or 1" ;; esac
[[ "$weight_load_threads" =~ ^[0-9]+$ ]] && (( weight_load_threads > 0 )) || die "WEIGHT_LOAD_THREADS must be a positive integer"
[[ "$weight_loader_prefetch_threads" =~ ^[0-9]+$ ]] && (( weight_loader_prefetch_threads > 0 )) || die "WEIGHT_LOADER_PREFETCH_THREADS must be a positive integer"
case "$dspark_moe_mode" in none|deepep) ;; *) die "Invalid DSPARK_MOE_MODE=$dspark_moe_mode (expected: none|deepep)" ;; esac
case "$dspark_pd_draft_moe_mode" in none|deepep) ;; *) die "Invalid DSPARK_PD_DRAFT_MOE_MODE=$dspark_pd_draft_moe_mode (expected: none|deepep)" ;; esac
[[ "$mtp_mode" == dspark || "$dspark_moe_mode" == none ]] || die "DSPARK_MOE_MODE=$dspark_moe_mode requires MTP_MODE=dspark"

if [[ "$mtp_mode" == dspark ]]; then
    case "$dspark_variant" in
        static|compact-sps|compact-sps-sts|compact-nosps|compact-align|cap-accept-sps|g3|g8|sps-record) ;;
        *) die "Invalid DSPARK_VARIANT=$dspark_variant" ;;
    esac
    # This named arm is the reproducible Phase-3 SPS+STS test configuration.
    # An explicit DSPARK_CONFIDENCE_STS_PATH still takes precedence so that a
    # workload-specific calibration table can be evaluated without script edits.
    if [[ "$dspark_variant" == compact-sps-sts && -z "$dspark_confidence_sts_path" ]]; then
        dspark_confidence_sts_path=$dspark_default_sts_path
    fi
fi

rocshmem_env_vars=(
    "ROCSHMEM_DISABLE_HDP_FLUSH=1"
    "ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288"
    "ROCSHMEM_HEAP_SIZE=3173741824"
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128"
)

# Cross-node NCCL fabric selection. Required by ANY multi-node run, not just
# PD: this host exposes mlx5_0..mlx5_9, and with no explicit list RCCL can
# select mlx5_0/mlx5_1, which are not on the inter-node fabric. The first
# cross-node all-reduce then fails with "remote process exited or there was a
# network error". Verified on nmz20+nmz22: a 16-rank [32,7168] all-reduce
# fails without this list and passes with it, everything else held equal.
nccl_fabric_env_vars=(
    "NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1"
)

pd_env_vars=(
    "MC_ENABLE_DEST_DEVICE_AFFINITY=1"
    "UCX_NET_DEVICES=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1"
    "MC_ALLOWED_IBV_DEVICES=$IB_DEVICES"
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200"
)

env_vars=(
    "NCCL_SOCKET_IFNAME=$net_ifname"
    "GLOO_SOCKET_IFNAME=$net_ifname"
    "NCCL_MIN_NCHANNELS=16"
    "NCCL_MAX_NCHANNELS=16"
    "SGLANG_TORCH_PROFILER_DIR=/home/proj_dpsk-v4/profile"
    "SGLANG_OPT_USE_FUSED_STORE_CACHE=false"  # Can be enabled, but accuracy drops significantly.
    "SGLANG_OPT_USE_FUSED_HASH_TOPK=true"
    "SGLANG_OPT_SWIGLU_CLAMP_FUSION=false"    # CUDA-only fused kernel; keep disabled on HIP.
    "SGLANG_TOPK_TRANSFORM_512_TORCH=false"
    "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true"
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE=0"
    "SGLANG_USE_AITER_AG=0"                   # Use TP all-gather instead of AITER custom all-gather.
    "${rocshmem_env_vars[@]}"
    # MoE and GEMM kernel optimization.
    "SGLANG_ROCM_USE_AITER_MOE=${SGLANG_ROCM_USE_AITER_MOE:-1}"
    "SGLANG_USE_OPT_CAT=1"
    "SGLANG_USE_FUSED_MLA_CAT=1"
    "SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=$is_fp8"
    "SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=$is_fp8"
    "SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1"
    "SGLANG_APPLY_CONFIG_BACKUP=none"
    # mHC and attention optimization.
    "SGLANG_ROCM_USE_AITER_TILELANG_MHC=1"
    "SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1"
    "SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1"
    # Fused kernel optimization.
    "SGLANG_USE_LIGHTOP=1"
    "SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1"
    "SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1"
    "SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1"
    "SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1"
    "SGLANG_USE_LIGHTOP_EP_SCATTER=1"
    "SGLANG_USE_LIGHTOP_EP_GATHER=1"
    "SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1"
)

# PD_MODE prefill/decode historically enables the DeepGEMM environment even
# without PD_OPEN; retain that behavior while avoiding duplicate exports.
if [[ "$pd_open" == 1 || "$pd_mode" != none ]] || (( is_multi_node )); then
    env_vars+=("${nccl_fabric_env_vars[@]}")
fi
if [[ "$pd_open" == 1 || "$pd_mode" != none ]]; then
    env_vars+=("${pd_env_vars[@]}")
fi
if [[ "$pd_mode" != none ]]; then
    env_vars+=(
        "SGLANG_USE_FP8_W8A8_MOE=$is_fp8"
        "SGLANG_USE_DEEPGEMM_MOE=1"
    )
fi
if [[ "$mtp_mode" != none ]]; then
    # DSpark FP8-channel checkpoints have BF16 wo_a; FP8 GEMM is invalid here.
    env_vars+=("SGLANG_OPT_FP8_WO_A_GEMM=0")
fi
if [[ "$mtp_mode" == dspark ]]; then
    case "$dspark_variant" in
        compact-sps|compact-sps-sts|compact-nosps|compact-align|sps-record) dspark_ragged_verify_mode=compact ;;
        cap-accept-sps) dspark_ragged_verify_mode=cap-accept ;;
        *) dspark_ragged_verify_mode=static ;;
    esac
    env_vars+=(
        "SGLANG_RAGGED_VERIFY_MODE=$dspark_ragged_verify_mode"
        "SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS=${DSPARK_CONFIDENCE_RELAY_LAG_STEPS:-2}"
        "SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=${DSPARK_OPT_MARKOV_W2_TP_SHARD:-1}"
        "SGLANG_DSPARK_ENABLE_MULTI_STREAM=${DSPARK_ENABLE_MULTI_STREAM:-1}"
        "SGLANG_DSPARK_FAST_KERNEL=${DSPARK_FAST_KERNEL:-1}"
        "SGLANG_DSPARK_FAST_SAMPLING=${DSPARK_FAST_SAMPLING:-1}"
    )
    if [[ "$dspark_moe_mode" == deepep ]]; then
        env_vars+=(
            "SGLANG_USE_FP8_W8A8_MOE=$is_fp8"
            "SGLANG_USE_DEEPGEMM_MOE=1"
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${DSPARK_DEEPEP_MAX_DISPATCH_TOKENS:-256}"
        )
    fi
    if [[ "$dspark_variant" == sps-record ]]; then
        env_vars+=(
            "SGLANG_DSPARK_ENABLE_SPS_RECORD=1"
            "SGLANG_SIMULATE_ACC_LEN=${DSPARK_SIMULATE_ACC_LEN:-1.0}"
        )
    fi
    [[ -z "${DSPARK_STS_COLLECT_PATH:-}" ]] || env_vars+=("SGLANG_DSPARK_STS_COLLECT_PATH=$DSPARK_STS_COLLECT_PATH")
fi
if [[ "$pc_enable" != 0 && "$pd_open" == 1 && "$pd_mode" == decode ]]; then
    # DSV4 uses compressed KV plus SWA. Decode-side prefix reuse therefore
    # requires the unified radix tree and the guarded DSV4 implementation.
    env_vars+=(
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE=1"
        "SGLANG_EXPERIMENTAL_DSV4_DECODE_RADIX_CACHE=1"
    )
fi
if [[ "$moe_mode" == megamoe ]]; then
    env_vars+=(
        "SGLANG_DCU_MEGA_MOE_RUNTIME=megamoe"
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096"
        "SGLANG_DSV4_CHANNEL_FP8_SCALE=1"
    )
fi
[[ "$is_int4" != 1 ]] || env_vars+=("SGLANG_W4A8_TPMOE_BACKEND=triton")

echo "---- Current Env Variables Setup ------"
for kv in "${env_vars[@]}"; do
    export "$kv"
    echo "export $kv"
done

dist_init_addr=$(resolve_dist_init_addr) || exit $?
cuda_graph_max_bs=32
mem_fraction_static=0.85
if [[ "$mtp_mode" == dspark ]]; then
    cuda_graph_max_bs=${DSPARK_CUDA_GRAPH_MAX_BS:-32}
    mem_fraction_static=${DSPARK_MEM_FRACTION_STATIC:-0.90}
fi

DEFAULT_ARGS=(
    # --load-format fastsafetensors
    --reasoning-parser deepseek-v4
    --tool-call-parser deepseekv4
    --tp-size "$tp_size"
    --dist-timeout 10000
    --watchdog-timeout 3600
    --port "$port"
    --host 0.0.0.0
    --model-path "$model_path"
    --model-loader-extra-config "{\"enable_multithread_load\": \"$([[ "$weight_load_multithread" == 1 ]] && echo true || echo false)\", \"num_threads\": $weight_load_threads}"
    --trust-remote-code
    --chunked-prefill-size 32768
    --disable-flashinfer-autotune
    --skip-server-warmup
    --cuda-graph-max-bs "$cuda_graph_max_bs"
    --mem-fraction-static "$mem_fraction_static"
)
append_distributed_init_args

if [[ "$weight_loader_prefetch" == 1 ]]; then
    DEFAULT_ARGS+=(
        --weight-loader-prefetch-checkpoints
        --weight-loader-prefetch-num-threads "$weight_loader_prefetch_threads"
    )
    if [[ "$weight_load_multithread" == 1 ]]; then
        echo "WARNING: checkpoint prefetch and multithread loading are both enabled; this is intended only for local-NVMe experiments."
    fi
fi

case "$mtp_mode" in
    dspark) append_dspark_args ;;
    mtp) append_mtp_args ;;
esac
case "$pd_mode" in
    prefill) append_prefill_parallel_args ;;
    decode) append_decode_parallel_args ;;
esac

[[ "$is_int8" != 1 ]] || DEFAULT_ARGS+=(--quantization w8a8_int8)
[[ "$is_int4" != 1 ]] || DEFAULT_ARGS+=(--quantization slimquant_marlin)
# [[ "$is_int4" != 1 ]] || DEFAULT_ARGS+=(--moe-runner-backend aiter)

[[ "$pc_enable" != 0 ]] || DEFAULT_ARGS+=(--disable-radix-cache)
if [[ "$pc_enable" != 0 && "$pd_open" == 1 && "$pd_mode" == decode ]]; then
    DEFAULT_ARGS+=(--disaggregation-decode-enable-radix-cache)
fi


FINAL_ARGS=("${DEFAULT_ARGS[@]}")
print_launch_profile
echo "---- Current Running Cmd ------"
print_command

postfix="${the_host}_${pd_mode}"
printf '%s\n' " 2>&1 | tee running_dpsk-v4_${postfix}.log"
printf '%s\n' "--------------------------------"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "DRY_RUN=1: command validated; service was not started."
    exit 0
fi

sglang serve "${FINAL_ARGS[@]}" 2>&1 | tee "running_dpsk-v4_${postfix}.log"
