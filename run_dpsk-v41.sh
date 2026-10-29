#!/bin/bash

# DeepSeek-V4.1 launcher, modelled after run_dpsk-v4.sh:
#   - echoes every environment variable it exports and the final CLI line
#   - PD_MODE=none|prefill|decode picks TP / P-side CP+EP / D-side DP+EP args
#   - PD_OPEN=1 adds the disaggregation transport on top of that topology;
#     PD_OPEN=0 (default) runs the same topology standalone on one node
#   - HCU_NUM>8 switches to a multi-node launch (NNODES / NODE_RANK)
#
# 需要新增 / 修改「指令」或「环境变量」时，只改下面 OPT-BEGIN 那个登记处。
# source /home/unset_proxy.sh  # uncomment if a proxy breaks the PD bootstrap server

set -o pipefail

# ===========================================================================
#  动态优化点登记处
#
#  以后要新增 / 修改「指令」或「环境变量」，只改这一段，下面任何代码都不用动。
#  区块内每一行都是注释，写错也不影响脚本运行。
#
#  格式：
#      #env NAME=VALUE        追加一个环境变量
#      #arg --flag value      追加一段启动指令
#      把行首的 # 改成 ## 即可临时停用该行（保留记录，不生效）
#
#  规则：
#      - 按日期分组，用  # ---- YYYYMMDD ----  这样的注释行隔开，便于看清
#        每个优化点是哪天加进去的
#      - 这里只写字面值，不支持 $变量、不支持条件判断。需要按 PD_MODE /
#        PD_OPEN / MOE_MODE 等条件才生效的开关，请写进下面的 bash 逻辑里
#        （见 env_vars 构建处）
#      - 不要删掉 BEGIN / END 两行标记，脚本靠它们定位这一段
# ===========================================================================
# >>> OPT-BEGIN
#
# ---- Common Optimizations ----
#env SGLANG_TORCH_PROFILER_DIR=/home/proj_dpsk-v4/profile
#env SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
#env SGLANG_USE_AITER_AG=0
#env SGLANG_USE_LIGHTOP=1
#env SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
#env SGLANG_USE_DEEPGEMM_MOE=1
#env SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
#env SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
#env SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
#env SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
#env SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
#env SGLANG_USE_LIGHTOP_PAGED_MQA_LOGITS_FP4=1
#
# ---- 20261009 Optimizations ----
##DEEPEP_ENABLE_LL_LAYERED_OPT=1, ROCSHMEM_DISABLE_HDP_FLUSH=1, --enable-dp-attention-local-control-broadcast only in decode
##arg --enable-cp-cache-layer-split, work only in prefill or ifb but no spec
#env SGLANG_USE_LIGHTOP_DSV41_PAGED_INDEXER=1
#
# ---- 20261010 Optimizations ----
#env ROCBLAS_TENSILE_LIBPATH=/home/proj_dpsk-v4/configs/dpsk-v41-dp8ep8-lib
# 
# ---- 20261013 Optimizations ----
##arg --enable-single-batch-overlap work only in prefill or ifb but no spec
#
# ---- 20261014 Optimizations ----
##--enable-cp-cache-layer-split in dspark, works only in prefill or ifb but no spec
#
# ---- 20261019 Optimizations ----
##env SGLANG_HCU_OPT_DSV41_PAIR_PARTNERS. default as 1
#env NCCL_SIMPLE_CHANNELS=32
##arg --enable-dp-attention-local-control-broadcast, work only in decode
##arg --enable-decoder-swa-bounded-replay, work only in prefill or ifb
#
# ---- 20261020 Optimizations ----
##env SGLANG_HCU_PREFILL_PAGED_SPARSE=1, work only in prefill or ifb, to control the mqa logits, default as 1
##env SGLANG_HCU_PREFILL_SPARSE_FUSION=1, work only in prefill or ifb, to control the mqa logits fusion, default as 1
##arg --enable-single-batch-overlap, both prefill and decode support SBO but has to be disabled in megamoe mode.
#
# ---- 20261027 Optimizations ----
#env HSA_USERPTR_FOR_PAGED_MEM=0
#
##arg --cuda-graph-backend-decode disabled
##arg --enable-cp-decode-attn-tp
##env SGLANG_LIGHTOP_DEQUANTIZE_K_CACHE_PAGED=1, need to lightop support special page_size
#
# <<< OPT-END
#
#  ---- 以下是解析逻辑，不需要修改 ----
OPT_ENV=()
OPT_ARGS=()
while IFS= read -r _opt_line; do
    case "$_opt_line" in
        "#env "*) OPT_ENV+=("${_opt_line#\#env }") ;;
        "#arg "*) OPT_ARGS+=("${_opt_line#\#arg }") ;;
    esac
done < <(sed -n '/^# >>> OPT-BEGIN$/,/^# <<< OPT-END$/p' "${BASH_SOURCE[0]}")

# export PYTHONPATH=/home/proj_dpsk-v4/sglang-das/python:${PYTHONPATH:-}

readonly DEEPEP_CONFIG=/home/proj_dpsk-v4/configs/deepep_IntraConfig.json
# Pick the RDMA NIC by fabric, not by its mlx5_N name (RoCE v2 GID index 3):
#   IB_RAIL=1..8  13.13.<rail>.x  nmz7/15/18/20/21/22/26/28 only (8 rails each)
#   IB_RAIL=12    12.12.12.x      every host; the only NIC of nmz104/105/107/110
# The mlx5_N numbering differs per host (rail 2 is mlx5_3 on most, mlx5_1 on
# nmz7), and the fabrics are separate subnets: P and D on different ones fail
# the mooncake handshake with "Failed to modify QP to RTR ... Connection timed
# out". Default rail 2, or 12 when this host or the PD peer (PD_PEER) has no
# 13.13 fabric. IB_DEVICES overrides all of this.
ib_device_on_rail() {
    local dev gid prefix
    if [[ "$1" == 12 ]]; then
        prefix=0000:0000:0000:0000:0000:ffff:0c0c:0c
    else
        prefix=0000:0000:0000:0000:0000:ffff:0d0d:$(printf %02x "$1")
    fi
    for dev in /sys/class/infiniband/*; do
        gid=$(cat "$dev/ports/1/gids/3" 2>/dev/null) || continue
        if [[ "$gid" == "$prefix"?? && "$(cat "$dev/ports/1/state" 2>/dev/null)" == *ACTIVE ]]; then
            basename "$dev"
            return
        fi
    done
}
default_ib_rail=2
case "${PD_PEER:-}" in
    *104|*105|*107|*110) default_ib_rail=12 ;;  # nmz104 / node104 / 12.12.12.104 ...
esac
[[ -n "$(ib_device_on_rail 2)" ]] || default_ib_rail=12
readonly IB_RAIL=${IB_RAIL:-$default_ib_rail}
default_ib_devices=$(ib_device_on_rail "$IB_RAIL")
if [[ -z "${IB_DEVICES:-}" && -z "$default_ib_devices" && -n "$(ib_device_on_rail 12)" ]]; then
    # On the zz fabric but the requested rail has no active NIC here.
    echo "ERROR: no active RDMA NIC on IB_RAIL=$IB_RAIL on $(hostname -s)" >&2
    exit 1
fi
# Hosts outside these fabrics (sz S*, sglang*) keep the old name-based default.
readonly IB_DEVICES=${IB_DEVICES:-${default_ib_devices:-mlx5_3}}
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
  SGLANG_HCU_MEGA_MOE_RUNTIME=megamoe|deep_gemm
                            HCU Mega implementation (default: megamoe).
                            deep_gemm requires deepgemm.mega with V4.1 DSpark TopK3.
  MEGAMOE_HCU_BACKEND=normal|auto|ll
                            Standalone defaults to normal; LL is opt-in.
  DG_MEGA_HT_EXPECTED       deep_gemm defaults to 0 (validated HT path).
  DSPARK_MOE_MODE=none|deepep|megamoe
                            D-side draft backend (default: target MOE_MODE);
                            P-side draft remains standalone triton.
  CHUNK_SIZE=N              Prefill chunk (default: 32768; Mega P: 16384).
                            Mega cap: P full chunk, D ceil(chunk/HCU_NUM).
  MOE_MODE=none|triton|deepep|megamoe|aiter|humming
                            MoE runtime for the P/D paths (default: deepep when
                            PD_MODE is set, none otherwise). P/D paths require
                            deepep or megamoe.
  PC_ENABLE=0|1             Radix cache (default: 0). PD_OPEN=1 decode with
                            PC_ENABLE=1 enables the DSV4 decode radix path.
  DSPARK=0|1                Enable DSpark speculation (default: 1). The
                            standalone P profile (PD_OPEN=0) needs DSPARK=0:
                            DSpark + CP only runs on the PD prefill worker.
  DSPARK_BLOCK_SIZE=N       DSpark block size (default: 5).
  DSPARK_DRAFT_MODEL_PATH=P Draft checkpoint (default: MODEL_PATH).
  MEM_FRACTION_STATIC=F     KV cache memory fraction (default: 0.85; Mega P: 0.80).
  MAX_PREFILL_TOKENS=N      Optional prefill token cap on the P side.
  MAX_RUNNING_REQUESTS=N    Scheduler concurrency cap (default: 32).
  IB_RAIL=N                 RDMA fabric: 1..8 = 13.13.N.x, 12 = 12.12.12.x
                            (default 2; 12 on nmz104/105/107/110 or when
                            PD_PEER is one of them). Both PD ends must match.
  PD_PEER=HOST              The other PD end (nmz107, node107, ...); picks
                            IB_RAIL=12 when it is only on 12.12.12.x.
  IB_DEVICES=LIST           PD/distributed RDMA devices; overrides IB_RAIL
  NNODES=N                  Multi-node count (default: 2).
  NODE_RANK=N               Overrides positional RANK_ID.
  DIST_INIT_PORT=N          Generic multi-node init port (default: 6245).
  PD_PREFILL_DIST_PORT=N    PD P-side init port (default: 6144).
  PD_DECODE_DIST_PORT=N     PD D-side init port (default: 6244).
  DECODE_DIST_INIT_PORT=N   Standalone decode init port (default: PORT + 733).
  DRY_RUN=1                 Print the resolved command without starting.

Adding an optimization point:
  Put its environment variables / CLI flags in the OPT-BEGIN block at the top
  of this file, grouped under a  # ---- YYYYMMDD ----  comment. They are then
  appended to the exported environment and to the `sglang serve` line. Nothing
  else in the script needs to change. Switches that must apply only for some
  topologies stay in the bash logic below (env_vars construction).

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
        S1) echo 10.243.1.1 ;;
        S4) echo 10.243.1.4 ;;
        S5) echo 10.243.1.5 ;;
        S7) echo 10.243.1.5 ;;
        S9) echo 10.243.1.9 ;;
        node18) echo 13.13.2.18 ;;
        node20) echo 13.13.2.20 ;;
        node22) echo 13.13.2.22 ;;
        node26) echo 13.13.2.26 ;;
        node104) echo 12.12.12.104 ;;
        node105) echo 12.12.12.105 ;;
        node107) echo 12.12.12.107 ;;
        node110) echo 12.12.12.110 ;;
        sglang2) echo 10.16.1.33 ;;
        *) die "Invalid HOST=$host_arg (expected: node18|node20|node22|node26|node104|node105|node107|node110|sglang2)" ;;
    esac
}

resolve_network_interface() {
    case "$1" in
        nmz26|nmz20|nmz22|nmz21|nmz18|nmz15|nmz28) echo ens66f1np1 ;;
        nmz7) echo ens61f1np1 ;;  # rail 2 (13.13.2.7); its ens66f1np1 is rail 4
        nmz105|nmz107|nmz104|nmz110) echo ens65f0np0 ;;
        S1|S4|S5|S7|S9) echo ens30np0 ;;
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

append_draft_moe_args() {
    # No draft model is loaded when DSPARK is off, so these speculative MoE
    # backend args would be dead configuration.
    [[ "$dspark_enable" == 1 ]] || return 0
    local mode=$dspark_moe_mode
    # [[ "$1" != decode ]] || mode=$dspark_moe_mode
    case "$mode" in
        none)
            DEFAULT_ARGS+=(--speculative-moe-a2a-backend none --speculative-moe-runner-backend triton)
            ;;
        deepep)
            DEFAULT_ARGS+=(--speculative-moe-a2a-backend deepep --speculative-moe-runner-backend deep_gemm)
            ;;
        megamoe)
            DEFAULT_ARGS+=(--speculative-moe-a2a-backend megamoe)
            ;;
    esac
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
        --enable-decoder-swa-bounded-replay
    )
    # Cache LayerSplit persists only the layers owned by each CP rank, and the
    # decode side pulls the rest through PD transfer, so a non-PD worker (which
    # also decodes) cannot read them locally. Keep it on the PD prefill half
    # only: PD_OPEN=0 runs the same topology standalone on a single node.
    if [[ "$pd_open" == 1 ]]; then
        DEFAULT_ARGS+=(--enable-cp-cache-layer-split)
    fi
    case "$moe_mode" in
        none|triton)
            DEFAULT_ARGS+=(
                --moe-runner-backend triton
                --enable-single-batch-overlap
            )
            ;;
        aiter)
            DEFAULT_ARGS+=(
                --moe-runner-backend aiter
                --enable-single-batch-overlap
            )
            ;;
        deepep)
            append_deepep_args
            DEFAULT_ARGS+=(
                --deepep-mode auto
                --moe-runner-backend deep_gemm
                --enable-single-batch-overlap
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
    append_draft_moe_args prefill
    [[ "$pd_open" != 1 ]] || append_pd_args prefill
}

# D side: DP + EP decode with DP attention and a separate LM head.
append_decode_args() {
    DEFAULT_ARGS+=(
        --dp "$dp_size"
        --enable-dp-attention
        --enable-dp-attention-local-control-broadcast
        --enable-dp-lm-head
        --ep "$tp_size"
    )
    case "$moe_mode" in
        none|triton)
            DEFAULT_ARGS+=(
                --moe-runner-backend triton
                --enable-single-batch-overlap
            )
            ;;
        aiter)
            DEFAULT_ARGS+=(
                --moe-runner-backend aiter
                --enable-single-batch-overlap
            )
            ;;
        deepep)
            append_deepep_args
            DEFAULT_ARGS+=(
                --deepep-mode auto
                --moe-runner-backend deep_gemm
                --enable-single-batch-overlap
            )
            ;;
        megamoe)
            DEFAULT_ARGS+=(--moe-a2a-backend megamoe)
            ;;
        humming)
            # NOTE: was `--deepep-dispatcher-output-fp8` here, which is not a
            # real flag (the prefill branch and sglang's own error message both
            # use `--deepep-dispatcher-output-dtype`). Kept as -dtype fp8, which
            # matches the prefill branch's intent.
            DEFAULT_ARGS+=(
                --moe-runner-backend humming
                --deepep-dispatcher-output-dtype fp8
                )
            ;;
    esac
    append_draft_moe_args decode
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
    echo "rdma=$IB_DEVICES (IB_RAIL=$IB_RAIL) socket_ifname=$net_ifname"
    echo "HCU_NUM=$tp_size PD_MODE=$pd_mode PD_OPEN=$pd_open DSPARK=$dspark_enable MOE_MODE=$moe_mode PC_ENABLE=$pc_enable"
    echo "SGLANG_HCU_MEGA_MOE_RUNTIME=$hcu_mega_moe_runtime DSPARK_MOE_MODE=$dspark_moe_mode CHUNK_SIZE=$chunk_size"
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
# On the 12/13.13 fabrics the socket interface follows the RDMA NIC, so a
# multi-node group on IB_RAIL=12 also talks TCP over 12.12.12.x.
if [[ -n "$default_ib_devices" && "$IB_DEVICES" == "$default_ib_devices" ]]; then
    ib_netdev=$(ls "/sys/class/infiniband/$IB_DEVICES/device/net" 2>/dev/null | head -1)
    [[ -z "$ib_netdev" ]] || net_ifname=$ib_netdev
fi

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
hcu_mega_moe_runtime=${SGLANG_HCU_MEGA_MOE_RUNTIME:-megamoe}
chunk_size_default=32768
# if [[ "$pd_mode" == prefill && "$moe_mode" == megamoe ]]; then
#     # Reserve space for the NORMAL scratch buffers of the V4.1 shape.
#     chunk_size_default=16384
# fi
chunk_size=${CHUNK_SIZE:-$chunk_size_default}
if [[ "$pd_mode" != none && -z "${MOE_MODE+x}" ]]; then
    moe_mode=deepep
fi
dspark_moe_mode=${DSPARK_MOE_MODE:-$moe_mode}
# if [[ "$pd_mode" == decode ]]; then
#     dspark_moe_mode=${DSPARK_MOE_MODE:-$moe_mode}
# fi
pc_enable=${PC_ENABLE:-0}
dspark_enable=${DSPARK:-1}
dspark_block_size=${DSPARK_BLOCK_SIZE:-5}
dspark_draft_model_path=${DSPARK_DRAFT_MODEL_PATH:-$model_path}
mem_fraction_default=0.85
# V4.1 counts draft weights in the KV budget; 0.75 is below the loaded
# checkpoint minimum (~0.761). Reserve a viable cache for every MoE backend.
if [[ "$pd_mode" == prefill && "$moe_mode" == megamoe ]]; then
    mem_fraction_default=0.80
fi
mem_fraction_static=${MEM_FRACTION_STATIC:-$mem_fraction_default}
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
case "$dspark_moe_mode" in none|triton|aiter|humming) dspark_moe_mode=none ;; deepep|megamoe) ;; *) die "Invalid DSPARK_MOE_MODE=$dspark_moe_mode (expected: none|deepep|megamoe)" ;; esac
case "$dspark_enable" in 0|1) ;; *) die "DSPARK must be 0 or 1 (got: $dspark_enable)" ;; esac
[[ "$chunk_size" =~ ^[0-9]+$ ]] && (( chunk_size > 0 )) || die "CHUNK_SIZE must be positive"
case "$pc_enable" in 0|1) ;; *) die "PC_ENABLE must be 0 or 1" ;; esac
case "$pd_open" in 0|1) ;; *) die "PD_OPEN must be 0 or 1" ;; esac
if [[ "$pd_open" == 1 && "$pd_mode" == none ]]; then
    die "PD_OPEN=1 requires PD_MODE=prefill or PD_MODE=decode"
fi
# DSpark's CP path is only implemented for the PD prefill worker; a standalone
# prefill service also decodes, so the runtime rejects DSPARK + attn_cp_size>1.
if [[ "$pd_mode" == prefill && "$pd_open" == 0 && "$dspark_enable" == 1 ]] && (( tp_size > 1 )); then
    die "DSPARK + CP only runs on the PD prefill worker: use PD_OPEN=1 for real PD, or DSPARK=0 for the single-node P profile (prefill-only perf testing)."
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
)

# 动态优化点登记处里的环境变量（见文件顶部 OPT-BEGIN 区块）
for _opt_env in "${OPT_ENV[@]}"; do
    env_vars+=("$_opt_env")
done

# 下面这些开关跟着拓扑走，不属于「随时间新增的优化点」，所以留在代码里。
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
        # Runtime-specific configuration is shared by target and draft below.
        :
        ;;
    humming)
        env_vars+=(
            'SGLANG_HUMMING_INPUT_QUANT_CONFIG={"dtype": "float8e4m3", "group_size": 128}'
        )
        ;;
esac

if [[ "$moe_mode" == megamoe || "$dspark_moe_mode" == megamoe ]]; then
    case "$hcu_mega_moe_runtime" in
        megamoe|deep_gemm) ;;
        *) die "Invalid SGLANG_HCU_MEGA_MOE_RUNTIME=$hcu_mega_moe_runtime" ;;
    esac
    # CP admission is per batch: reserve the unsharded chunk on the P side.
    mega_capacity_default=$(((chunk_size + tp_size - 1) / tp_size))
    mega_capacity=${SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK:-$mega_capacity_default}
    [[ "$pd_mode" != prefill ]] || mega_capacity=$mega_capacity_default
    [[ "$pd_mode" != decode ]] || mega_capacity=${SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK:-256}
    [[ "$mega_capacity" =~ ^[0-9]+$ ]] && (( mega_capacity > 0 )) || die "Mega token capacity must be positive"
    env_vars+=(
        "SGLANG_HCU_MEGA_MOE_RUNTIME=$hcu_mega_moe_runtime"
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=$mega_capacity"
    )
    if [[ "$hcu_mega_moe_runtime" == megamoe ]]; then
        env_vars+=(
            "SGLANG_DSV4_CHANNEL_FP8_SCALE=1"
            "MEGAMOE_HCU_BACKEND=${MEGAMOE_HCU_BACKEND:-normal}"
        )
    fi
fi

if [[ "$pd_mode" == decode ]]; then
    env_vars+=(
        "DEEPEP_ENABLE_LL_LAYERED_OPT=1"
        "ROCSHMEM_DISABLE_HDP_FLUSH=1"
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
    --skip-server-warmup
    --mem-fraction-static "$mem_fraction_static"
    --max-running-requests "$max_running_requests"
    --cuda-graph-max-bs-decode 32
    --reasoning-parser auto
    --tool-call-parser auto
    --chunked-prefill-size "$chunk_size"
)
append_distributed_init_args

# DSpark drives speculative decoding; without it the service is a plain
# target-model server (used by the single-node P profile for prefill tests).
if [[ "$dspark_enable" == 1 ]]; then
    DEFAULT_ARGS+=(
        --speculative-algorithm DSPARK
        --speculative-dspark-block-size "$dspark_block_size"
    )
fi

# The original standalone profile runs the MoE through the triton runner;
# the P/D paths select explicit target and draft MoE backends per MOE_MODE.
if [[ "$pd_mode" == none ]]; then
    DEFAULT_ARGS+=(--moe-runner-backend triton)
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

# 动态优化点登记处里的指令（见文件顶部 OPT-BEGIN 区块）
if (( ${#OPT_ARGS[@]} > 0 )); then
    declare -a _opt_words=()
    for _opt_arg in "${OPT_ARGS[@]}"; do
        _opt_words=()
        read -r -a _opt_words <<< "$_opt_arg"
        DEFAULT_ARGS+=("${_opt_words[@]}")
    done
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
