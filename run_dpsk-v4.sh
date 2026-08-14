#!/bin/bash

# Prefill workers register to the local bootstrap server with Python requests.
# Proxy variables inherited from the terminal would route 0.0.0.0:8998 through
# localhost:18086 and leave the bootstrap server with zero registered workers.
# source /home/unset_proxy.sh

# clean the jit cache
# rm -rf ~/.cache # clean jit cache, optional


port=${1:-30000}
model_path=${2:- /parastor/home/public_user/wanglong/DeepSeek-V4-Flash-FP8-Channel}

rank_id=${3:-0}
host_arg="${4:-}"
if [[ -z "$host_arg" ]]; then
    ip="$(hostname -I | awk '{print $1}')"
else
    case "$host_arg" in
        node18) ip="13.13.2.18" ;;
        node20) ip="13.13.2.20" ;;
        node22) ip="13.13.2.22" ;;
        node26) ip="13.13.2.26" ;;
        sglang2) ip="10.16.1.33" ;;
        *)
            echo "Invalid host identifier: $host_arg (expected: node36|node39|node42|node43|node46|node51|node54|node55)"
            exit 1
            ;;
    esac
fi

the_host=$(hostname)
net_ifname=ens19f0
if [ "$the_host" = "nmz26" ] || [ "$the_host" = "nmz20" ] || [ "$the_host" = "nmz22" ] || [ "$the_host" = "nmz18" ] || [ "$the_host" = "nmz15" ] ; then
    net_ifname=ens14f0
fi
if [ "$the_host" = "sglang5" ] ; then
    net_ifname=eth0
fi
if [ "$the_host" = "sglang8" ] ; then
    net_ifname=enp113s0f0np0
fi
if [ "$the_host" = "sglang6" ] ; then
    net_ifname=eth10
fi

if [ "$the_host" = "nmz26" ] ; then
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libibverbs:${LD_LIBRARY_PATH}
fi

is_int8=${IS_INT8:-0}
is_fp8=${IS_FP8:-1}
deepep_mode=${DEEPEP_MODE:-auto}
prefill_or_decode=${PD_MODE:-none}
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

if [[ "$mtp_mode" != "none" && "$mtp_mode" != "dspark" && "$mtp_mode" != "mtp" ]]; then
    echo "Invalid MTP_MODE: $mtp_mode (expected: none|dspark|mtp)"
    exit 1
fi

if [[ "$dspark_moe_mode" != "none" && "$dspark_moe_mode" != "deepep" ]]; then
    echo "Invalid DSPARK_MOE_MODE: $dspark_moe_mode (expected: none|deepep)"
    exit 1
fi
if [[ "$dspark_pd_draft_moe_mode" != "none" && "$dspark_pd_draft_moe_mode" != "deepep" ]]; then
    echo "Invalid DSPARK_PD_DRAFT_MOE_MODE: $dspark_pd_draft_moe_mode (expected: none|deepep)"
    exit 1
fi
if [[ "$mtp_mode" != "dspark" && "$dspark_moe_mode" != "none" ]]; then
    echo "DSPARK_MOE_MODE=$dspark_moe_mode requires MTP_MODE=dspark"
    exit 1
fi

# # The current HCU DeepEP low-latency kernel only accepts the target model's
# # 256-expert layout. DSpark's draft MoE carries an additional shared-expert
# # slot (257 total), so default this combination to the normal DeepEP path.
# # An explicit DEEPEP_MODE still takes precedence for future kernel versions.
# if [[ "$mtp_mode" == "dspark" && "$dspark_moe_mode" == "deepep" && -z "${DEEPEP_MODE+x}" ]]; then
#     deepep_mode=normal
# fi

# DSpark Phase 1/2/3 validation is pure TP. CP/DP remains intentionally
# excluded until the HCU DSpark target-verify path has separate compatibility
# coverage.
# if [[ "$mtp_mode" != "none" && "$prefill_or_decode" != "none" ]]; then
#     echo "MTP_MODE=$mtp_mode currently supports only PD_MODE=none (pure TP DSpark/MTP validation)."
#     exit 1
# fi

if [ "$mtp_mode" = "dspark" ]; then
    case "$dspark_variant" in
        static|compact-sps|compact-sps-sts|compact-nosps|compact-align|cap-accept-sps|g3|g8|sps-record)
            ;;
        *)
            echo "Invalid DSPARK_VARIANT: $dspark_variant"
            echo "Expected: static|compact-sps|compact-sps-sts|compact-nosps|compact-align|cap-accept-sps|g3|g8|sps-record"
            exit 1
            ;;
    esac

    # This named arm is the reproducible Phase-3 SPS+STS test configuration.
    # An explicit DSPARK_CONFIDENCE_STS_PATH still takes precedence so that a
    # workload-specific calibration table can be evaluated without script edits.
    if [ "$dspark_variant" = "compact-sps-sts" ] && [ -z "$dspark_confidence_sts_path" ]; then
        dspark_confidence_sts_path=$dspark_default_sts_path
    fi
fi

rocshmem_env_vars=(
    "ROCSHMEM_DISABLE_HDP_FLUSH=1"
    "ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288"
    "ROCSHMEM_HEAP_SIZE=3173741824"
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128"
    # "ROCSHMEM_IB_GID_INDEX=0" # 需要确认 default_gid(ibstatus)和show_gids对应的index是否匹配
    # "ROCSHMEM_TOPO_FILE_FORCE=/home/proj_dpsk-v4/${the_host}_topo_200g.config"
)
pd_disaggreation_env_vars=(
    "MC_ENABLE_DEST_DEVICE_AFFINITY=1 "
    "UCX_NET_DEVICES=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1"
    "NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1"
    # "MC_TOPO_FILE_FORCE=/home/mc_topo_400g.config"
    "MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200"
)

echo "---- Current Env Variables Setup ------"
env_vars=(
    "NCCL_SOCKET_IFNAME=$net_ifname"
    "GLOO_SOCKET_IFNAME=$net_ifname"
    "NCCL_MIN_NCHANNELS=16"
    "NCCL_MAX_NCHANNELS=16"
    "SGLANG_TORCH_PROFILER_DIR=/home/proj_dpsk-v4/profile"
    "SGLANG_OPT_USE_FUSED_STORE_CACHE=false" # couble be enable, but accuracy drop a lot, need further investigation
    "SGLANG_OPT_USE_FUSED_HASH_TOPK=true"
    "SGLANG_OPT_SWIGLU_CLAMP_FUSION=false" # fused silu_and_mul_clamp kernel is CUDA-only; HIP must disable SWIGLU_CLAMP_FUSION
    "SGLANG_TOPK_TRANSFORM_512_TORCH=false"
    "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true"
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE=0"
    # "SGLANG_ENABLE_SPEC_V2=0" # default to use spec_v2, disable it to use spec_v1
    # "SGLANG_USE_AITER_AR=0" # using aiter allreduce default, disable this to use sglang allreduce
    "SGLANG_USE_AITER_AG=0" # using aiter custom allgather default, disable this to use tp allgather
    # deep_ep
    "${rocshmem_env_vars[@]}"
    # # pd disaggregation
    # "${pd_disaggreation_env_vars[@]}"
    # MoE and gemm kernel optimization
    "SGLANG_ROCM_USE_AITER_MOE=1" # 开启aiter moe算子优化
    "SGLANG_USE_OPT_CAT=1" 
    "SGLANG_USE_FUSED_MLA_CAT=1" 
    "SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=${is_fp8}" # 开启act_quant算子优化
    "SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=${is_fp8}" # 开启silu_mul_fp8_quant融合算子优化
    "SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1" # 开启linear_bf16_fp32算子优化
    "SGLANG_APPLY_CONFIG_BACKUP=none"
    # mHC optimization
    "SGLANG_ROCM_USE_AITER_TILELANG_MHC=1" # 开启aiter mhc算子优化
    # attention optimization
    "SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1" # prefill/decode split mla
    "SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0" # sparse prefill does not support NSA CP yet
    # fused kernel optimization
    "SGLANG_USE_LIGHTOP=1" # enable lightop rope and topk kernel
    "SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1"
    "SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1"
    "SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1"
    "SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1"
    "SGLANG_USE_LIGHTOP_EP_SCATTER=1"
    "SGLANG_USE_LIGHTOP_EP_GATHER=1"
    "SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1"
    # # EPLB static generation
    # "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/home/proj_dpsk-v4/configs/eplb_dump"

    # # debug use
    # "HSA_ENABLE_COREDUMP=1"
    # "GPU_FLUSH_ON_EXECUTION=true"
)

if [ "$pd_open" = "1" ]; then
    env_vars+=(
        "MC_ENABLE_DEST_DEVICE_AFFINITY=1 "
        "UCX_NET_DEVICES=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1"
        "NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1"
        # "MC_TOPO_FILE_FORCE=/home/mc_topo_400g.config"
        "MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200"
    )

fi

if [ "$mtp_mode" != "none" ]; then
    # DSpark FP8-channel checkpoint has BF16 wo_a; its FP8 GEMM fast path is
    # invalid. Keeping this setting on both Phase-1 arms makes the loader
    # configuration explicit and comparable.
    env_vars+=("SGLANG_OPT_FP8_WO_A_GEMM=0")
fi

if [ "$mtp_mode" = "dspark" ]; then
    case "$dspark_variant" in
        compact-sps|compact-sps-sts|compact-nosps|compact-align|sps-record)
            dspark_ragged_verify_mode=compact
            ;;
        cap-accept-sps)
            dspark_ragged_verify_mode=cap-accept
            ;;
        *)
            dspark_ragged_verify_mode=static
            ;;
    esac

    env_vars+=(
        "SGLANG_RAGGED_VERIFY_MODE=$dspark_ragged_verify_mode"
        "SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS=${DSPARK_CONFIDENCE_RELAY_LAG_STEPS:-2}"
        "SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD=${DSPARK_OPT_MARKOV_W2_TP_SHARD:-1}"
        "SGLANG_DSPARK_ENABLE_MULTI_STREAM=${DSPARK_ENABLE_MULTI_STREAM:-1}"
        "SGLANG_DSPARK_FAST_KERNEL=${DSPARK_FAST_KERNEL:-1}"
        "SGLANG_DSPARK_FAST_SAMPLING=${DSPARK_FAST_SAMPLING:-1}"
    )

    if [ "$dspark_moe_mode" = "deepep" ]; then
        env_vars+=(
            "SGLANG_USE_FP8_W8A8_MOE=${is_fp8}"
            "SGLANG_USE_DEEPGEMM_MOE=1"
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${DSPARK_DEEPEP_MAX_DISPATCH_TOKENS:-256}"
        )
    fi

    if [ "$dspark_variant" = "sps-record" ]; then
        env_vars+=(
            "SGLANG_DSPARK_ENABLE_SPS_RECORD=1"
            "SGLANG_SIMULATE_ACC_LEN=${DSPARK_SIMULATE_ACC_LEN:-1.0}"
        )
    fi

    if [ -n "${DSPARK_STS_COLLECT_PATH:-}" ]; then
        env_vars+=("SGLANG_DSPARK_STS_COLLECT_PATH=$DSPARK_STS_COLLECT_PATH")
    fi
fi

if [ "$prefill_or_decode" = "prefill" ] || [ "$prefill_or_decode" = "decode" ]; then
    env_vars+=(
        # ep path to enable deepgemm
        "SGLANG_USE_FP8_W8A8_MOE=${is_fp8}" 
        "SGLANG_USE_DEEPGEMM_MOE=1"
        # pd disaggregation
        "MC_ENABLE_DEST_DEVICE_AFFINITY=1 "
        "UCX_NET_DEVICES=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1"
        "NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1"
        "MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200"
    )
fi
if [ "$moe_mode" = "megamoe" ]; then
    env_vars+=(
        "SGLANG_DCU_MEGA_MOE_RUNTIME=megamoe"
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096"
        "SGLANG_DSV4_CHANNEL_FP8_SCALE=1"
    )
fi  

for kv in "${env_vars[@]}"; do
    export "$kv"
    echo "export $kv"
done

echo "---- Current Running Cmd ------"
tp_size=${HCU_NUM:-8}
pp_size=2
dp_size=$tp_size
ep_size=1
cuda_graph_max_bs=32
mem_fraction_static=0.8
if [ "$mtp_mode" = "dspark" ]; then
    cuda_graph_max_bs=${DSPARK_CUDA_GRAPH_MAX_BS:-32}
    mem_fraction_static=${DSPARK_MEM_FRACTION_STATIC:-0.8}
fi
DEFAULT_ARGS=(
    --reasoning-parser deepseek-v4
    --tool-call-parser deepseekv4
    --tp-size $tp_size
    --dist-timeout 10000
    --watchdog-timeout 3600
    --port $port
    --host 0.0.0.0
    --model-path $model_path
    --disable-radix-cache
    --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}'
    --trust-remote-code
    --chunked-prefill-size 32768
    --disable-flashinfer-autotune
    # --attention-backend dcu_mla
    --skip-server-warmup
    --cuda-graph-max-bs "$cuda_graph_max_bs"
    --mem-fraction-static "$mem_fraction_static"
    # opt1: cp+ep(prefill recommended out of the box)
    
    # # opt2: pp+tp
    # --pipeline-parallel-size $pp_size

    # # opt3: dp+tp+mtp
    # --dp $dp_size
    # --enable-dp-attention
    # --speculative-algo EAGLE
    # --speculative-num-draft-tokens 2
    # --speculative-eagle-topk 1
    # --speculative-num-steps 1

    # opt4: dp+ep+mtp(decode recommended out of the box)

    # # opt5: cp+megamoe(unsupported in v5.12, but ready in v5.13)
    # --enable-prefill-cp
    # --cp-strategy interleave
    # --moe-a2a-backend megamoe

    # # opt6: dp+megamoe+mtp
    # --dp $dp_size
    # --enable-dp-attention
    # --moe-a2a-backend megamoe  
    # --speculative-algo EAGLE
    # --speculative-num-draft-tokens 2
    # --speculative-eagle-topk 1
    # --speculative-num-steps 1 

    # # Additions: pd disaggregation
    # --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
    # --disaggregation-mode $prefill_or_decode

    # # dump data
    # --msprobe-dump-config /home/scripts/acc_test/model_dump/config.json
    # # eplb dump(with SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR), enable it only first running dump
    # --expert-distribution-recorder-mode stat
    # --expert-distribution-recorder-buffer-size -1
    # # eplb static loading
    # --init-expert-location /home/proj_dpsk-v4/eplb_dump/expert_distribution_recorder_1780023319.0668871.pt
    # --ep-dispatch-algorithm static
    # --ep-num-redundant-experts 32
    # --eplb-algorithm deepseek_vec
    
    # --disable-cuda-graph
)

if [ "$mtp_mode" = "dspark" ]; then

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
    if [ "$pd_open" = "1" ] && [ "$prefill_or_decode" = "prefill" ]; then
        DEFAULT_ARGS+=(
            --disable-cuda-graph
            --max-total-tokens "${DSPARK_PD_PREFILL_MAX_TOTAL_TOKENS:-131072}"
            --speculative-moe-a2a-backend none
        )
    else
        DEFAULT_ARGS+=(
            --dp $dp_size
            --enable-dp-attention
            --enable-dp-lm-head
            --ep $tp_size
        )
        if [ "$dspark_moe_mode" = "deepep" ]; then
            DEFAULT_ARGS+=(
                --moe-a2a-backend deepep
                --moe-runner-backend deep_gemm
                --deepep-mode "$deepep_mode"
                --deepep-config /home/proj_dpsk-v4/configs/deepep_IntraConfig.json
            )
            if [ "$pd_open" = "1" ] && [ "$prefill_or_decode" = "decode" ]; then
                if [ "$dspark_pd_draft_moe_mode" = "none" ]; then
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
            else
                DEFAULT_ARGS+=(
                    --speculative-moe-a2a-backend deepep
                    --speculative-moe-runner-backend deep_gemm
                )
            fi
        else
            DEFAULT_ARGS+=(--moe-a2a-backend none)
        fi
    fi

    case "$dspark_variant" in
        compact-sps|compact-sps-sts|compact-align|cap-accept-sps)
            if [ ! -f "$dspark_sps_table_path" ]; then
                echo "DSPARK_VARIANT=$dspark_variant requires an SPS table: $dspark_sps_table_path"
                echo "Generate it first; see /home/proj_sglang_open/dspark/README_bench_hcu.md"
                exit 1
            fi
            DEFAULT_ARGS+=(
                --speculative-dspark-sps-table-path "$dspark_sps_table_path"
            )
            ;;
    esac

    case "$dspark_variant" in
        compact-align)
            DEFAULT_ARGS+=(
                --speculative-dspark-align-verify-tokens-to-graph-tier
            )
            ;;
        g3)
            DEFAULT_ARGS+=(--speculative-dspark-block-size 3)
            ;;
        g8)
            DEFAULT_ARGS+=(--speculative-dspark-block-size 8)
            ;;
    esac

    if [ -n "${DSPARK_BLOCK_SIZE:-}" ]; then
        if [ "$dspark_variant" = "g3" ] || [ "$dspark_variant" = "g8" ]; then
            echo "DSPARK_BLOCK_SIZE cannot be combined with DSPARK_VARIANT=$dspark_variant"
            exit 1
        fi
        DEFAULT_ARGS+=(--speculative-dspark-block-size "$DSPARK_BLOCK_SIZE")
    fi

    if [ -n "$dspark_confidence_sts_path" ]; then
        if [ ! -f "$dspark_confidence_sts_path" ]; then
            echo "DSPARK_CONFIDENCE_STS_PATH does not exist: $dspark_confidence_sts_path"
            exit 1
        fi
        DEFAULT_ARGS+=(
            --speculative-dspark-confidence-sts-path "$dspark_confidence_sts_path"
        )
    fi

    if [ "$dspark_variant" = "compact-sps-sts" ]; then
        echo "DSpark SPS+STS test arm: SPS=$dspark_sps_table_path STS=$dspark_confidence_sts_path"
    fi

    case "${DSPARK_ENABLE_METRICS:-1}" in
        1|true|TRUE|yes|YES)
            DEFAULT_ARGS+=(--enable-metrics)
            ;;
        0|false|FALSE|no|NO)
            ;;
        *)
            echo "Invalid DSPARK_ENABLE_METRICS=${DSPARK_ENABLE_METRICS} (expected boolean)"
            exit 1
            ;;
    esac
elif [ "$mtp_mode" = "mtp" ]; then
    DEFAULT_ARGS+=(
        --speculative-algorithm EAGLE
        --speculative-num-steps 3
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4
        --max-running-requests "${DSPARK_MAX_RUNNING_REQUESTS:-32}"
        --context-length "${DSPARK_CONTEXT_LENGTH:-4096}"
        # --disable-cuda-graph
    )
    # Avoid the known EAGLE target-verify graph in-place-write issue during
    # the initial HCU parity phase.
    # export SGLANG_PREP_IN_CUDA_GRAPH=0
fi

# recommended parallel cmd for prefill/decode
if [ "$prefill_or_decode" = "prefill" ]; then
    # cp limits: strategy=interleave, dp_size=1, tp_size<=8, disable flashmla sparse prefill
    if [ "$moe_mode" = "megamoe" ]; then
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
            --attn-cp-size $tp_size
            --enable-dp-attention
            --moe-a2a-backend deepep
            --deepep-mode "$deepep_mode"
            --deepep-config /home/proj_dpsk-v4/configs/deepep_IntraConfig.json
        )
        if [ "$mtp_mode" = "dspark" ]; then
            DEFAULT_ARGS+=(--moe-runner-backend deep_gemm)
        fi
    fi

    if [ "$pd_open" = "1" ]; then
        DEFAULT_ARGS+=(
            --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
            --disaggregation-mode $prefill_or_decode
            --disaggregation-transfer-backend mooncake
            --disaggregation-bootstrap-port "$pd_bootstrap_port"
            --dist-init-addr "$ip:$pd_prefill_dist_port"
        )
    fi

fi
if [ "$prefill_or_decode" = "decode" ]; then
    dist_init_port=$((port + 733))
    if [ "$mtp_mode" != "dspark" ]; then
        if [ "$moe_mode" = "megamoe" ]; then
            DEFAULT_ARGS+=(--moe-a2a-backend megamoe)
        else
            DEFAULT_ARGS+=(
                --moe-a2a-backend deepep
                --deepep-mode "$deepep_mode"
                --deepep-config /home/proj_dpsk-v4/configs/deepep_IntraConfig.json
            )
        fi
        DEFAULT_ARGS+=(
            --dp $dp_size
            --enable-dp-attention
            --dist-init-addr 127.0.0.1:$dist_init_port
            --speculative-algo EAGLE
            --speculative-num-draft-tokens 4
            --speculative-eagle-topk 1
            --speculative-num-steps 3
        )
    fi
    if [ "$pd_open" = "1" ]; then
        DEFAULT_ARGS+=(
            --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
            --disaggregation-mode $prefill_or_decode
            --disaggregation-transfer-backend mooncake
            --disaggregation-bootstrap-port "$pd_bootstrap_port"
            --dist-init-addr "$ip:$pd_decode_dist_port"
        )
    fi    
fi

if [ "$is_int8" = "1" ]; then
    DEFAULT_ARGS+=(
        --quantization slimquant_marlin
    )
fi

FINAL_ARGS=("${DEFAULT_ARGS[@]}")

# Print command with formatted arguments (one per line)
printf '%s\n' "sglang serve \\"
i=0
while [[ $i -lt ${#FINAL_ARGS[@]} ]]; do
  arg="${FINAL_ARGS[$i]}"
  if [[ "$arg" == --* ]]; then
    next=$((i + 1))
    if [[ $next -lt ${#FINAL_ARGS[@]} && "${FINAL_ARGS[$next]}" != --* ]]; then
      printf ' %s %s \\\n' "$arg" "${FINAL_ARGS[$next]}"
      i=$((i + 2))
    else
      printf ' %s \\\n' "$arg"
      i=$((i + 1))
    fi
  else
    printf ' %s \\\n' "$arg"
    i=$((i + 1))
  fi
done


postfix="${the_host}_${prefill_or_decode}"
printf '%s\n' " 2>&1 | tee running_dpsk-v4_${postfix}.log"
printf '%s\n' "--------------------------------"
sglang serve "${FINAL_ARGS[@]}" 2>&1 | tee running_dpsk-v4_${postfix}.log
