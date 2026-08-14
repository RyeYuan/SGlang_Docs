#!/bin/bash
# rm -rf ~/.cache/
# rm -rf ~/.triton/

# 健康检查环境变量
export SGLANG_HEALTH_CHECK_TIMEOUT=180

export SGLANG_ENABLE_SPEC_V2=1

export NCCL_SOCKET_IFNAME=enp113s0f0np0
export GLOO_SOCKET_IFNAME=enp113s0f0np0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1 
export UCX_NET_DEVICES=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16

export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_ROCM_USE_AITER_MOE=1 
export SGLANG_USE_OPT_CAT=1 
export SGLANG_USE_FUSED_MLA_CAT=1 
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1 
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=1 
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1 
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export HIP_H2D_DISABLE_COPY_BUFFER=0 
export HIP_D2H_DISABLE_COPY_BUFFER=0 
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768 
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768 
export HIP_D2H_DIRECT_COPY_THRESHOLD=512 
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512 
export USE_DCU_CUSTOM_ALLREDUCE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1 
export SGLANG_USE_LIGHTOP=1 
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_TOPK_TRANSFORM_512_TORCH=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_APPLY_CONFIG_BACKUP=none

# mooncake
# export MC_TOPO_FILE_FORCE=/home/mc_topo_400g.config
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200

option+=" --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 "
option+=" --disaggregation-mode decode "

time=$(date +"%Y%m%d-%H%M%S")
nodes=1
rank=0
host_ip=12.12.12.18
master_ip=$host_ip

# --speculative-algorithm EAGLE \
# --speculative-num-steps 1 \
# --speculative-eagle-topk 1 \
# --speculative-num-draft-tokens 2 \
# --dist-init-addr $master_ip:5123 --nnodes $nodes --node-rank $rank \
sglang serve ${option} \
  --trust-remote-code \
  --model-path /home/model/DeepSeek-V4-Flash-FP8-Channel \
  --tp-size 8 \
  --chunked-prefill-size 16384 \
  --disable-flashinfer-autotune \
  --dp-size 8 \
  --enable-dp-attention --moe-dense-tp-size 1  --enable-dp-lm-head \
  --host $host_ip  --port 30001  \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --dist-init-addr $master_ip:5123 --nnodes $nodes --node-rank $rank \
  2>&1 | tee serve-d_${time}.log