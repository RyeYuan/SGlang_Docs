# baseline
EPLB_DISPATCH=lp SGLANG_LPLB_HIP_SOLVER=device \
EPLB_LOCATION=/home/proj_dpsk-v4/expert_dist/eplb_fp8_runB64_r32.pt \
EPLB_REDUNDANT=32 DSPARK=1 \
PYTHONPATH=/home/proj_sglang_fork/sglang-das/python:$PYTHONPATH \
PD_MODE=prefill PD_OPEN=1 MOE_MODE=megamoe DSPARK_MOE_MODE=megamoe \
SGLANG_HCU_MEGA_MOE_RUNTIME=deep_gemm \
MAX_RUNNING_REQUESTS=64 MAX_PREFILL_TOKENS=32768 CHUNK_SIZE=32768 MEM_FRACTION_STATIC=0.80 \
bash run_dpsk-v41.sh 10015 \
/module/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8-wo_a-markov_head-lm_head


# Optimization cmd
MODEL_PATH=/module/DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8-wo_a-markov_head-lm_head \
PYTHONPATH=/home/proj_sglang_fork/sglang-das/python:$PYTHONPATH \
EPLB_DISPATCH=lp SGLANG_LPLB_HIP_SOLVER=device \
EPLB_LOCATION=/home/proj_dpsk-v4/expert_dist/C-r32-ht.pt EPLB_REDUNDANT=32 \
DSPARK=1 PC_ENABLE=0 PD_MODE=prefill PD_OPEN=1 MOE_MODE=megamoe DSPARK_MOE_MODE=megamoe \
SGLANG_HCU_MEGA_MOE_RUNTIME=deep_gemm SGLANG_DSV4_CP_OUTPUT_INPLACE=0 \
MAX_RUNNING_REQUESTS=64 MAX_PREFILL_TOKENS=32768 CHUNK_SIZE=32768 MEM_FRACTION_STATIC=0.80 \
bash /home/scripts/SGlang_Docs/run_dpsk-v41.sh 10015 "$MODEL_PATH"