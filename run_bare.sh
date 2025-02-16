#!/bin/bash

GPUs=(0)

# Define default values for the arguments
MODEL_IDS=("mistralai/Mistral-7B-Instruct-v0.2")
ITER=50
MAX_NEW_TOKENS=512
CACHE_DIR="/path/to/where/you/store/hf/models"
OUTPUT_FOLDER="results/SLIA"
TEST_FOLDER="correct/SLIA"
PROMPT_FOLDER="prompts/SLIA"

function check_gpu_free {
    local gpu_id=$1
    local gpu_util=$(nvidia-smi -i $gpu_id --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    [ $gpu_util -lt 50 ] # returns true if GPU is less than 50% utilized
}

for MODEL_ID in "${MODEL_IDS[@]}"; do
    found_free_gpu=false
    while [ "$found_free_gpu" = false ]; do
        for gpu in "${GPUs[@]}"; do
            if check_gpu_free $gpu; then
                echo "GPU $gpu is free. Running model: $MODEL_ID, on GPU: $gpu"
                CUDA_VISIBLE_DEVICES=$gpu python run_inference_bare.py \
                --model_id "$MODEL_ID" \
                --cache_dir "$CACHE_DIR" \
                --num_return_sequences 1 \
                --repetition_penalty 1.0 \
                --iter $ITER \
                --temperature 1.0 \
                --top_p 1.0 \
                --top_k 0 \
                --max_new_tokens $MAX_NEW_TOKENS \
                --dtype "float32" \
                --output_folder "$OUTPUT_FOLDER" \
                --test_folder "$TEST_FOLDER" \
                --prompt_folder "$PROMPT_FOLDER" \
                --seed 42 \
                --device "cuda" &
                found_free_gpu=true
                break
            fi
        done
        if [ "$found_free_gpu" = false ]; then
            echo "All GPUs are busy. Waiting for 60 seconds..."
            sleep 60
        fi
    done
done

wait
echo "All experiments have finished."
