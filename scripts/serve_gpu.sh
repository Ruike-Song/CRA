#!/bin/bash
# [REUSED] This script is reused from compute-optimal-tts
# (https://github.com/RyanLiu112/compute-optimal-tts).
# It starts the FastChat controller and workers (reward model + policy model).

POLICY_MODEL_PATH=$1
VALUE_MODEL_PATH=$2
HOST_ADDR=$3
CONTROLLER_PORT=$4
WORKER_BASE_PORT=$5
PYTHON_EXECUTABLE=${6:-python}
NUM_RM_WORKER=${7:-1}
NUM_LM_WORKER=${8:-1}
VISIBLE_DEVICES=${9:-0}
MODEL_TYPE=${10:-step_logit}

export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
n_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "n_gpus: $n_gpus"

IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
echo "GPU_LIST:"
echo "${GPU_LIST[@]}"

export PYTHONPATH=$(pwd)

LOGDIR=${PYTHONPATH}/logs_fastchat
export LOGDIR=$LOGDIR
session_name=tts
if tmux has-session -t $session_name 2>/dev/null; then
    echo "Session $session_name already exists. Killing it."
    tmux kill-session -t $session_name
fi
tmux start-server
tmux new-session -s $session_name -n controller -d
tmux send-keys "export LOGDIR=${LOGDIR} && cd ${PYTHONPATH}" Enter
tmux send-keys "${PYTHON_EXECUTABLE} -m fastchat.serve.controller --port ${CONTROLLER_PORT} --host $HOST_ADDR" Enter

echo "Wait 5 seconds ..."
sleep 5

echo "Starting workers"
for i in $(seq 0 $((NUM_RM_WORKER-1)))
do
    WORKER_PORT=$((WORKER_BASE_PORT+i))
    tmux new-window -n reward_$i
    tmux send-keys "export LOGDIR=${LOGDIR} && cd ${PYTHONPATH}" Enter
    if [[ "$VALUE_MODEL_PATH" =~ "dummy" ]]; then
        command="pwd"
    else
        command="CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} ${PYTHON_EXECUTABLE} -m compute_optimal_tts.llm_service.workers.reward_model_worker --model-path $VALUE_MODEL_PATH --model-type $MODEL_TYPE --controller-address http://$HOST_ADDR:$CONTROLLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT"
    fi
    tmux send-keys "$command" Enter
    echo "Reward worker $i started on GPU ${GPU_LIST[$i]} with port $WORKER_PORT, model: $VALUE_MODEL_PATH"
done

for i in $(seq $((NUM_RM_WORKER)) $((NUM_LM_WORKER+NUM_RM_WORKER-1)))
do
    WORKER_PORT=$((WORKER_BASE_PORT+i))
    tmux new-window -n policy_$i
    tmux send-keys "export LOGDIR=${LOGDIR} && cd ${PYTHONPATH}" Enter

    command="CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} ${PYTHON_EXECUTABLE} -m compute_optimal_tts.llm_service.workers.model_worker --model-path $POLICY_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT"

    tmux send-keys "$command" Enter
    echo "Policy worker $i started on GPU ${GPU_LIST[$i]} with port $WORKER_PORT, model: $POLICY_MODEL_PATH"
done
