# sft_omni.sh (multi-GPU)
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Make all 4 GPUs visible (or just don't set this var at all)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# (Optional) if you have no Infiniband on this box:
export NCCL_IB_DISABLE=1

# (Optional) pick a free port to avoid collisions
export MASTER_PORT=$((RANDOM%20000+20000))

export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export VIDEO_MIN_PIXELS=39200
export FPS_MAX_FRAMES=60
export VIDEO_MAX_PIXELS=4816896

learning_rate=1e-5
run_name="qwen2.5_lr${learning_rate}_$(date +%Y%m%d_%H%M%S)"

WANDB_PROJECT='reactionbench' TOKENIZERS_PARALLELISM=false \
deepspeed --num_gpus=4 train_omni.py \
  --deepspeed scripts/deepspeed_zero2.json \
  --output_dir /orcd/scratch/orcd/002/qua/data/reaction_data/checkpoints/$run_name \
  --overwrite_output_dir True \
  --run_name $run_name \
  --save_on_each_node True \
  --do_train True \
  --eval_strategy no \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate $learning_rate \
  --warmup_ratio 0.10 \
  --weight_decay 0.01 \
  --optim adamw_torch \
  --lr_scheduler_type constant_with_warmup \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 1000 \
  --save_total_limit 5 \
  --gradient_checkpointing True \
  --max_grad_norm 0.5 \
  --pretrained_model_name_or_path /orcd/scratch/seedfund/001/multimodal/qua/huggingface/hub/Qwen2.5-Omni-7B \
  --annotation_paths /home/qua/code/reaction/livecc/data/reaction_clean/livecc_reactions_clean.jsonl \
  --dataloader_num_workers 4 \
  --use_liger_kernel True \
  --bf16 True \
  --report_to wandb
