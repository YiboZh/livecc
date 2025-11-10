# Force training to use GPU 0 only and mitigate CUDA memory fragmentation
export CUDA_VISIBLE_DEVICES=0,1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export VIDEO_MIN_PIXELS=39200 # 100*28*28. the minimum visual frame tokens sent to llm is 100
export FPS_MAX_FRAMES=60 # maximum number of frames for each video (480/60/2 = 4min)
export VIDEO_MAX_PIXELS=4816896 # 24576*28*28. the maximum overall video tokens sent to llm is 24k (leave 8k for language)
# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
learning_rate=1e-5 # sft uses 2e-5 lr
run_name="qwen2.5_lr${learning_rate}_$(date +%Y%m%d_%H%M%S)"
WANDB_PROJECT='reactionbench' TOKENIZERS_PARALLELISM=false torchrun --standalone --nproc_per_node=2 train_omni.py \
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
  --annotation_paths /orcd/scratch/seedfund/001/multimodal/qua/reaction_data/livecc_reaction_stream_no_music_10000.jsonl \
  --dataloader_num_workers 4 \
  --use_liger_kernel True \
  --bf16 True \
  --report_to wandb \
  # --freeze_modules thinker.visual,thinker.audio \
  # --freeze_modules visual \
  # --use_lora True \
  # --lora_r 16 \
  # --lora_alpha 32 \
  # --lora_dropout 0.05 \
  # --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  # --modules_to_save "lm_head"


# --annotation_paths /orcd/scratch/orcd/002/qua/data/reaction_data/output_conversation_rewritten.jsonl \\


# /home/qua/code/reaction/livecc/data/reaction_clean/livecc_reactions_clean.jsonl