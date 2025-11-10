import os
from dataclasses import asdict
import sys

DEFAULT_ALLOCATOR_CONFIG = "expandable_segments:True,max_split_size_mb:512"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", DEFAULT_ALLOCATOR_CONFIG)

# Check if LoRA is being used before importing anything that might apply liger_kernel
# This prevents the incompatibility issue between LoRA and liger_kernel
if "--use_lora" in sys.argv and ("True" in sys.argv or "true" in sys.argv):
    os.environ["DISABLE_LIGER_KERNEL"] = "1"
    print("⚠️  LoRA enabled: Disabling liger_kernel to avoid compatibility issues")

import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, AutoProcessor, HfArgumentParser, TrainingArguments, AutoConfig, logging
from transformers.trainer_callback import TrainerCallback

from models import ModelArguments, RuntimeArguments
from data.lmm_dataset import DataArguments, LMMDataset
from utils.special_tokens import ensure_special_tokens
from utils.memory_utils import apply_runtime_env_defaults, run_memory_preflight_check

logger = logging.get_logger(__name__)


class LoRADDPCallback(TrainerCallback):
    """Callback to fix LoRA + DDP + gradient checkpointing compatibility issues."""
    
    def __init__(self):
        self.static_graph_set = False
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Set static graph on DDP model at the start of training."""
        if not self.static_graph_set and args.world_size > 1:
            # Check if model is wrapped in DDP
            if hasattr(model, '_set_static_graph'):
                logger.info("Setting static graph for DDP to fix LoRA + gradient checkpointing compatibility")
                model._set_static_graph()
                self.static_graph_set = True
            elif hasattr(model, 'module') and hasattr(model.module, '_set_static_graph'):
                logger.info("Setting static graph for DDP wrapper to fix LoRA + gradient checkpointing compatibility")
                model._set_static_graph()
                self.static_graph_set = True


def _maybe_run_memory_preflight(model, dataset, training_args, runtime_args):
    if not runtime_args.run_memory_check:
        return
    if not torch.cuda.is_available():
        logger.warning("Skipping memory preflight because CUDA is not available.")
        return
    if len(dataset) == 0:
        logger.warning("Skipping memory preflight because the dataset is empty.")
        return
    global_rank = int(os.environ.get("RANK", "-1"))
    if global_rank not in (-1, 0):
        logger.info("Memory preflight runs on global rank 0 only; skipping on this process.")
        return

    sample_index = max(0, min(runtime_args.memory_check_sample_index, len(dataset) - 1))
    logger.info(f"Running memory preflight on sample index {sample_index}.")
    sample_batch = dataset[sample_index]
    batch_mapping = dict(sample_batch)
    device_index = training_args.local_rank if training_args.local_rank != -1 else 0
    device = torch.device("cuda", device_index)
    fits, peak_bytes, total_bytes = run_memory_preflight_check(
        model,
        batch_mapping,
        device=device,
        dtype=runtime_args.memory_check_dtype,
        safety_margin_gib=runtime_args.memory_check_safety_margin_gib,
    )
    peak_gib = peak_bytes / (1024**3)
    total_gib = total_bytes / (1024**3)
    logger.info(f"Memory preflight peak={peak_gib:.2f} GiB of {total_gib:.2f} GiB capacity.")
    if not fits:
        raise RuntimeError(
            "Memory preflight failed: peak usage exceeds device capacity minus the safety margin. "
            "Reduce per-device batch size, frame count, or resolution before retrying."
        )


def _parse_comma_separated_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_default_lora_target_modules(model_architecture: str) -> list[str]:
    """Get default LoRA target modules based on model architecture."""
    # For most transformer models (Qwen, LLaMA, etc.), these are the standard modules
    # Includes both attention layers and MLP layers for best performance
    default_modules = ["q_proj", "v_proj"]
    logger.info(f"Using default LoRA target modules for {model_architecture}: {default_modules}")
    return default_modules



if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments, RuntimeArguments))
    training_args, model_args, data_args, runtime_args = parser.parse_args_into_dataclasses()
    apply_runtime_env_defaults(fail_fast_ddp=runtime_args.fail_fast_ddp)
    
    config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        model_args.pretrained_model_name_or_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )
    
    # Freeze modules BEFORE applying LoRA (important for correct gradient flow)
    for module_name in model_args.freeze_modules:
        logger.warning(f"Freezing module {module_name} before LoRA application")
        getattr(model, module_name).requires_grad_(False)
    
    # Apply LoRA if configured
    if model_args.use_lora:
        logger.info("Applying LoRA configuration")
        
        # Use default target modules if none specified
        target_modules = model_args.lora_target_modules
        if not target_modules:
            target_modules = _get_default_lora_target_modules(config.architectures[0])
        
        # Use default modules to save if none specified
        modules_to_save = model_args.lora_modules_to_save
        if not modules_to_save:
            modules_to_save = None  # Let PEFT handle defaults
        
        logger.info(f"LoRA config: r={model_args.lora_r}, alpha={model_args.lora_alpha}, dropout={model_args.lora_dropout}")
        logger.info(f"Target modules: {target_modules}")
        if modules_to_save:
            logger.info(f"Modules to save: {modules_to_save}")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    

    if "Qwen2VL" in model.config.architectures[0]:
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", padding_side="right"
        )  # Qwen2vl-base processor has some bugs. otherwise we do not need this
    else:
        processor = AutoProcessor.from_pretrained(
            model_args.pretrained_model_name_or_path, padding_side="right"
        )
    ensure_special_tokens(processor, model=model)

    train_dataset = LMMDataset(
        **asdict(data_args),
        **asdict(training_args),
        **asdict(model_args),
        processor=processor,
    )

    _maybe_run_memory_preflight(model, train_dataset, training_args, runtime_args)

    # Prepare callbacks for LoRA + DDP compatibility
    callbacks = []
    if model_args.use_lora:
        callbacks.append(LoRADDPCallback())
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=train_dataset.data_collator,
        train_dataset=train_dataset,
        processing_class=processor,
        callbacks=callbacks,
    )
    
    trainer.train(resume_from_checkpoint=not training_args.overwrite_output_dir)
