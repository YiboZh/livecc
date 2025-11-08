import os
from dataclasses import asdict

DEFAULT_ALLOCATOR_CONFIG = "expandable_segments:True,max_split_size_mb:512"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", DEFAULT_ALLOCATOR_CONFIG)

import torch
import transformers
from transformers import Trainer, AutoProcessor, HfArgumentParser, TrainingArguments, AutoConfig, logging

from models import ModelArguments, RuntimeArguments
from data.lmm_dataset import DataArguments, LMMDataset
from utils.special_tokens import ensure_special_tokens
from utils.memory_utils import apply_runtime_env_defaults, run_memory_preflight_check

logger = logging.get_logger(__name__)


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


if __name__ == "git":
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments, RuntimeArguments))
    training_args, model_args, data_args, runtime_args = parser.parse_args_into_dataclasses()
    apply_runtime_env_defaults(fail_fast_ddp=runtime_args.fail_fast_ddp)

    config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        model_args.pretrained_model_name_or_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )
    for module_name in model_args.freeze_modules:
        logger.warning(f"Freezing module {module_name}")
        getattr(model, module_name).requires_grad_(False)

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

    Trainer(
        model=model,
        args=training_args,
        data_collator=train_dataset.data_collator,
        train_dataset=train_dataset,
        processing_class=processor,
    ).train(resume_from_checkpoint=not training_args.overwrite_output_dir)
