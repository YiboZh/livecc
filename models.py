import transformers
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = ''
    freeze_modules: list[str] = field(default_factory=lambda: [])
    
    # LoRA arguments
    use_lora: bool = False
    lora_r: int = 16  # Standard LoRA rank for good balance between performance and efficiency
    lora_alpha: int = 32  # Typically 2x of lora_r
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [])  # Auto-detected if empty
    lora_modules_to_save: list[str] = field(default_factory=lambda: [])


@dataclass
class RuntimeArguments:
    run_memory_check: bool = False
    memory_check_sample_index: int = 0
    memory_check_safety_margin_gib: float = 6.0
    memory_check_dtype: str = "bfloat16"
    fail_fast_ddp: bool = True
