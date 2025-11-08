import transformers
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = ''
    freeze_modules: list[str] = field(default_factory=lambda: [])


@dataclass
class RuntimeArguments:
    run_memory_check: bool = False
    memory_check_sample_index: int = 0
    memory_check_safety_margin_gib: float = 6.0
    memory_check_dtype: str = "bfloat16"
    fail_fast_ddp: bool = True
