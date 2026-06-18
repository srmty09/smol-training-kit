from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import tomllib

import torch  # type: ignore

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class TrainingConfig:
    model_type: str = "causal_lm"
    model_name: str = "lilLM"
    tokenizer_name: str = "LilTok"
    dataset_name: str = "TinyStories"
    max_steps: int = 10000
    batch_size: int = 8
    grad_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
  
    seed: int = 20040905
    dropout: float = 0.1
  
    logging_steps: int = 10
    save_steps: int = 200
    save_dir: str = "checkpoints"
    use_wandb: bool = True
    wandb_login: str = ""
    wandb_project_name: str = ""
  
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
  
    bf16: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_file(cls, file_path: str):
        file_path = Path(file_path) 
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        if file_path.suffix == ".toml":
            with open(file_path, "rb") as f:
                data = tomllib.load(f)
        elif file_path.suffix in [".yaml", ".yml"]:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError("Only .toml, .yaml, and .yml files are supported")

        return cls(**data)

    def show(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")