# Pre-training Kit

Config and trainer selection for pre-training tasks.

## Config

Use `model_type` to select the trainer.

```yaml
model_type: "causal_lm"

model_name: "sshleifer/tiny-gpt2"
tokenizer_name: "sshleifer/tiny-gpt2"
dataset_name: "tiny-test"

max_steps: 5
batch_size: 2
grad_accumulation_steps: 2
learning_rate: 0.0005
weight_decay: 0.01
warmup_ratio: 0.03
max_grad_norm: 1.0

logging_steps: 1
save_steps: 5
save_dir: "checkpoints"

use_wandb: false
use_lora: false
bf16: false
device: "cpu"
```

## Points

* `model_type` selects the training implementation.
* `model_name` is the model name or local path.
* `tokenizer_name` is the tokenizer name or local path.
* `max_steps` is the number of optimizer updates.
* `batch_size` is the dataloader batch size.
* `grad_accumulation_steps` controls batches before one optimizer update.
* `learning_rate` is the optimizer learning rate.
* `logging_steps` controls logging frequency.
* `save_steps` controls checkpoint saving frequency.
* `save_dir` is the checkpoint folder.
* `use_wandb` enables WandB logging.
* `use_lora` enables LoRA training.
* `bf16` enables bfloat16 training.
* `device` can be `"cpu"` or `"cuda"`.

## Functions

* `TrainingConfig.from_file(path)`
  Load config from YAML or TOML.

* `TrainingConfig.show()`
  Print config values.

* `get_trainer(...)`
  Select trainer using `model_type`.
