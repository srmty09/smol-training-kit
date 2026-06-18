# Causal LM Pre-training

Small implementation for causal language model pre-training.

## Model Type

Use:

```yaml
model_type: "causal_lm"
```

## Usage

```python
from pre_training_kit import TrainingConfig, get_trainer

cfg = TrainingConfig.from_file("config.yaml")

trainer = get_trainer(
    training_cfg=cfg,
    train_dataloader=train_loader,
    tokenizer=tokenizer,
)

trainer.train()
```

## Functions

* `trainer.train()`
  Start training.

* `trainer.evaluate()`
  Run evaluation.

* `trainer.save_checkpoint(name)`
  Save checkpoint.

* `trainer.load_checkpoint(path)`
  Load checkpoint.

## Output

Checkpoints are saved in:

```text
save_dir/checkpoint-step
save_dir/final
```
