from .TrainingConfig import TrainingConfig


def get_trainer(
    training_cfg: TrainingConfig,
    train_dataloader,
    model=None,
    eval_dataset=None,
    tokenizer=None,
    optimizer=None,
    lr_scheduler=None,
):
    if training_cfg.model_type == "causal_lm":
        from .causal_lm.Trainer import Trainer

        return Trainer(
            training_cfg=training_cfg,
            train_dataloader=train_dataloader,
            model=model,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    raise ValueError(f"Unknown model_type: {training_cfg.model_type}")