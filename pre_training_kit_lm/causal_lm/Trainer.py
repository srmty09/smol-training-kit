import os
import random
import time

import torch  # type: ignore
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup  # type: ignore

from ..TrainingConfig import TrainingConfig


class Trainer:
    def __init__(
        self,
        training_cfg: TrainingConfig,
        train_dataloader,
        model=None,
        eval_dataset=None,
        tokenizer=None,
        optimizer=None,
        lr_scheduler=None,
    ):
        self.train_cfg = training_cfg
        self._set_seed(training_cfg.seed)

        self.model = model if model is not None else AutoModelForCausalLM.from_pretrained(
            training_cfg.model_name,
            torch_dtype=torch.bfloat16 if training_cfg.bf16 else None,
        )

        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            training_cfg.tokenizer_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if training_cfg.use_lora:
            self._apply_lora()

        self.model = self.model.to(self.train_cfg.device)

        self.train_dataloader = train_dataloader
        self.eval_dataset = eval_dataset
        self.global_step = 0

        self.optimizer = optimizer if optimizer is not None else AdamW(
            self.model.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )

        total_training_steps = self.train_cfg.max_steps
        warmup_steps = int(total_training_steps * self.train_cfg.warmup_ratio)

        self.lr_scheduler = lr_scheduler if lr_scheduler is not None else get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        if training_cfg.use_wandb:
            if training_cfg.wandb_login:
                wandb.login(key=training_cfg.wandb_login)

            wandb.init(
                project=training_cfg.wandb_project_name or None,
                config=training_cfg.__dict__,
            )

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _apply_lora(self):
        try:
            from peft import LoraConfig, get_peft_model  # type: ignore
        except ImportError as e:
            raise ImportError("LoRA needs peft. Install it using: pip install peft") from e

        lora_cfg = LoraConfig(
            r=self.train_cfg.lora_r,
            lora_alpha=self.train_cfg.lora_alpha,
            lora_dropout=self.train_cfg.lora_dropout,
            target_modules=self.train_cfg.lora_target_modules,
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

    def _get_grad_norm(self):
        total_norm_sq = 0.0

        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                total_norm_sq += param_norm.item() ** 2

        return total_norm_sq ** 0.5

    def _optimizer_step(self, accum_tokens: int, accum_start_time: float):
        grad_norm = self._get_grad_norm()

        if self.train_cfg.max_grad_norm is not None and self.train_cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_cfg.max_grad_norm,
            )

        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        self.global_step += 1

        elapsed = time.time() - accum_start_time
        tokens_per_second = accum_tokens / elapsed if elapsed > 0 else 0.0
        current_lr = self.optimizer.param_groups[0]["lr"]

        return current_lr, tokens_per_second, grad_norm

    def _log_and_save(
        self,
        current_loss,
        perplexity,
        current_lr,
        tokens_per_second,
        grad_norm,
    ):
        if (
            self.train_cfg.use_wandb
            and self.train_cfg.logging_steps > 0
            and self.global_step % self.train_cfg.logging_steps == 0
        ):
            wandb.log(
                {
                    "train/loss": current_loss,
                    "train/perplexity": perplexity,
                    "train/lr": current_lr,
                    "train/tokens_per_second": tokens_per_second,
                    "train/grad_norm": grad_norm,
                },
                step=self.global_step,
            )

        if (
            self.train_cfg.save_steps > 0
            and self.global_step % self.train_cfg.save_steps == 0
        ):
            self.save_checkpoint(f"checkpoint-{self.global_step}")

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()

        current_lr = self.optimizer.param_groups[0]["lr"]
        tokens_per_second = 0.0
        grad_norm = 0.0

        current_loss = 0.0
        perplexity = 0.0

        accum_tokens = 0
        accum_start_time = time.time()

        pbar = tqdm(total=self.train_cfg.max_steps, desc="Training", leave=True)

        while self.global_step < self.train_cfg.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                if self.global_step >= self.train_cfg.max_steps:
                    break

                inputs = {
                    key: value.to(self.train_cfg.device)
                    for key, value in batch.items()
                }

                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    labels=inputs["labels"],
                )

                loss = outputs.loss
                loss = loss / self.train_cfg.grad_accumulation_steps
                loss.backward()

                accum_tokens += inputs["input_ids"].numel()

                current_loss = loss.item() * self.train_cfg.grad_accumulation_steps
                perplexity = torch.exp(torch.tensor(current_loss)).item()

                should_step = (batch_idx + 1) % self.train_cfg.grad_accumulation_steps == 0

                if should_step:
                    current_lr, tokens_per_second, grad_norm = self._optimizer_step(
                        accum_tokens,
                        accum_start_time,
                    )

                    self._log_and_save(
                        current_loss,
                        perplexity,
                        current_lr,
                        tokens_per_second,
                        grad_norm,
                    )

                    accum_tokens = 0
                    accum_start_time = time.time()

                    pbar.update(1)

                    pbar.set_postfix(
                        loss=f"{current_loss:.4f}",
                        perplexity=f"{perplexity:.2f}",
                        lr=f"{current_lr:.2e}",
                        grad_norm=f"{grad_norm:.4f}",
                        tok_s=f"{tokens_per_second:.0f}",
                    )

                    if self.global_step >= self.train_cfg.max_steps:
                        break

            if self.eval_dataset is not None:
                self.evaluate()

        pbar.close()
        self.save_checkpoint("final")

        if self.train_cfg.use_wandb:
            wandb.finish()

    def evaluate(self):
        self.model.eval()

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
            collate_fn=getattr(self.train_dataloader, "collate_fn", None),
        )

        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in eval_loader:
                inputs = {
                    key: value.to(self.train_cfg.device)
                    for key, value in batch.items()
                }

                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    labels=inputs["labels"],
                )

                loss = outputs.loss

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        print(
            f"-eval- step {self.global_step} | "
            f"loss {avg_loss:.4f} | "
            f"perplexity {perplexity:.2f}"
        )

        if self.train_cfg.use_wandb:
            wandb.log(
                {
                    "eval/loss": avg_loss,
                    "eval/perplexity": perplexity,
                },
                step=self.global_step,
            )

        self.model.train()
        return avg_loss

    def save_checkpoint(self, name: str):
        save_path = os.path.join(self.train_cfg.save_dir, name)
        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        torch.save(
            {
                "global_step": self.global_step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.lr_scheduler.state_dict(),
            },
            os.path.join(save_path, "trainer_state.pt"),
        )

        print(f"saved checkpoint {save_path}")

    def load_checkpoint(self, path: str):
        loaded_model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16 if self.train_cfg.bf16 else None,
        )

        self.model.load_state_dict(loaded_model.state_dict())
        self.model.to(self.train_cfg.device)

        del loaded_model

        self.tokenizer = AutoTokenizer.from_pretrained(path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        state_path = os.path.join(path, "trainer_state.pt")

        if os.path.exists(state_path):
            state = torch.load(
                state_path,
                map_location=self.train_cfg.device,
            )

            if state.get("optimizer") is not None:
                self.optimizer.load_state_dict(state["optimizer"])

            if state.get("scheduler") is not None:
                self.lr_scheduler.load_state_dict(state["scheduler"])

            self.global_step = state.get("global_step", 0)

        print(
            f"resumed from checkpoint -> {path} "
            f"(global_step={self.global_step})"
        )
