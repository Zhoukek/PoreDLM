from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .config import DDPGradSyncMode, DistributedStrategy
from .eval import Evaluator
from .torch_util import barrier, get_world_size, move_to_device
from .train import Trainer


class DLMTrainer(Trainer):
    def train_micro_batch(
        self, micro_batch: Dict[str, Any], batch_size_in_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        del batch_size_in_tokens
        output = self.dist_model(
            input_ids=micro_batch["input_ids"],
            attention_mask=micro_batch.get("attention_mask"),
            attention_bias=micro_batch.get("attention_bias"),
            diffusion=True,
            dlm_t_min=self.cfg.dlm.t_min,
            dlm_t_max=self.cfg.dlm.t_max,
            dlm_reduction="mean",
        )
        loss = output.loss * self.cfg.dlm.loss_weight
        return loss, loss.detach(), None

    def train_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        micro_batches = self.split_batch(batch)
        del batch

        batch_loss = torch.tensor(0.0, device=self.device)
        num_micro_batches = len(micro_batches)

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            grad_sync_context = nullcontext
            if (
                self.cfg.distributed_strategy == DistributedStrategy.ddp
                and self.cfg.ddp is not None
                and self.cfg.ddp.grad_sync_mode == DDPGradSyncMode.batch
                and micro_batch_idx != num_micro_batches - 1
            ):
                grad_sync_context = self.dist_model.no_sync

            output_hooks: List[torch.utils.hooks.RemovableHandle] = []
            output_hooks += self._setup_module_output_save_hooks(micro_batch_idx)

            with grad_sync_context():
                autocast_device = "mps" if self.device.type == "mps" else "cuda"
                with torch.autocast(autocast_device, enabled=True, dtype=self.cfg.autocast_precision):
                    loss, logged_loss, _ = self.train_micro_batch(micro_batch, 0)
                    loss = loss / num_micro_batches
                    batch_loss += logged_loss.detach() / num_micro_batches
                loss.backward()

            for hook in output_hooks:
                hook.remove()

        return batch_loss, None

    def train_step(self, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if self.indices_file is not None and "index" in batch:
            indices = "\t".join(str(int(i)) for i in batch["index"])
            self.indices_file.write(f"{self.global_step}\t{indices}\n")

        if (instance_mask := batch.get("instance_mask")) is not None:
            metrics["train/masked_instances_local_rank"] = (~instance_mask).sum().item()

        self.optim.zero_grad(set_to_none=True)
        batch = move_to_device(batch, self.device)

        batch_loss, _ = self.train_batch(batch)

        if reduce_global_loss:
            dist.reduce(batch_loss, 0)
            batch_loss.div_(get_world_size())

        should_log_optim_metrics_this_step = self.should_log_optim_metrics_this_step()
        optim_metrics = self.optim.clip_grads_and_collect_metrics(
            self.global_step,
            collect_param_metrics=should_log_optim_metrics_this_step,
            process_group=self.dist_model.process_group,
        )

        for group in self.optim.param_groups:
            group["lr"] = self.scheduler.get_lr(
                self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max
            )
            group["max_grad_norm"] = self.scheduler.get_max_grad_norm(
                self.cfg.max_grad_norm, self.scheduler_current, self.scheduler_max
            )
            group["max_grad_norm_ratio"] = self.scheduler.get_max_grad_norm(
                self.cfg.max_grad_norm_ratio, self.scheduler_current, self.scheduler_max
            )

        self.optim.step()

        if torch.isnan(batch_loss):
            raise ValueError("nan DLM loss encountered")
        for key, value in optim_metrics.items():
            metrics[f"optim/{key}"] = value.item()
        self.cur_train_loss = batch_loss.item()
        self.min_train_loss = min(self.min_train_loss, self.cur_train_loss)
        metrics["train/DLMLoss"] = self.cur_train_loss

        if should_log_optim_metrics_this_step:
            optim_metrics = self.optim.get_post_step_metrics(
                self.dist_model, process_group=self.dist_model.process_group
            )
            for key, value in optim_metrics.items():
                metrics[f"optim/{key}"] = value.item()

        return metrics

    def eval_step(self, batch: Dict[str, Any], evaluator: Evaluator) -> None:
        batch = move_to_device(batch, self.device)
        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                output = self.dist_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    attention_bias=batch.get("attention_bias"),
                    diffusion=True,
                    dlm_t_min=self.cfg.dlm.t_min,
                    dlm_t_max=self.cfg.dlm.t_max,
                    dlm_reduction="mean",
                )
        loss = output.loss.detach().expand(batch["input_ids"].shape[0])
        dummy_logits = torch.empty(
            (*batch["input_ids"].shape, 0),
            device=batch["input_ids"].device,
            dtype=loss.dtype,
        )
        evaluator.update_metrics(batch, loss, dummy_logits)
        barrier()
