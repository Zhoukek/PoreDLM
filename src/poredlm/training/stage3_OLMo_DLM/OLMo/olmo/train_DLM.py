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
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        del batch_size_in_tokens
        # 打印 micro_batch 的所有内容
        print("=" * 50)
        print("micro_batch first sample contents:")
        for key, value in micro_batch.items():
            if isinstance(value, torch.Tensor):
                # 取第一个样本
                first_sample = value[0]
                print(f"  {key}: shape={value.shape}, first_sample_shape={first_sample.shape}, dtype={first_sample.dtype}, device={first_sample.device}")
                # 输出第一个样本的所有值（因为长度是1000）
                print(f"    values: {first_sample.tolist()}")
                print(f"    values length: {len(first_sample.tolist())}")
            else:
                print(f"  {key}: {value} (type: {type(value)})")
        print("=" * 50)

        output = self.dist_model(
            input_ids=micro_batch["input_ids"],
            encoder_attention_mask=micro_batch.get("encoder_attention_mask"),
            attention_mask=micro_batch.get("attention_mask"),
            attention_bias=micro_batch.get("attention_bias"),
            cond_seq_mask=micro_batch.get("cond_seq_mask"),
            label_drop_mask=micro_batch.get("label_drop_mask"),
            elf_diffusion=True,
            label_drop_prob=self.cfg.dlm.label_drop_prob,
            denoiser_p_mean=self.cfg.dlm.denoiser_p_mean,
            denoiser_p_std=self.cfg.dlm.denoiser_p_std,
            denoiser_noise_scale=self.cfg.dlm.denoiser_noise_scale,
            t_eps=self.cfg.dlm.t_eps,
            time_schedule=self.cfg.dlm.time_schedule,
            decoder_prob=self.cfg.dlm.decoder_prob,
            decoder_noise_scale=self.cfg.dlm.decoder_noise_scale,
            decoder_p_mean=self.cfg.dlm.decoder_p_mean,
            decoder_p_std=self.cfg.dlm.decoder_p_std,
            self_cond_prob=self.cfg.dlm.self_cond_prob,
            self_cond_cfg_min=self.cfg.dlm.self_cond_cfg_min,
            self_cond_cfg_max=self.cfg.dlm.self_cond_cfg_max,
            num_self_cond_cfg_tokens=self.cfg.dlm.num_self_cond_cfg_tokens,
        )
        loss = output.loss * self.cfg.dlm.loss_weight
        metrics = {
            "l2_loss": output.l2_loss.detach(),
            "ce_loss": output.ce_loss.detach(),
            "decoder_step_active": output.decoder_step_active.detach().to(dtype=torch.float32),
        }
        return loss, loss.detach(), metrics

    def train_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        micro_batches = self.split_batch(batch)
        del batch

        batch_loss = torch.tensor(0.0, device=self.device)
        batch_l2_loss = torch.tensor(0.0, device=self.device)
        batch_ce_loss = torch.tensor(0.0, device=self.device)
        batch_decoder_frac = torch.tensor(0.0, device=self.device)
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
                    loss, logged_loss, micro_metrics = self.train_micro_batch(micro_batch, 0)
                    loss = loss / num_micro_batches
                    batch_loss += logged_loss.detach() / num_micro_batches
                    if micro_metrics is not None:
                        batch_l2_loss += micro_metrics["l2_loss"] / num_micro_batches
                        batch_ce_loss += micro_metrics["ce_loss"] / num_micro_batches
                        batch_decoder_frac += micro_metrics["decoder_step_active"] / num_micro_batches
                loss.backward()

            for hook in output_hooks:
                hook.remove()

        metrics = {
            "l2_loss": batch_l2_loss.detach(),
            "ce_loss": batch_ce_loss.detach(),
            "decoder_step_frac": batch_decoder_frac.detach(),
        }
        return batch_loss, metrics

    def train_step(self, batch: Dict[str, Any], reduce_global_loss: bool = True) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if self.indices_file is not None and "index" in batch:
            indices = "\t".join(str(int(i)) for i in batch["index"])
            self.indices_file.write(f"{self.global_step}\t{indices}\n")

        if (instance_mask := batch.get("instance_mask")) is not None:
            metrics["train/masked_instances_local_rank"] = (~instance_mask).sum().item()

        self.optim.zero_grad(set_to_none=True)
        batch = move_to_device(batch, self.device)

        batch_loss, dlm_metrics = self.train_batch(batch)

        if reduce_global_loss:
            dist.reduce(batch_loss, 0)
            batch_loss.div_(get_world_size())
            if dlm_metrics is not None:
                for value in dlm_metrics.values():
                    dist.reduce(value, 0)
                    value.div_(get_world_size())

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
        if dlm_metrics is not None:
            metrics["train/L2Loss"] = dlm_metrics["l2_loss"].item()
            metrics["train/CELoss"] = dlm_metrics["ce_loss"].item()
            metrics["train/DecoderStepFrac"] = dlm_metrics["decoder_step_frac"].item()

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
                    encoder_attention_mask=batch.get("encoder_attention_mask"),
                    attention_mask=batch.get("attention_mask"),
                    attention_bias=batch.get("attention_bias"),
                    cond_seq_mask=batch.get("cond_seq_mask"),
                    label_drop_mask=batch.get("label_drop_mask"),
                    elf_diffusion=True,
                    label_drop_prob=0.0,
                    denoiser_p_mean=self.cfg.dlm.denoiser_p_mean,
                    denoiser_p_std=self.cfg.dlm.denoiser_p_std,
                    denoiser_noise_scale=self.cfg.dlm.denoiser_noise_scale,
                    t_eps=self.cfg.dlm.t_eps,
                    time_schedule=self.cfg.dlm.time_schedule,
                    decoder_prob=0.0,
                    decoder_noise_scale=self.cfg.dlm.decoder_noise_scale,
                    decoder_p_mean=self.cfg.dlm.decoder_p_mean,
                    decoder_p_std=self.cfg.dlm.decoder_p_std,
                    self_cond_prob=self.cfg.dlm.self_cond_prob,
                    self_cond_cfg_min=self.cfg.dlm.self_cond_cfg_min,
                    self_cond_cfg_max=self.cfg.dlm.self_cond_cfg_max,
                    num_self_cond_cfg_tokens=self.cfg.dlm.num_self_cond_cfg_tokens,
                )
        loss = output.loss.detach().expand(batch["input_ids"].shape[0])
        dummy_logits = torch.empty(
            (*batch["input_ids"].shape, 0),
            device=batch["input_ids"].device,
            dtype=loss.dtype,
        )
        evaluator.update_metrics(batch, loss, dummy_logits)
        barrier()
