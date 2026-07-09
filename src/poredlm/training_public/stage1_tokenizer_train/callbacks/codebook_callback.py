from transformers import TrainerCallback
import wandb

from transformers import TrainerCallback
import wandb

class CodebookCallback(TrainerCallback):
    def __init__(self, monitor, target_layer):
        self.monitor = monitor
        # 挂载 Hook：自动拦截量化层输出（保持你原有的逻辑）
        target_layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # output 假设是 (quantized, indices)
        self.monitor.update(output[1])

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """👈 新增：训练开始时，动态拦截最外层的 PoreRSQWrapper"""
        if model is not None:
            # 挂载第二个 Hook，专门用来抓取 Wrapper 返回的 {"loss": loss, "recon": recon}
            model.register_forward_hook(self._wrapper_hook)

    def _wrapper_hook(self, module, input, output):
        """👈 新增：抓取 Wrapper forward 出来的 loss 并喂给 monitor"""
        if isinstance(output, dict) and "loss" in output:
            self.monitor.update_loss(output["loss"])

    def on_step_end(self, args, state, control, **kwargs):
        """👈 新增：每 10 个 step 结算并上传一次平均 Loss"""
        if state.global_step > 0 and state.global_step % 10 == 0:
            # 🚨 注意：此函数必须所有卡同时调用，否则 DDP 会死锁
            global_avg_loss = self.monitor.get_and_reset_loss_average()

            # 🚨 只有主进程（Rank 0）被允许写入 Wandb，防止多卡写冲突
            if state.is_world_process_zero and global_avg_loss is not None:
                wandb.log(
                    {"train/recon_loss_smooth10": global_avg_loss}, 
                    step=state.global_step
                )

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = self.monitor.get_metrics()
        # 规范化：也让评估期的糊弄指标只在 rank0 上传
        if state.is_world_process_zero:
            wandb.log(metrics, step=state.global_step)
        self.monitor.reset()
