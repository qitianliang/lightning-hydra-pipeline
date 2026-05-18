from typing import Any, Dict

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.trainer.states import TrainerFn  # 导入 TrainerFn 枚举
from lightning.pytorch.utilities import rank_zero_only

import wandb
from src.utils import RankedLogger, send_smtp_email

log = RankedLogger(__name__, rank_zero_only=True)


class SMTPWandbLogger(WandbLogger):
    """A custom WandbLogger that sends a wandb.alert on run completion."""

    def __init__(self, notification, *args, **kwargs):
        """Pass all arguments to the parent WandbLogger."""
        super().__init__(*args, **kwargs)
        self.notification = notification
        self._did_run_test = False
        self._email_sent = False  # 添加一个标志位，防止任何意外的重复发送

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        # 父类方法必须被调用
        super().log_metrics(metrics, step)

        # 检查是否有 'test/' 前缀的指标，这是 Lightning 的默认命名习惯
        # 如果有，就说明 test 阶段已经运行
        for key in metrics.keys():
            if key.startswith("test"):
                self._did_run_test = True
                break

    @rank_zero_only
    def finalize(self, status: str) -> None:
        """This is the key method.

        It's called by Lightning when the run concludes. We'll first call the parent's finalize()
        to ensure proper cleanup, then we'll send our custom alert.
        """
        # 1. First, let the parent logger do its finalization. This is crucial
        # for syncing files, marking the run as finished, etc.
        super().finalize(status)

        # 2. Now, add our custom alert logic.
        # We access the wandb run object via self.experiment
        if self.experiment:
            run_name = self.experiment.name
            run_url = self.experiment.url

            # Customize the alert title and text based on the run's final status
            if status == "success":
                subject = f"✅ Run Finished: {run_name}"
                body = f"Run '{run_name}' completed successfully.\nCheck the results at: {run_url}"
            elif status == "failed":
                subject = f"❌ Run Failed: {run_name}"
                body = f"Run '{run_name}' failed.\nPlease check the logs.\nURL: {run_url}"
            else:  # e.g., 'interrupted'
                subject = f"⚠️ Run Interrupted: {run_name}"
                body = f"Run '{run_name}' was interrupted with status '{status}'.\nURL: {run_url}"

            # =========================================================
            # 邮件通知逻辑
            # =========================================================
            if self._did_run_test and not self._email_sent:
                try:

                    send_smtp_email(self.notification, subject, body)
                    log.info(
                        f"\n[SMTPWandbLogger] Sent a '{status}' email for run '{run_name}'.\n recipient: {self.notification.recipient_email}"
                    )
                except Exception as e:
                    log.info(f"\n[SMTPWandbLogger] Failed to send email: {e}")
