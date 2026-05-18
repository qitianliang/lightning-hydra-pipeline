from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only

import wandb


class CustomWandbLogger(WandbLogger):
    """A custom WandbLogger that sends a wandb.alert on run completion."""

    def __init__(self, *args, **kwargs):
        """Pass all arguments to the parent WandbLogger."""
        super().__init__(*args, **kwargs)

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
                title = f"✅ Run Finished: {run_name}"
                text = f"Run '{run_name}' completed successfully.\nCheck the results at: {run_url}"
                level = wandb.AlertLevel.INFO
            elif status == "failed":
                title = f"❌ Run Failed: {run_name}"
                text = f"Run '{run_name}' failed.\nPlease check the logs.\nURL: {run_url}"
                level = wandb.AlertLevel.ERROR
            else:  # e.g., 'interrupted'
                title = f"⚠️ Run Interrupted: {run_name}"
                text = f"Run '{run_name}' was interrupted with status '{status}'.\nURL: {run_url}"
                level = wandb.AlertLevel.WARN

            # Send the alert!
            try:
                wandb.alert(title=title, text=text, level=level)
                print(f"\n[CustomWandbLogger] Sent a '{status}' alert for run '{run_name}'.")
            except Exception as e:
                print(f"\n[CustomWandbLogger] Failed to send wandb.alert: {e}")
