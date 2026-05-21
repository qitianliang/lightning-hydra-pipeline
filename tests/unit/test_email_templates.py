import math
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from omegaconf import OmegaConf

from src.utils.email_templates import _html_to_markdown, build_ablation_email, build_sensitivity_email


def _notification_cfg():
    return OmegaConf.create(
        {
            "smtp": {"username": "sender@example.com"},
            "recipient_email": "recipient@example.com",
        }
    )


def _html(msg: MIMEMultipart) -> str:
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True)
            return payload.decode("utf-8")
    raise AssertionError("message has no HTML part")


def test_ablation_email_uses_report_directory_without_csv_row():
    msg = build_ablation_email(
        notification_cfg=_notification_cfg(),
        subject="Ablation debug",
        full_model_metrics={"test/acc": {"mean": 0.95, "std": 0.01}},
        ablation_results=[
            {"name": "no_lin1_bn", "metrics": {"test/acc": {"mean": 0.9, "std": 0.02}}}
        ],
        workflow_info={
            "sweep_id": "test_sweep",
            "sweep_description": "debug sweep",
            "best_run_config": '{"model.optimizer.lr": "0.01"}',
            "report_label": "Ablation Report Directory",
            "report_json_path": "/tmp/reports/ablation_test_sweep",
            "report_csv_path": "N/A",
            "log_dir": "/tmp/reports/ablation_test_sweep",
            "run_url": "N/A",
        },
        reproduction_scripts=[],
        group_urls={},
    )

    html = _html(msg)
    assert "Ablation Report Directory" in html
    assert "/tmp/reports/ablation_test_sweep" in html
    assert "Report (CSV)" not in html


def test_sensitivity_email_uses_sensitivity_report_json_without_csv_row():
    msg = build_sensitivity_email(
        notification_cfg=_notification_cfg(),
        subject="Sensitivity debug",
        best_params_metrics={"test/acc": {"mean": 0.95, "std": 0.01}},
        param_grid_results=[
            {"param_desc": "model.optimizer.lr=0.001", "metrics": {"test/acc": {"mean": 0.94, "std": 0.02}}}
        ],
        image_paths=[],
        workflow_info={
            "sweep_id": "test_sweep",
            "sweep_description": "debug sweep",
            "best_run_config": '{"model.optimizer.lr": "0.01"}',
            "report_label": "Sensitivity Report (JSON)",
            "report_json_path": "/tmp/reports/sensitivity_test_sweep.json",
            "report_csv_path": "N/A",
            "log_dir": "/tmp/reports",
            "run_url": "N/A",
        },
        reproduction_scripts=[],
        group_urls={},
    )

    html = _html(msg)
    assert "Sensitivity Report (JSON)" in html
    assert "/tmp/reports/sensitivity_test_sweep.json" in html
    assert "Report (CSV)" not in html


def test_email_templates_do_not_render_nan_std_or_body_as_bold_marker():
    msg = build_ablation_email(
        notification_cfg=_notification_cfg(),
        subject="Ablation single seed",
        full_model_metrics={"test/acc": {"mean": 0.95, "std": math.nan}},
        ablation_results=[
            {"name": "no_lin1_bn", "metrics": {"test/acc": {"mean": 0.9, "std": math.nan}}}
        ],
        workflow_info={
            "sweep_id": "test_sweep",
            "sweep_description": "debug sweep",
            "best_run_config": "{}",
            "report_label": "Ablation Report Directory",
            "report_json_path": "/tmp/reports/ablation_test_sweep",
            "report_csv_path": "N/A",
            "log_dir": "/tmp/reports/ablation_test_sweep",
            "run_url": "N/A",
        },
        reproduction_scripts=[],
        group_urls={},
    )

    html = _html(msg)
    assert "nan" not in html.lower()

    markdown_msg = MIMEMultipart()
    markdown_msg.attach(MIMEText("<html><body><h2>Title</h2><p><b>bold</b></p></body></html>", "html"))
    markdown = _html_to_markdown(markdown_msg)
    assert markdown.startswith("## Title")
    assert not markdown.startswith("**")
    assert "**bold**" in markdown
