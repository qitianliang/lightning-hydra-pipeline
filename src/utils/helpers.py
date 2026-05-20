# src/utils/helpers.py
import re
import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def build_wandb_sweep_url(base_url: str, entity: str, project: str, sweep_id: str) -> str:
    """构建 wandb sweep 页面 URL。"""
    base = base_url.rstrip("/")
    return f"{base}/{entity}/{project}/sweeps/{sweep_id}"


def build_wandb_group_url(base_url: str, entity: str, project: str, group_name: str) -> str:
    """构建 wandb group 页面 URL。"""
    from urllib.parse import quote
    base = base_url.rstrip("/")
    encoded_group = quote(group_name, safe="")
    return f"{base}/{entity}/{project}/groups/{encoded_group}"


def build_wandb_run_url(base_url: str, entity: str, project: str, run_id: str) -> str:
    """构建 wandb run 页面 URL。"""
    base = base_url.rstrip("/")
    return f"{base}/{entity}/{project}/runs/{run_id}/overview"


# ── 配置 key 校验 ──────────────────────────────────────────────────────────

_VALID_KEY_PREFIXES = (
    "model.optimizer.", "model.scheduler.", "model.net.", "model.compile",
    "data.", "trainer.", "logger.", "callbacks.", "seed", "hydra.",
)


def validate_config_keys(keys: list, source: str = "config") -> list:
    """校验配置 key 列表是否为合法 Hydra override key。"""
    valid = []
    for key in keys:
        if "." not in key:
            log.error(f"❌ [{source}] Key '{key}' has no '.' — skipping.")
            continue
        if any(key.startswith(p) or key == p.rstrip(".") for p in _VALID_KEY_PREFIXES):
            valid.append(key)
        else:
            log.warning(f"⚠️ [{source}] Key '{key}' unknown prefix. Keeping.")
            valid.append(key)
    return valid


def format_reproduction_script(base_args: list, overrides: list, seeds: list, group_name: str) -> str:
    """生成单组复现脚本命令。"""
    parts = list(base_args)
    parts.extend(overrides)
    parts.append(f"seed=[{','.join(str(s) for s in seeds)}]")
    parts.append(f"logger.wandb.group={group_name}")
    return " ".join(parts)


# ── Markdown 邮件存档 ──────────────────────────────────────────────────────

def _sanitize_filename(subject: str) -> str:
    """邮件标题 → 安全文件名 (去 emoji/特殊字符, 空格→_)."""
    # 去 emoji 和非 ASCII
    clean = re.sub(r"[^\x20-\x7E]", "", subject)
    # 特殊字符 → _
    clean = re.sub(r"[|\+\s/\\:*?\"<>#]", "_", clean)
    # 合并连续 _
    clean = re.sub(r"_+", "_", clean)
    return clean.strip("_")


def save_email_markdown(subject: str, markdown_content: str, send_success: bool, mode: str = ""):
    """将邮件内容以 Markdown 格式保存到本地。

    目录结构: logs/mail/{mode}/ 和 logs/mail_fail/{mode}/
    文件名: 去掉模式前缀, 格式 {project}_{sid}_{ts}.md

    Args:
        subject: 邮件标题 (用于文件名)
        markdown_content: Markdown 格式邮件内容
        send_success: SMTP 是否发送成功
        mode: 模式名 (eval/ablation/sensitivity/override), 空则存根目录
    """
    safe_name = _sanitize_filename(subject)
    # 去掉模式前缀: "Sweep_Eval_mnist_de5u7x" → "mnist_de5u7x"
    _mode_prefixes = ("Sweep_Eval_", "Eval_", "Ablation_", "Sensitivity_", "Override_")
    for prefix in _mode_prefixes:
        if safe_name.startswith(prefix):
            safe_name = safe_name[len(prefix):]
            break
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{ts}.md"

    # 保存到 logs/mail/{mode}/
    mail_dir = Path("logs/mail") / mode if mode else Path("logs/mail")
    mail_dir.mkdir(parents=True, exist_ok=True)
    out_path = mail_dir / filename
    out_path.write_text(markdown_content, encoding="utf-8")
    log.info(f"📧 Email markdown saved to {out_path}")

    # SMTP 失败 → 额外保存到 logs/mail_fail/{mode}/
    if not send_success:
        fail_dir = Path("logs/mail_fail") / mode if mode else Path("logs/mail_fail")
        fail_dir.mkdir(parents=True, exist_ok=True)
        fail_path = fail_dir / filename
        fail_path.write_text(markdown_content, encoding="utf-8")
        log.warning(f"📧 Email markdown (send failed) also saved to {fail_path}")


# ── 统一 SMTP 发送 + 异常处理 ──────────────────────────────────────────────

def _send_smtp_with_fallback(cfg, msg, subject: str, markdown_content: str, mode: str = ""):
    """SMTP 发送邮件, 失败时保存 markdown 到 logs/mail_fail/{mode}/。

    无论成功失败, 都保存到 logs/mail/{mode}/。
    """
    send_success = False
    try:
        _send_smtp(cfg, msg)
        send_success = True
    except Exception as e:
        log.error(f"❌ Failed to send email: {e}")
        send_success = False

    # 始终保存 markdown
    save_email_markdown(subject, markdown_content, send_success, mode=mode)


def _send_smtp(cfg, msg):
    """内部 SMTP 发送 (代理感知，零全局污染)。

    重构说明:
    原版直接用 smtplib 连接，在 Docker 无 DNS 环境必死。
    现版委托 email_templates.send_email_proxy_aware 处理:
    - 通过 PySocks 劫持 socket → HTTP CONNECT 隧道 → DNS 甩锅到代理
    - SSL 加密在隧道内封装，代理只看到乱码
    - try/finally 保证 socket 恢复，W&B 等组件不受影响
    """
    from src.utils.email_templates import send_email_proxy_aware

    send_email_proxy_aware(cfg, msg)


# ── Eval 邮件 (per-rank 分段展示) ─────────────────────────────────────────

def send_eval_email(
    cfg, sweep_id, sweep_url, sweep_description, mode_label, task_desc,
    rank_data, baseline_info=None,
    report_json="", report_csv="", log_dir="",
):
    """发送 Evaluate 邮件 (per-rank 分段展示) + 保存 Markdown。"""
    import html as html_mod

    if not cfg.enabled:
        log.info("Email notification is disabled. Skipping.")
        return

    # ── 通用 Workflow Details (HTML) ─────────────────────────────────────
    sweep_url_row = ""
    if sweep_url:
        sweep_url_row = f'<tr><th>Sweep URL</th><td><a href="{sweep_url}">{sweep_url}</a></td></tr>'

    workflow_html = f"""
    <hr>
    <h3>📋 Workflow Details</h3>
    <table class="dataframe">
        <tr><th>Sweep ID</th><td><code>{sweep_id}</code></td></tr>
        {sweep_url_row}
        <tr><th>Description</th><td>{sweep_description}</td></tr>
        <tr><th>Report (JSON)</th><td><code>{report_json}</code></td></tr>
        <tr><th>Report (CSV)</th><td><code>{report_csv}</code></td></tr>
        <tr><th>Log Directory</th><td><code>{log_dir}</code></td></tr>
    </table>
    """

    # ── Per-rank 段 (HTML) ───────────────────────────────────────────────
    rank_sections = []
    md_rank_sections = []

    for rd in rank_data:
        meta = rd.get("best_run_metadata", {})
        host = meta.get("host", "N/A")
        created = meta.get("created_at", "N/A")
        duration = meta.get("duration", "N/A")
        rank_label = rd["rank_label"]

        # Rank label: top-1 → "rank 1", top-2 → "rank 2" (通用格式)
        rank_num = int(rank_label.split("-")[1])
        rank_display = f"rank {rank_num}"

        # Metrics
        metric_rows = ""
        md_metrics = []
        for m_name, m_val in rd.get("metrics", {}).items():
            mean_str = f"{m_val['mean']:.4g}" if not pd.isna(m_val['mean']) else "N/A"
            std_str = f"{m_val['std']:.4g}" if not pd.isna(m_val.get('std', float('nan'))) else ""
            metric_rows += f"<tr><td>{m_name}</td><td>{mean_str}</td><td>{std_str}</td></tr>"
            if std_str:
                md_metrics.append(f"- **{m_name}**: {mean_str} ± {std_str}")
            else:
                md_metrics.append(f"- **{m_name}**: {mean_str}")

        # Checkpoint paths
        ckpt_paths = rd.get("checkpoint_paths", [])
        ckpt_rows = ""
        md_ckpt = []
        if ckpt_paths:
            for i, p in enumerate(ckpt_paths):
                ckpt_rows += f'<tr><th>Checkpoint {i+1}</th><td><code>{p}</code></td></tr>'
                md_ckpt.append(f"  {i+1}. `{p}`")
        else:
            ckpt_rows = '<tr><th>Checkpoints</th><td>N/A</td></tr>'

        # Reproduction script
        scr = rd.get("reproduction_script", {})
        cmd_escaped = html_mod.escape(scr.get("command", "N/A"))

        # Run URL
        run_url = rd.get('run_url', 'N/A')

        # HTML section
        run_url_html = f'<a href="{run_url}">{run_url}</a>' if run_url != 'N/A' else 'N/A'
        rank_section = f"""
        <hr>
        <h3>🏆 {rank_label} Eval</h3>
        <table class="dataframe">
            <tr><th>Run</th><td>{rd.get('best_run_name', 'N/A')}</td></tr>
            <tr><th>Run URL</th><td>{run_url_html}</td></tr>
            <tr><th>Host</th><td><code>{host}</code></td></tr>
            <tr><th>Created</th><td>{created}</td></tr>
            <tr><th>Duration</th><td>{duration}</td></tr>
            {ckpt_rows}
        </table>
        <h4>⚙️ Run Config ({rank_display} best)</h4>
        <pre style="background:#f5f5f5; padding:12px; border-radius:4px; font-size:11px; overflow-x:auto;">{html_mod.escape(rd.get('best_run_config', 'N/A'))}</pre>
        <h4>📊 Test Metrics</h4>
        <table class="dataframe">
            <tr><th>Metric</th><th>Mean</th><th>Std</th></tr>
            {metric_rows}
        </table>
        <h4>🔧 Reproduction</h4>
        <table class="dataframe">
            <tr><th>Group</th><th>Command</th></tr>
            <tr><td><b>{scr.get('label', '')}</b></td><td><code>{cmd_escaped}</code></td></tr>
        </table>
        """
        rank_sections.append(rank_section)

        # Markdown section
        md_run_url = f"[{run_url}]({run_url})" if run_url != 'N/A' else 'N/A'
        md_section = f"""
### 🏆 {rank_label} Eval

| Field | Value |
|-------|-------|
| Run | {rd.get('best_run_name', 'N/A')} |
| Run URL | {md_run_url} |
| Host | `{host}` |
| Created | {created} |
| Duration | {duration} |

**Run Config ({rank_display} best):**
```json
{rd.get('best_run_config', 'N/A')}
```

**Test Metrics:**
{chr(10).join(md_metrics)}

**Checkpoints:**
{chr(10).join(md_ckpt) if md_ckpt else '  N/A'}

**Reproduction:**
```
{scr.get('command', 'N/A')}
```
"""
        md_rank_sections.append(md_section)

    # ── Baseline comparison (HTML + MD) ──────────────────────────────────
    baseline_html = ""
    md_baseline = ""
    if baseline_info:
        improvement_pct = baseline_info.get("improvement_pct", 0)
        icon = "📈" if improvement_pct > 0 else "📉" if improvement_pct < 0 else "➡️"
        baseline_html = f"""
        <hr>
        <h3>{icon} Baseline Comparison (top-1)</h3>
        <table class="dataframe">
            <tr><th>Metric</th><td>{baseline_info.get('metric', 'N/A')}</td></tr>
            <tr><th>Baseline</th><td>{baseline_info.get('baseline_value', 'N/A'):.4g}</td></tr>
            <tr><th>Current (mean)</th><td>{baseline_info.get('current_mean', 'N/A'):.4g}</td></tr>
            <tr><th>Improvement</th><td><b>{improvement_pct:+.2f}%</b></td></tr>
        </table>
        """
        md_baseline = f"""
### {icon} Baseline Comparison (top-1)

| Field | Value |
|-------|-------|
| Metric | {baseline_info.get('metric', 'N/A')} |
| Baseline | {baseline_info.get('baseline_value', 'N/A'):.4g} |
| Current (mean) | {baseline_info.get('current_mean', 'N/A'):.4g} |
| Improvement | **{improvement_pct:+.2f}%** |
"""

    # ── 组装 HTML ────────────────────────────────────────────────────────
    styled_html = f"""
    <html><head><style>{_EMAIL_CSS}</style></head><body>
    <h2>✅ {mode_label} | {task_desc}</h2>
    {workflow_html}
    {"".join(rank_sections)}
    {baseline_html}
    </body></html>
    """

    email_subject = f"✅ {mode_label} | {task_desc}"
    if baseline_info:
        email_subject += f" | {baseline_info['metric']} {baseline_info['improvement_pct']:+.2f}%"

    msg = MIMEMultipart()
    msg["Subject"] = email_subject
    msg["From"] = cfg.smtp.username
    msg["To"] = cfg.recipient_email
    msg.attach(MIMEText(styled_html, "html"))

    # ── 组装 Markdown ────────────────────────────────────────────────────
    md_sweep_url = f"[{sweep_url}]({sweep_url})" if sweep_url else "N/A"
    md_content = f"""# ✅ {mode_label} | {task_desc}

## 📋 Workflow Details

| Field | Value |
|-------|-------|
| Sweep ID | `{sweep_id}` |
| Sweep URL | {md_sweep_url} |
| Description | {sweep_description} |
| Report (JSON) | `{report_json}` |
| Report (CSV) | `{report_csv}` |
| Log Directory | `{log_dir}` |

{"".join(md_rank_sections)}
{md_baseline}
"""

    _send_smtp_with_fallback(cfg, msg, email_subject, md_content, mode="eval")


_EMAIL_CSS = """
table.dataframe {
    border-collapse: collapse; border: 1px solid #ddd; width: 100%;
    font-family: Arial, sans-serif; font-size: 12px;
}
table.dataframe th {
    background-color: #f2f2f2; color: #333; font-weight: bold;
    padding: 8px; text-align: left; border: 1px solid #ddd;
}
table.dataframe td { padding: 8px; border: 1px solid #ddd; text-align: left; }
table.dataframe tr:nth-child(even) { background-color: #f9f9f9; }
table.dataframe tr:hover { background-color: #f1f1f1; }
code {
    background-color: #f0f0f0; padding: 2px 6px; border-radius: 3px;
    font-family: 'Courier New', monospace; font-size: 11px;
}
"""


def send_smtp_email(cfg: DictConfig, subject: str, body: str):
    """使用 smtplib 发送邮件通知 (简单文本)。"""
    if not cfg.enabled:
        log.info("Email notification is disabled in the config. Skipping.")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = cfg.smtp.username
    msg["To"] = cfg.recipient_email

    md_content = f"# {subject}\n\n{body}"
    _send_smtp_with_fallback(cfg, msg, subject, md_content)


def send_email_with_dataframe(
    cfg: DictConfig, subject: str, body: str, metric_data: pd.DataFrame,
    baseline_info: dict = None, workflow_info: dict = None,
    reproduction_scripts: list = None, sweep_url: str = None,
    eval_rank: int = 1,
):
    """发送邮件通知 (ablation/sensitivity 用, eval_rank 感知)。"""
    import html as html_mod

    if not cfg.enabled:
        log.info("Email notification is disabled in the config. Skipping.")
        return

    # eval_rank 感知标签 (通用格式: rank N)
    rank_display = f"rank {eval_rank}"
    config_label = f"Run Config ({rank_display} best)"
    metrics_label = f"Run Metrics ({rank_display} best)"

    for col in metric_data.select_dtypes(include=["float64", "int64"]).columns:
        metric_data[col] = metric_data[col].apply(lambda x: f"{x:.4g}" if pd.notnull(x) else "")

    html_table = metric_data.to_html(index=False, escape=False, classes="dataframe", border=0)

    # Baseline
    baseline_html = ""
    md_baseline = ""
    if baseline_info:
        improvement_pct = baseline_info.get("improvement_pct", 0)
        icon = "📈" if improvement_pct > 0 else "📉" if improvement_pct < 0 else "➡️"
        baseline_html = f"""
        <hr>
        <h3>{icon} Baseline Comparison</h3>
        <table class="dataframe">
            <tr><th>Metric</th><td>{baseline_info.get('metric', 'N/A')}</td></tr>
            <tr><th>Baseline</th><td>{baseline_info.get('baseline_value', 'N/A'):.4g}</td></tr>
            <tr><th>Current (mean)</th><td>{baseline_info.get('current_mean', 'N/A'):.4g}</td></tr>
            <tr><th>Improvement</th><td><b>{improvement_pct:+.2f}%</b></td></tr>
        </table>
        """
        md_baseline = f"""
### {icon} Baseline Comparison

| Field | Value |
|-------|-------|
| Metric | {baseline_info.get('metric', 'N/A')} |
| Baseline | {baseline_info.get('baseline_value', 'N/A'):.4g} |
| Current (mean) | {baseline_info.get('current_mean', 'N/A'):.4g} |
| Improvement | **{improvement_pct:+.2f}%** |
"""

    # Workflow info
    workflow_html = ""
    md_workflow = ""
    if workflow_info:
        sweep_id = workflow_info.get("sweep_id", "N/A")
        sweep_desc = workflow_info.get("sweep_description", "N/A")
        run_config = workflow_info.get("best_run_config", "N/A")
        report_json = workflow_info.get("report_json_path", "N/A")
        report_csv = workflow_info.get("report_csv_path", "N/A")
        log_dir_val = workflow_info.get("log_dir", "N/A")

        run_config_escaped = html_mod.escape(str(run_config))

        _sweep_url = sweep_url or workflow_info.get("sweep_url", "")
        sweep_url_row = ""
        md_sweep_url = "N/A"
        if _sweep_url:
            sweep_url_row = f'<tr><th>Sweep URL</th><td><a href="{_sweep_url}">{_sweep_url}</a></td></tr>'
            md_sweep_url = f"[{_sweep_url}]({_sweep_url})"

        run_host = workflow_info.get("best_run_host", "N/A")
        run_created = workflow_info.get("best_run_created_at", "N/A")
        run_duration = workflow_info.get("best_run_duration", "N/A")

        ckpt_paths = workflow_info.get("checkpoint_paths", [])
        ckpt_rows = ""
        md_ckpt = []
        if ckpt_paths:
            for i, p in enumerate(ckpt_paths):
                ckpt_rows += f'<tr><th>Checkpoint {i+1}</th><td><code>{p}</code></td></tr>'
                md_ckpt.append(f"  {i+1}. `{p}`")
        else:
            ckpt_rows = '<tr><th>Checkpoints</th><td>N/A</td></tr>'

        workflow_html = f"""
        <hr>
        <h3>📋 Workflow Details</h3>
        <table class="dataframe">
            <tr><th>Sweep ID</th><td><code>{sweep_id}</code></td></tr>
            {sweep_url_row}
            <tr><th>Description</th><td>{sweep_desc}</td></tr>
            <tr><th>Run Host</th><td><code>{run_host}</code></td></tr>
            <tr><th>Run Created</th><td>{run_created}</td></tr>
            <tr><th>Run Duration</th><td>{run_duration}</td></tr>
            {ckpt_rows}
            <tr><th>Report (JSON)</th><td><code>{report_json}</code></td></tr>
            <tr><th>Report (CSV)</th><td><code>{report_csv}</code></td></tr>
            <tr><th>Log Directory</th><td><code>{log_dir_val}</code></td></tr>
        </table>
        <h4>⚙️ {config_label}</h4>
        <pre style="background:#f5f5f5; padding:12px; border-radius:4px; font-size:11px; overflow-x:auto;">{run_config_escaped}</pre>
        """

        md_workflow = f"""
## 📋 Workflow Details

| Field | Value |
|-------|-------|
| Sweep ID | `{sweep_id}` |
| Sweep URL | {md_sweep_url} |
| Description | {sweep_desc} |
| Run Host | `{run_host}` |
| Run Created | {run_created} |
| Run Duration | {run_duration} |
| Report (JSON) | `{report_json}` |
| Report (CSV) | `{report_csv}` |
| Log Directory | `{log_dir_val}` |

**{config_label}:**
```json
{run_config}
```

**Checkpoints:**
{chr(10).join(md_ckpt) if md_ckpt else '  N/A'}
"""

    # Reproduction scripts
    scripts_html = ""
    md_scripts = ""
    if reproduction_scripts:
        script_rows = []
        md_script_list = []
        for scr in reproduction_scripts:
            label = scr.get("label", "N/A")
            cmd = scr.get("command", "N/A")
            cmd_escaped = html_mod.escape(cmd)
            script_rows.append(f'<tr><td><b>{label}</b></td><td><code>{cmd_escaped}</code></td></tr>')
            md_script_list.append(f"**{label}:**\n```\n{cmd}\n```")

        scripts_html = f"""
        <hr>
        <h3>🔧 Reproduction Scripts</h3>
        <table class="dataframe">
            <tr><th>Group</th><th>Command</th></tr>
            {''.join(script_rows)}
        </table>
        """
        md_scripts = f"""
## 🔧 Reproduction Scripts

{chr(10).join(md_script_list)}
"""

    # 组装 HTML
    styled_html = f"""
    <html><head><style>{_EMAIL_CSS}</style></head><body>
    <p>{body}</p>
        {workflow_html}
        <h3>📊 {metrics_label}</h3>
        {html_table}
        {baseline_html}
        {scripts_html}
    </body></html>
    """

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = cfg.smtp.username
    msg["To"] = cfg.recipient_email
    msg.attach(MIMEText(styled_html, "html"))

    # 组装 Markdown
    md_metrics_table = metric_data.to_markdown(index=False) if hasattr(metric_data, 'to_markdown') else str(metric_data)
    md_content = f"""# {subject}

{body}
{md_workflow}

## 📊 {metrics_label}

{md_metrics_table}
{md_baseline}
{md_scripts}
"""

    _send_smtp_with_fallback(cfg, msg, subject, md_content)