# src/utils/email_templates.py
"""邮件模板模块 — 纯函数, 构建消融/敏感性专用邮件。

重构要点：代理感知 SMTP 发送。
Docker 容器内无外网 DNS，smtplib 原生 socket 直连必死。
通过 PySocks 劫持 socket → HTTP CONNECT 隧道 → DNS 甩锅到代理服务器。
严格 try/finally 保证零全局污染，W&B 等组件的 socket 不受影响。
"""

import html as html_mod
import os
import socket
import smtplib
import ssl
from contextlib import contextmanager
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


# ── 通用 CSS ────────────────────────────────────────────────────────────────
_CSS = """
table.dataframe {
    border-collapse: collapse;
    border: 1px solid #ddd;
    width: 100%;
    font-family: Arial, sans-serif;
    font-size: 12px;
}
table.dataframe th {
    background-color: #f2f2f2;
    color: #333;
    font-weight: bold;
    padding: 8px;
    text-align: left;
    border: 1px solid #ddd;
}
table.dataframe td {
    padding: 8px;
    border: 1px solid #ddd;
    text-align: left;
}
table.dataframe tr:nth-child(even) { background-color: #f9f9f9; }
table.dataframe tr:hover { background-color: #f1f1f1; }
code {
    background-color: #f0f0f0;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 11px;
}
"""


# ── 代理感知 SMTP 连接工厂 ─────────────────────────────────────────────────
# 核心问题：Docker 容器无外网 DNS，smtplib 原生 TCP socket 直连 smtp.xxx.com 必死。
# 解法：PySocks 劫持 socket → HTTP CONNECT 隧道 → DNS 由代理服务器完成。
# 红线约束：绝对不能永久替换 socket.socket，W&B 等组件需要原生 socket。
# 因此用 contextmanager 做临时劫持，try/finally 保证无论成败都恢复原状。


def _parse_proxy_from_env() -> Optional[tuple]:
    """从环境变量解析 HTTP 代理地址，返回 (host, port) 或 None。

    自适应设计：无代理环境变量时返回 None，调用方走直连路径；
    有代理时返回 (host, port)，走 HTTP CONNECT 隧道路径。
    优先级: HTTPS_PROXY > HTTP_PROXY > https_proxy > http_proxy
    仅解析 http:// 或 https:// 前缀的代理 URL (socks5 等交给 PySocks 自己处理)。
    """
    proxy_url = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("http_proxy")
    )
    if not proxy_url or not proxy_url.strip():
        return None

    proxy_url = proxy_url.strip()
    # 去掉 http:// 或 https:// 前缀
    cleaned = proxy_url
    if cleaned.startswith("https://"):
        cleaned = cleaned[len("https://"):]
    elif cleaned.startswith("http://"):
        cleaned = cleaned[len("http://"):]

    # 去掉尾部路径
    cleaned = cleaned.split("/")[0]

    # 解析 host:port
    if ":" in cleaned:
        host, port_str = cleaned.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            log.warning(f"⚠️ 代理端口解析失败: {proxy_url}, 忽略代理配置。")
            return None
        return (host, port)
    else:
        # 只有 host 没有端口，用默认 8080
        log.warning(f"⚠️ 代理 URL 缺少端口: {proxy_url}, 默认使用 8080。")
        return (cleaned, 8080)


@contextmanager
def _proxy_socket_guard():
    """【安全隔离区】自适应代理：有代理时临时劫持 socket，无代理时直接透传。

    自适应逻辑:
    - 无代理环境变量 → yield 后直接返回，socket 保持原生 (本地开发机等有正常 DNS 的环境)
    - 有代理环境变量 → 临时劫持 socket.socket → HTTP CONNECT 隧道 → finally 恢复

    设计考量:
    1. 进入时保存原生 socket.socket → 设置 PySocks 代理 → 替换全局 socket.socket
    2. 退出时 (无论正常还是异常) 在 finally 中恢复原生 socket.socket
    3. 这确保了 W&B / requests / 其他网络组件在邮件发送完毕后不受影响
    4. HTTP CONNECT 隧道使得 DNS 解析发生在代理服务器端，绕过 Docker DNS 黑洞

    用法:
        with _proxy_socket_guard():
            # 有代理: smtplib 走代理隧道; 无代理: smtplib 直连
            ...
        # 此区间外，socket.socket 已恢复原状，W&B 正常工作
    """
    proxy_info = _parse_proxy_from_env()
    if proxy_info is None:
        # ── 自适应: 无代理 → 直连模式 ────────────────────────────────────
        # 本地开发机、有正常 DNS 的服务器等环境，无需任何 socket 劫持
        log.info("📧 SMTP 自适应模式: 直连 (未检测到代理环境变量)")
        yield
        return

    proxy_host, proxy_port = proxy_info

    try:
        import socks
    except ImportError:
        log.warning(
            "⚠️ 检测到代理环境变量但 PySocks 未安装！"
            "将回退到直连模式 (Docker 内可能失败)。请运行: pip install PySocks"
        )
        yield
        return

    # ── 保存原生 socket (安全隔离区起点) ──────────────────────────────────
    original_socket = socket.socket
    original_getaddrinfo = socket.getaddrinfo

    def _fake_getaddrinfo(host, port, *args, **kwargs):
        # 透传域名/地址，不做本地 DNS 解析。
        # socksocket 会在代理隧道内通过 HTTP CONNECT 发送域名，由代理解析。
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '', (host, port))]

    try:
        # ── 自适应: 有代理 → 隧道模式 ─────────────────────────────────────
        # 配置 PySocks: HTTP 类型代理，使用 HTTP CONNECT 方法建隧道
        # 关键：socks.HTTP 会让 PySocks 发送 "CONNECT smtp.xxx.com:465 HTTP/1.1"
        # 代理服务器在远端完成 DNS 解析和 TCP 连接，本地只需连 127.0.0.1:7899
        socks.set_default_proxy(socks.HTTP, proxy_host, proxy_port)

        # 劫持！所有新的 socket.socket() 调用将返回 socksocket 实例
        socket.socket = socks.socksocket
        # 必须同时劫持 getaddrinfo，否则 smtplib 在 create_connection 时本地 DNS 必挂
        socket.getaddrinfo = _fake_getaddrinfo

        log.info(f"📧 SMTP 自适应模式: 代理隧道 → {proxy_host}:{proxy_port} (DNS 由代理解析)")
        yield
    finally:
        # ── 恢复原生 socket (安全隔离区终点) ──────────────────────────────
        # 无论成功、异常、KeyboardInterrupt，都必须执行此恢复
        # 这是零全局污染的核心保障
        socket.socket = original_socket
        socket.getaddrinfo = original_getaddrinfo
        socks.set_default_proxy()  # 清除 PySocks 全局代理状态
        log.debug("🔒 socket 已恢复原生，代理劫持解除。")


def _create_smtp_connection(cfg) -> smtplib.SMTP:
    """代理感知 SMTP 连接工厂。

    流程:
    1. _proxy_socket_guard() 劫持 socket → HTTP CONNECT 隧道
    2. smtplib 在被劫持的 socket 上连接 SMTP 服务器
       - DNS 解析由代理服务器完成 (Docker DNS 黑洞被绕过)
    3. ssl.create_default_context() 包装加密层
       - 代理服务器只能看到加密乱码，账密安全
    4. 无论成败，_proxy_socket_guard 的 finally 块恢复原生 socket

    Args:
        cfg: DictConfig, 需包含 cfg.smtp.host/port/use_ssl/username/password

    Returns:
        已连接并认证的 smtplib.SMTP 实例
    """
    context = ssl.create_default_context()

    if cfg.smtp.use_ssl:
        # ── SMTP_SSL: 全程加密 (端口 465) ────────────────────────────────
        # 在代理隧道内建立 SSL 连接，代理服务器只看到加密字节流
        server = smtplib.SMTP_SSL(
            cfg.smtp.host, cfg.smtp.port, context=context, timeout=15
        )
    else:
        # ── STARTTLS: 先明文后升级加密 (端口 587/25) ────────────────────
        # 在代理隧道内先建明文连接，再通过 STARTTLS 升级为加密
        server = smtplib.SMTP(cfg.smtp.host, cfg.smtp.port, timeout=15)
        server.ehlo()  # 必须先 ehlo 才能 starttls
        server.starttls(context=context)
        server.ehlo()  # 加密后再 ehlo 一次

    server.login(cfg.smtp.username, cfg.smtp.password)
    return server


def send_email_proxy_aware(cfg, msg) -> None:
    """自适应代理的邮件发送入口 — 无代理直连，有代理走隧道。

    自适应调度逻辑:
    1. _proxy_socket_guard 检测环境变量:
       - 无代理 → 直连模式 (socket 保持原生，DNS 在本地解析)
       - 有代理 → 隧道模式 (临时劫持 socket，DNS 由代理服务器解析)
    2. _create_smtp_connection 建立加密 SMTP 连接 (SMTP_SSL 或 STARTTLS)
    3. 发送邮件
    4. _proxy_socket_guard 的 finally 保证: 有代理时恢复原生 socket (零全局污染)

    无论哪种模式，即使发送过程中抛出任何异常:
    - 有代理时: socket 一定在 finally 中恢复，不影响 W&B 等组件
    - 无代理时: 本就未劫持，无恢复需求

    使用场景:
    - 本地开发机 (有 DNS): 不设代理环境变量 → 自动走直连
    - Docker 容器 (无 DNS): 设 HTTP_PROXY=http://127.0.0.1:7899 → 自动走隧道
    无需修改任何代码。
    """
    with _proxy_socket_guard():
        with _create_smtp_connection(cfg) as server:
            server.send_message(msg)

    proxy_info = _parse_proxy_from_env()
    mode_label = "代理隧道" if proxy_info else "直连"
    log.info(f"✅ Email sent successfully ({mode_label}) to {cfg.recipient_email}.")


def _build_workflow_html(workflow_info: dict, eval_rank: int = 1) -> str:
    """通用 workflow 信息段 (eval_rank 感知)。"""
    rank_display = f"rank {eval_rank}"
    config_label = f"Run Config ({rank_display} best)"

    sweep_id = workflow_info.get("sweep_id", "N/A")
    sweep_desc = workflow_info.get("sweep_description", "N/A")
    best_config = html_mod.escape(str(workflow_info.get("best_run_config", "N/A")))
    report_json = workflow_info.get("report_json_path", "N/A")
    report_csv = workflow_info.get("report_csv_path", "N/A")
    log_dir = workflow_info.get("log_dir", "N/A")
    sweep_url = workflow_info.get("sweep_url", "")

    sweep_url_row = ""
    if sweep_url:
        sweep_url_row = f'<tr><th>Sweep URL</th><td><a href="{sweep_url}">{sweep_url}</a></td></tr>'

    # Best Run 元信息
    best_run_host = workflow_info.get("best_run_host", "N/A")
    run_url = workflow_info.get("run_url", "N/A")
    best_run_created = workflow_info.get("best_run_created_at", "N/A")
    best_run_duration = workflow_info.get("best_run_duration", "N/A")

    # Checkpoint 路径
    ckpt_paths = workflow_info.get("checkpoint_paths", [])
    ckpt_rows = ""
    if ckpt_paths:
        for i, p in enumerate(ckpt_paths):
            ckpt_rows += f'<tr><th>Checkpoint {i+1}</th><td><code>{p}</code></td></tr>'
    else:
        ckpt_rows = '<tr><th>Checkpoints</th><td>N/A</td></tr>'

    return f"""
    <hr>
    <h3>📋 Workflow Details</h3>
    <table class="dataframe">
        <tr><th>Sweep ID</th><td><code>{sweep_id}</code></td></tr>
        {sweep_url_row}
        <tr><th>Description</th><td>{sweep_desc}</td></tr>
        <tr><th>Run URL</th><td><a href="{run_url}">{run_url}</a></td></tr>
        <tr><th>Run Host</th><td><code>{best_run_host}</code></td></tr>
        <tr><th>Run Created</th><td>{best_run_created}</td></tr>
        <tr><th>Run Duration</th><td>{best_run_duration}</td></tr>
        {ckpt_rows}
        <tr><th>Report (JSON)</th><td><code>{report_json}</code></td></tr>
        <tr><th>Report (CSV)</th><td><code>{report_csv}</code></td></tr>
        <tr><th>Log Directory</th><td><code>{log_dir}</code></td></tr>
    </table>
    <h4>⚙️ {config_label}</h4>
    <pre style="background:#f5f5f5; padding:12px; border-radius:4px; font-size:11px; overflow-x:auto;">{best_config}</pre>
    """


def _build_reproduction_scripts_html(reproduction_scripts: list) -> str:
    """构建复现脚本 HTML 段。"""
    if not reproduction_scripts:
        return ""
    rows = []
    for scr in reproduction_scripts:
        label = html_mod.escape(scr.get("label", "N/A"))
        cmd = html_mod.escape(scr.get("command", "N/A"))
        rows.append(f'<tr><td><b>{label}</b></td><td><code>{cmd}</code></td></tr>')
    return f"""
    <hr>
    <h3>🔧 Reproduction Scripts</h3>
    <table class="dataframe">
        <tr><th>Group</th><th>Command</th></tr>
        {''.join(rows)}
    </table>
    """


def _build_group_urls_html(group_urls: dict) -> str:
    """构建 wandb group URLs HTML 段。"""
    if not group_urls:
        return ""
    rows = []
    for name, url in group_urls.items():
        rows.append(f'<tr><td><b>{html_mod.escape(name)}</b></td><td><a href="{url}">{url}</a></td></tr>')
    return f"""
    <hr>
    <h3>🔗 W&B Group Pages</h3>
    <table class="dataframe">
        <tr><th>Group</th><th>URL</th></tr>
        {''.join(rows)}
    </table>
    """


# ── 消融实验邮件 ────────────────────────────────────────────────────────────
def build_ablation_email(
    notification_cfg: DictConfig,
    subject: str,
    full_model_metrics: dict,
    ablation_results: List[dict],
    workflow_info: dict,
    reproduction_scripts: list = None,
    group_urls: dict = None,
    eval_rank: int = 1,
) -> MIMEMultipart:
    """构建消融实验对比邮件。

    Args:
        full_model_metrics: {"test/acc": {"mean": 0.95, "std": 0.01}, ...}
        ablation_results: [{"name": "no_attn", "metrics": {"test/acc": {"mean": 0.90, "std": 0.02}}, ...}]
        workflow_info: 通用 workflow 信息字典
        reproduction_scripts: [{"label": "no_attn", "command": "python ..."}, ...]
        group_urls: {"no_attn": "http://ip:port/entity/project/groups/group_name", ...}
    """
    # 构建对比表
    rows = []
    for metric_name, vals in full_model_metrics.items():
        row = {"Metric": metric_name, "Full Model": f"{vals['mean']:.4g} ± {vals['std']:.4g}"}
        for abl in ablation_results:
            abl_val = abl["metrics"].get(metric_name, {})
            if abl_val:
                row[abl["name"]] = f"{abl_val['mean']:.4g} ± {abl_val['std']:.4g}"
            else:
                row[abl["name"]] = "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    html_table = df.to_html(index=False, escape=False, classes="dataframe", border=0)

    # 计算 relative drop
    drop_rows = []
    primary_metric = list(full_model_metrics.keys())[0] if full_model_metrics else None
    if primary_metric:
        full_mean = full_model_metrics[primary_metric]["mean"]
        for abl in ablation_results:
            abl_mean = abl["metrics"].get(primary_metric, {}).get("mean", 0)
            drop = full_mean - abl_mean
            drop_pct = (drop / full_mean * 100) if full_mean != 0 else 0
            icon = "📉" if drop > 0 else "📈" if drop < 0 else "➡️"
            drop_rows.append(f"<tr><td>{abl['name']}</td><td>{icon} {drop_pct:+.2f}%</td></tr>")

    drop_html = ""
    if drop_rows:
        drop_html = f"""
        <hr>
        <h3>📉 Relative Drop vs Full Model (metric: {primary_metric})</h3>
        <table class="dataframe">
            <tr><th>Ablation</th><th>Drop</th></tr>
            {''.join(drop_rows)}
        </table>
        """

    workflow_html = _build_workflow_html(workflow_info, eval_rank=eval_rank)
    scripts_html = _build_reproduction_scripts_html(reproduction_scripts or [])
    group_urls_html = _build_group_urls_html(group_urls or {})

    # eval_rank 感知标题
    header_label = f"Based on rank {eval_rank} Best Run"

    styled_html = f"""
    <html><head><style>{_CSS}</style></head><body>
    <h2>🧪 Ablation Study Results</h2>
    <p><em>{header_label}</em></p>
    {workflow_html}
    <h3>📊 Metrics Comparison</h3>
    {html_table}
    {drop_html}
    {scripts_html}
    {group_urls_html}
    </body></html>
    """

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = notification_cfg.smtp.username
    msg["To"] = notification_cfg.recipient_email
    msg.attach(MIMEText(styled_html, "html"))
    return msg


# ── 参数敏感性邮件 ──────────────────────────────────────────────────────────
def build_sensitivity_email(
    notification_cfg: DictConfig,
    subject: str,
    best_params_metrics: dict,
    param_grid_results: List[dict],
    image_paths: List[tuple] = None,
    image_png_path: Path = None,
    image_pdf_path: Path = None,
    workflow_info: dict = None,
    reproduction_scripts: list = None,
    group_urls: dict = None,
    eval_rank: int = 1,
) -> MIMEMultipart:
    """构建参数敏感性邮件 (支持多图嵌入 + 多 PDF 附件)。

    Args:
        best_params_metrics: {"test/acc": {"mean": 0.95, "std": 0.01}, ...}
        param_grid_results: 每个参数组合的结果列表
        image_paths: 多图列表 [(png_path, pdf_path), ...], 优先使用
        image_png_path: 单图 (向后兼容)
        image_pdf_path: 单PDF (向后兼容)
        workflow_info: 通用 workflow 信息字典
        reproduction_scripts: [{"label": "lr_sensitivity_lr=0.001", "command": "python ..."}, ...]
        group_urls: {"lr_sensitivity_lr=0.001": "http://ip:port/.../groups/group_name", ...}
    """
    # 向后兼容: 单图 → 包装为列表
    if image_paths is None:
        image_paths = []
        if image_png_path and image_pdf_path:
            image_paths.append((image_png_path, image_pdf_path))

    # 最优参数指标表
    rows = []
    for metric_name, vals in best_params_metrics.items():
        rows.append({"Metric": metric_name, "Mean ± Std": f"{vals['mean']:.4g} ± {vals['std']:.4g}"})
    best_df = pd.DataFrame(rows)
    best_html = best_df.to_html(index=False, escape=False, classes="dataframe", border=0)

    # 参数网格结果摘要表
    grid_rows = []
    for res in param_grid_results:
        row = {"Params": res.get("param_desc", "N/A")}
        for m, v in res.get("metrics", {}).items():
            row[m] = f"{v['mean']:.4g} ± {v['std']:.4g}"
        grid_rows.append(row)
    grid_df = pd.DataFrame(grid_rows)
    grid_html = grid_df.to_html(index=False, escape=False, classes="dataframe", border=0)

    workflow_html = _build_workflow_html(workflow_info, eval_rank=eval_rank) if workflow_info else ""

    # eval_rank 感知标签
    metrics_label = f"Run Metrics (rank {eval_rank} best)"

    # 多图嵌入
    img_tags = []
    for idx, img_entry in enumerate(image_paths):
        # 兼容 2-tuple (png, pdf) 和 3-tuple (png, pdf, group_name)
        if len(img_entry) == 3:
            png_path, pdf_path, plot_group_name = img_entry
        else:
            png_path, pdf_path = img_entry
            plot_group_name = Path(png_path).stem if png_path else f"Study {idx+1}"

        cid = f"sensitivity_plot_{idx}"
        study_label = html_mod.escape(plot_group_name)
        img_tags.append(
            f'<h4>📈 {study_label}</h4>'
            f'<img src="cid:{cid}" style="max-width:100%;">'
            f'<p>PDF: <code>{pdf_path}</code></p>'
        )

    plots_section = ""
    if img_tags:
        plots_section = f'<h3>📈 Sensitivity Plots</h3>{"".join(img_tags)}'

    scripts_html = _build_reproduction_scripts_html(reproduction_scripts or [])
    group_urls_html = _build_group_urls_html(group_urls or {})

    styled_html = f"""
    <html><head><style>{_CSS}</style></head><body>
    <h2>📐 Sensitivity Analysis Results</h2>
    {workflow_html}
    <h3>📊 {metrics_label}</h3>
    {best_html}
    <h3>📊 Parameter Grid Results</h3>
    {grid_html}
    {plots_section}
    {scripts_html}
    {group_urls_html}
    </body></html>
    """

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = notification_cfg.smtp.username
    msg["To"] = notification_cfg.recipient_email
    msg.attach(MIMEText(styled_html, "html"))

    # 嵌入所有图片 + PDF 附件
    for idx, img_entry in enumerate(image_paths):
        if len(img_entry) == 3:
            png_path, pdf_path, _ = img_entry
        else:
            png_path, pdf_path = img_entry
        cid = f"sensitivity_plot_{idx}"

        # 嵌入 PNG 图片
        if Path(png_path).exists():
            with open(png_path, "rb") as f:
                img = MIMEImage(f.read())
                img.add_header("Content-ID", f"<{cid}>")
                img.add_header("Content-Disposition", "inline", filename=Path(png_path).name)
                msg.attach(img)

        # PDF 附件
        if Path(pdf_path).exists():
            from email.mime.base import MIMEBase
            from email import encoders
            with open(pdf_path, "rb") as f:
                pdf_part = MIMEBase("application", "pdf")
                pdf_part.set_payload(f.read())
                encoders.encode_base64(pdf_part)
                pdf_part.add_header(
                    "Content-Disposition", "attachment", filename=Path(pdf_path).name
                )
                msg.attach(pdf_part)

    return msg


def send_email_with_mimemultipart(notification_cfg: DictConfig, msg: MIMEMultipart, subject: str = "", md_content: str = "", mode: str = ""):
    """发送已构建好的 MIMEMultipart 邮件 + 保存 Markdown。"""
    from src.utils.helpers import _send_smtp_with_fallback

    if not notification_cfg.enabled:
        log.info("Email notification disabled. Skipping.")
        return

    # 从 msg 提取 subject (如果未提供)
    if not subject:
        subject = str(msg.get("Subject", "workflow_notification"))

    # 未提供 md_content → 从 HTML body 自动提取文本
    if not md_content:
        md_content = _html_to_markdown(msg)

    _send_smtp_with_fallback(notification_cfg, msg, subject, md_content, mode=mode)


def _html_to_markdown(msg: MIMEMultipart) -> str:
    """从 MIMEMultipart HTML body 提取纯文本, 转成可读 markdown.

    - 表格 → pipe-delimited 格式
    - 标题 → # 前缀
    - 链接 → [text](url)
    - 代码 → 反引号
    - CSS/style/注释全清除
    - 退化: subject 行
    """
    import re

    html = ""
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            payload = part.get_payload(decode=True)
            if payload:
                html = payload.decode("utf-8", errors="replace")
            break
    if not html:
        return f"# {msg.get('Subject', 'workflow_notification')}"

    # 清除 style / 注释
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    # 换行标签
    html = re.sub(r"<br\s*/?>", "\n", html)
    html = re.sub(r"</p>", "\n", html)
    html = re.sub(r"</tr>", "\n", html)
    html = re.sub(r"</th>", " | ", html)
    html = re.sub(r"</td>", " | ", html)
    html = re.sub(r"<th[^>]*>", "", html)
    html = re.sub(r"<td[^>]*>", "", html)

    # 标题
    html = re.sub(r"<h1[^>]*>", "\n# ", html)
    html = re.sub(r"</h1>", "\n", html)
    html = re.sub(r"<h2[^>]*>", "\n## ", html)
    html = re.sub(r"</h2>", "\n", html)
    html = re.sub(r"<h3[^>]*>", "\n### ", html)
    html = re.sub(r"</h3>", "\n", html)
    html = re.sub(r"<h4[^>]*>", "\n#### ", html)
    html = re.sub(r"</h4>", "\n", html)

    # 行内元素
    html = re.sub(r"<code[^>]*>", " `", html)
    html = re.sub(r"</code>", "` ", html)
    html = re.sub(r'<a\s+href="([^"]+)"[^>]*>(.*?)</a>', r'[\2](\1)', html)
    html = re.sub(r"</?strong[^>]*>", "**", html)
    html = re.sub(r"</?b[^>]*>", "**", html)
    html = re.sub(r"</?em[^>]*>", "*", html)

    # 清除残差标签
    html = re.sub(r"<[^>]+>", "", html)
    html = html_mod.unescape(html)

    # 折叠多余空行
    html = re.sub(r"\n{3,}", "\n\n", html)
    lines = [line.strip() for line in html.split("\n")]
    html = "\n".join(lines).strip()

    return html or f"# {msg.get('Subject', 'workflow_notification')}"
