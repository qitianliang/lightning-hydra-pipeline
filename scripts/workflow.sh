#!/usr/bin/env bash
# ============================================================================
# Workflow Runner Script — 固化多种运行模式
# ============================================================================
# Usage:
#   bash scripts/workflow.sh <mode> [sweep_id] [extra_args...]
#
# Modes:
#   (default)    — 全流程: sweep + pipeline (Stage 1→4)
#   sweep        — 仅超参搜索 (Stage 1)
#   agent        — 开启 W&B agent 在隔离沙盒中运行 (需 sweep_id)
#   evaluate     — 仅多种子评估 (Stage 2, 需 sweep_id)
#   pipeline     — 评估+消融+敏感性 (Stage 2→4, 需 sweep_id)
#   ablation     — 消融实验模式 (需 sweep_id, 自动获取 evaluate 结果)
#   sensitivity  — 参数敏感性模式 (需 sweep_id, 1D/2D 绘图+PDF)
#   dry-run      — 预览模式: 打印命令不执行 [sweep_id 可选]
#   notify       — 启用邮件通知运行 [sweep_id 可选]
# ============================================================================

set -euo pipefail

# ── 项目根目录 ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── 环境变量 ────────────────────────────────────────────────────────────────
export PYENV="${PYENV:-cu129}"
export TIMEOUT_SECS="${TIMEOUT_SECS:-300}"  # 5min 默认超时; 0=无限制

# ── 激活 mamba/conda 环境 ──────────────────────────────────────────────────
if [[ -n "${MAMBA_ROOT_PREFIX:-}" ]]; then
    MAMBA_BASE="${MAMBA_ROOT_PREFIX}"
else
    MAMBA_BASE="$(conda info --base 2>/dev/null || echo "")"
fi

if [[ -n "${MAMBA_BASE}" ]] && [[ -f "${MAMBA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${MAMBA_BASE}/etc/profile.d/conda.sh"
    conda activate "${PYENV}"
elif [[ -n "${MAMBA_BASE}" ]] && [[ -f "${MAMBA_BASE}/etc/profile.d/mamba.sh" ]]; then
    source "${MAMBA_BASE}/etc/profile.d/mamba.sh"
    mamba activate "${PYENV}" 2>/dev/null || conda activate "${PYENV}"
else
    echo "⚠️  conda init not found at ${MAMBA_BASE:-<empty>}, trying shell hook..."
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate "${PYENV}" \
        || { echo "❌ Failed to activate ${PYENV}"; exit 1; }
fi

echo "✅ Active env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "   Python: $(which python)"
echo "   Project: ${PROJECT_ROOT}"

# ── 加载 .env ───────────────────────────────────────────────────────────────
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    echo "📋 Loading .env..."
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# 若未显式设置 DEVICES，从 CUDA_VISIBLE_DEVICES 自动推断
#   CUDA_VISIBLE_DEVICES=0      → DEVICES=[0]
#   CUDA_VISIBLE_DEVICES=0,1,2  → DEVICES=[0,1,2]
if [[ -z "${DEVICES:-}" ]] && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    DEVICES="[$(echo "${CUDA_VISIBLE_DEVICES}" | tr -d ' ')]"
fi
export DEVICES="${DEVICES:-[0]}"

# ── wandb 登录检查 ──────────────────────────────────────────────────────────
if [[ -n "${WANDB_API_KEY:-}" ]] && [[ -n "${WANDB_BASE_URL:-}" ]]; then
    wandb login "${WANDB_API_KEY}" 2>/dev/null || true
fi

# ── 模式解析 ─────────────────────────────────────────────────────────────────
MODE="${1:-pipeline}"
SWEEP_ID=""

# Shift consumed args safely. Modes such as sweep do not accept a sweep_id, so
# every remaining token is treated as a Hydra override.
if [[ $# -ge 1 ]]; then
    shift 1
fi

case "${MODE}" in
    sweep)
        SWEEP_ID=""
        ;;
    evaluate|ablation|sensitivity|agent)
        SWEEP_ID="${1:-}"
        if [[ $# -ge 1 ]]; then
            shift 1
        fi
        ;;
    pipeline|notify|dry-run)
        if [[ $# -ge 1 && "${1}" != workflow.* && "${1}" != hydra.* && "${1}" != paths.* ]]; then
            SWEEP_ID="$1"
            shift 1
        fi
        ;;
    *)
        if [[ $# -ge 1 ]]; then
            SWEEP_ID="$1"
            shift 1
        fi
        ;;
esac

EXTRA_ARGS="${*:-}"

# 公共参数
BASE_CMD="python src/workflow.py"
COMMON_OPTS="workflow.sweep_task.conda_env=${PYENV} workflow.devices=${DEVICES}"

# TIMEOUT_SECS=0 → disable all Python-level timeouts too (evaluate/ablation/sensitivity)
if [[ "${TIMEOUT_SECS}" == "0" ]]; then
    TIMEOUT_OPTS="workflow.ablation_task.timeout_secs=0 workflow.evaluate_task.timeout_secs=0 workflow.sensitivity_task.timeout_secs=0"
else
    TIMEOUT_OPTS=""
fi

build_cmd() {
    local mode="$1"
    local sweep_id="$2"
    local extra="$3"

    case "${mode}" in
        pipeline)
            if [[ -n "${sweep_id}" ]]; then
                echo "${BASE_CMD} workflow.target_sweep_id=${sweep_id} ${COMMON_OPTS} ${TIMEOUT_OPTS} ${extra}"
            else
                echo "${BASE_CMD} ${COMMON_OPTS} ${TIMEOUT_OPTS} ${extra}"
            fi
            ;;
        sweep)
            echo "${BASE_CMD} workflow.task_name=sweep ${COMMON_OPTS} ${TIMEOUT_OPTS} ${extra}"
            ;;
        evaluate)
            if [[ -z "${sweep_id}" ]]; then
                echo "❌ evaluate 模式需要 sweep_id 参数" >&2
                echo "   Usage: bash scripts/workflow.sh evaluate <sweep_id>" >&2
                exit 1
            fi
            echo "${BASE_CMD} workflow.target_sweep_id=${sweep_id} ${COMMON_OPTS} ${TIMEOUT_OPTS} ${extra}"
            ;;
        agent)
            if [[ -z "${sweep_id}" ]]; then
                echo "❌ agent 模式需要 sweep_id 参数" >&2
                echo "   Usage: bash scripts/workflow.sh agent <sweep_id>" >&2
                exit 1
            fi
            echo "${BASE_CMD} workflow.task_name=agent workflow.target_sweep_id=${sweep_id} ${COMMON_OPTS} ${TIMEOUT_OPTS} ${extra}"
            ;;
        ablation)
            if [[ -z "${sweep_id}" ]]; then
                echo "❌ ablation 模式需要 sweep_id 参数" >&2
                echo "   Usage: bash scripts/workflow.sh ablation <sweep_id>" >&2
                exit 1
            fi
            echo "${BASE_CMD} workflow=ablation workflow.target_sweep_id=${sweep_id} ${COMMON_OPTS} ${TIMEOUT_OPTS} ${extra}"
            ;;
        sensitivity)
            if [[ -z "${sweep_id}" ]]; then
                echo "❌ sensitivity 模式需要 sweep_id 参数" >&2
                echo "   Usage: bash scripts/workflow.sh sensitivity <sweep_id>" >&2
                exit 1
            fi
            echo "${BASE_CMD} workflow=sensitivity workflow.target_sweep_id=${sweep_id} ${COMMON_OPTS} ${TIMEOUT_OPTS} ${extra}"
            ;;
        dry-run)
            local cmd="${BASE_CMD} workflow.dry_run=true ${COMMON_OPTS} ${TIMEOUT_OPTS} ${extra}"
            if [[ -n "${sweep_id}" ]]; then
                cmd="${BASE_CMD} workflow.dry_run=true workflow.target_sweep_id=${sweep_id} ${COMMON_OPTS} ${TIMEOUT_OPTS} ${extra}"
            fi
            echo "${cmd}"
            ;;
        notify)
            local cmd="${BASE_CMD} workflow.notification.enabled=true ${COMMON_OPTS} ${TIMEOUT_OPTS} ${extra}"
            if [[ -n "${sweep_id}" ]]; then
                cmd="${BASE_CMD} workflow.target_sweep_id=${sweep_id} workflow.notification.enabled=true ${COMMON_OPTS} ${TIMEOUT_OPTS} ${extra}"
            fi
            echo "${cmd}"
            ;;
        *)
            echo "❌ 未知模式: ${mode}" >&2
            echo "" >&2
            echo "可用模式:" >&2
            echo "  pipeline     — 全流程 (默认, 无sweep_id先sweep再pipeline)" >&2
            echo "  sweep        — 仅超参搜索 (Stage 1)" >&2
            echo "  agent        — 开启分布式 W&B agent (需 sweep_id，运行在隔离沙盒内)" >&2
            echo "  evaluate     — 仅多种子评估 (Stage 2, 需 sweep_id)" >&2
            echo "  ablation     — 消融实验 (需 sweep_id, 自动获取evaluate结果)" >&2
            echo "  sensitivity  — 参数敏感性 (需 sweep_id, 1D/2D绘图+PDF)" >&2
            echo "  dry-run      — 预览模式 [sweep_id 可选]" >&2
            echo "  notify       — 启用邮件通知 [sweep_id 可选]" >&2
            exit 1
            ;;
    esac
}

# ============================================================================
# 主流程
# ============================================================================

CMD=$(build_cmd "${MODE}" "${SWEEP_ID}" "${EXTRA_ARGS}")

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Workflow Runner                                  ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Mode:     ${MODE}"
echo "║  Sweep ID: ${SWEEP_ID:-<new>}"
echo "║  Timeout:  ${TIMEOUT_SECS}s"
echo "║  Env:      ${PYENV}"
echo "║  Devices:  ${DEVICES}"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Command:  ${CMD}"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 执行 (带超时) ────────────────────────────────────────────────────────────
START_TIME=$(date +%s)
EXIT_CODE=0

if [[ "${MODE}" == "dry-run" ]]; then
    echo "🔍 DRY-RUN — 仅预览命令，不执行"
    eval "${CMD}"
    EXIT_CODE=$?
elif [[ "${TIMEOUT_SECS}" -eq 0 ]]; then
    echo "🚀 执行中 (无超时限制)..."
    bash -c "${CMD}" 2>&1 || EXIT_CODE=$?
else
    echo "🚀 执行中 (超时 ${TIMEOUT_SECS}s)..."
    timeout "${TIMEOUT_SECS}s" bash -c "${CMD}" 2>&1 || EXIT_CODE=$?
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [[ "${EXIT_CODE:-0}" -eq 124 ]]; then
    echo ""
    echo "⏰ 超时! 运行超过 ${TIMEOUT_SECS}s"
    echo "   可通过 TIMEOUT_SECS=600 bash scripts/workflow.sh ${MODE} 增加超时"
elif [[ "${EXIT_CODE:-0}" -eq 0 ]]; then
    echo ""
    echo "✅ 完成! 耗时 ${ELAPSED}s"
else
    echo ""
    echo "❌ 失败 (exit code: ${EXIT_CODE})，耗时 ${ELAPSED}s"
fi

exit ${EXIT_CODE:-0}
