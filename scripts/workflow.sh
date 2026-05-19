#!/usr/bin/env bash
# ============================================================================
# Workflow Runner Script — 固化多种运行模式 + Worktree 隔离
# ============================================================================
# Usage:
#   bash scripts/workflow.sh <mode> [sweep_id] [extra_args...]
#
# Modes:
#   (default)  — 全流程: sweep + pipeline (Stage 1→4)
#   sweep      — 仅超参搜索 (Stage 1)
#   evaluate   — 仅多种子评估 (Stage 2, 需 sweep_id)
#   pipeline   — 评估+消融+敏感性 (Stage 2→4, 需 sweep_id)
#   ablation   — 消融实验模式 (需 sweep_id, 自动获取 evaluate 结果)
#   sensitivity— 参数敏感性模式 (需 sweep_id, 1D/2D 绘图+PDF)
#   dry-run    — 预览模式: 打印命令不执行 [sweep_id 可选]
#   notify     — 启用邮件通知运行 [sweep_id 可选]
#
# Worktree 隔离:
#   首次启动实验时，自动:
#     1. 将脏工作区 snapshot 到 exp/<mode>_<timestamp> 分支
#     2. 创建 git worktree 隔离目录
#     3. 复制 .env / data/ 等运行时依赖
#     4. 在 worktree 中执行实验
#   后续 eval/ablation/sensitivity 自动定位已有 worktree 或按需重建。
# ============================================================================

set -euo pipefail

# ── 项目根目录 ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── 环境变量 ────────────────────────────────────────────────────────────────
export PYENV="${PYENV:-myenv}"
export TIMEOUT_SECS="${TIMEOUT_SECS:-300}"  # 5min 默认超时; 0=无限制

# ── Worktree 配置 ──────────────────────────────────────────────────────────
export WORKTREE_ENABLED="${WORKTREE_ENABLED:-true}"
WORKTREE_EXTRA_FILES="${WORKTREE_EXTRA_FILES:-.env data/}"
WORKTREE_REGISTRY="${PROJECT_ROOT}/scripts/worktree_registry.json"
WORKTREE_PARENT_DIR="$(dirname "${PROJECT_ROOT}")"

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
SWEEP_ID="${2:-}"
# Shift consumed args safely
if [[ $# -ge 2 ]]; then
    shift 2
elif [[ $# -ge 1 ]]; then
    shift 1
fi
EXTRA_ARGS="${*:-}"

# 公共参数
BASE_CMD="python src/workflow.py"
COMMON_OPTS="workflow.sweep_task.conda_env=${PYENV} workflow.sweep_task.devices=${DEVICES}"

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
# Worktree 管理函数
# ============================================================================

# 读取 sweep config 的 name 字段
get_sweep_name() {
    local sweep_config="${1:-configs/sweep/mnist_sweep.yaml}"
    if [[ -f "${sweep_config}" ]]; then
        grep -E "^name:" "${sweep_config}" | head -1 | sed 's/name:\s*//' | tr -d '"' | tr -d "'"
    else
        echo "base"
    fi
}

# 生成时间戳 (格式: 20260517_203430)
get_timestamp() {
    date +"%Y%m%d_%H%M%S"
}

# 获取项目目录名 (如 lightning-hydra-sequence)
get_project_dirname() {
    basename "$(git -C "${PROJECT_ROOT}" rev-parse --show-toplevel)"
}

# 获取当前 HEAD commit hash (短)
get_head_commit() {
    git -C "${PROJECT_ROOT}" rev-parse HEAD
}

# 检查工作区是否有未提交更改
has_dirty_changes() {
    ! git -C "${PROJECT_ROOT}" diff --quiet 2>/dev/null || \
    ! git -C "${PROJECT_ROOT}" diff --cached --quiet 2>/dev/null || \
    [[ -n "$(git -C "${PROJECT_ROOT}" ls-files --others --exclude-standard 2>/dev/null)" ]]
}

# Snapshot 脏工作区 (add all + commit)
snapshot_dirty_workspace() {
    local msg="$1"
    if has_dirty_changes; then
        echo "📸 Dirty workspace detected. Creating snapshot..."
        git -C "${PROJECT_ROOT}" add -A
        git -C "${PROJECT_ROOT}" commit -m "${msg}" --allow-empty
        echo "✅ Snapshot committed: ${msg}"
    else
        echo "✅ Workspace clean, no snapshot needed."
    fi
}

# 复制 extra files 到 worktree
copy_extra_files() {
    local worktree_path="$1"
    local file
    for file in $(echo ${WORKTREE_EXTRA_FILES}); do
        local src="${PROJECT_ROOT}/${file}"
        local dst="${worktree_path}/${file}"
        if [[ -e "${src}" ]]; then
            if [[ -d "${src}" ]]; then
                # 目录: rsync 保持结构
                mkdir -p "${dst}"
                rsync -av --exclude='.git' "${src}/" "${dst}/" || {
                    echo "  ❌ rsync failed for ${file}, trying cp -r..."
                    cp -r "${src}" "${dst}" || {
                        echo "  ❌ cp -r also failed for ${file}!"
                        exit 1
                    }
                }
                echo "  📁 Copied directory: ${file}"
            else
                # 文件: 直接复制
                mkdir -p "$(dirname "${dst}")"
                cp "${src}" "${dst}"
                echo "  📄 Copied file: ${file}"
            fi
        else
            echo "  ⚠️  Extra file not found, skipping: ${file}"
        fi
    done
}

# 生成清理脚本
generate_cleanup_script() {
    local worktree_path="$1"
    local branch_name="$2"
    local registry_key="$3"
    local clean_dir="${PROJECT_ROOT}/scripts/clean"
    local clean_name
    clean_name="cleanup_worktree_${registry_key}.sh"
    local clean_path="${clean_dir}/${clean_name}"

    cat > "${clean_path}" <<CLEANUP_EOF
#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 自动生成 — 执行后自动销毁自身
# ============================================================================

CLEANUP_SCRIPT="${clean_path}"
MAIN_REPO="${PROJECT_ROOT}"
WORKTREE_PATH="${worktree_path}"
BRANCH_NAME="${branch_name}"

# --dry-run 模式
DRY_RUN=false
if [[ "\${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 DRY-RUN MODE — 仅预览，不执行删除"
fi

echo "╔══════════════════════════════════════════════════╗"
echo "║  实验清理脚本                                      ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Worktree: \${WORKTREE_PATH}"
echo "║  Branch:   \${BRANCH_NAME}"
echo "║  Main:     \${MAIN_REPO}"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Step 1: 删除 worktree
if [[ -d "\${WORKTREE_PATH}" ]]; then
    echo "🗑️  正在删除 worktree: \${WORKTREE_PATH}"
    if ! \${DRY_RUN}; then
        git -C "\${MAIN_REPO}" worktree remove "\${WORKTREE_PATH}" --force
    fi
    echo "✅ Worktree 已删除"
else
    echo "⚠️  Worktree 目录不存在，跳过: \${WORKTREE_PATH}"
fi

# Step 2: 清理 worktree 残留引用
echo "🧹 清理 worktree 残留引用..."
if ! \${DRY_RUN}; then
    git -C "\${MAIN_REPO}" worktree prune
fi
echo "✅ Worktree 引用已清理"

# Step 3: 删除实验分支
if git -C "\${MAIN_REPO}" branch --list "\${BRANCH_NAME}" | grep -q "\${BRANCH_NAME}"; then
    echo "🗑️  正在删除分支: \${BRANCH_NAME}"
    if ! \${DRY_RUN}; then
        git -C "\${MAIN_REPO}" branch -D "\${BRANCH_NAME}"
    fi
    echo "✅ 分支已删除"
else
    echo "⚠️  分支不存在，跳过: \${BRANCH_NAME}"
fi

echo ""
echo "🎉 清理完成!"
# Step 4: 自毁 — 清理脚本自身
if ! \${DRY_RUN}; then
    rm -f "\${CLEANUP_SCRIPT}"
    echo "🗑️  清理脚本已自毁"
fi
CLEANUP_EOF

    chmod +x "${clean_path}"
    echo "🧹 Cleanup script generated: ${clean_path}"
}

# 在 worktree 中重新执行当前命令
reexec_in_worktree() {
    local worktree_path="$1"
    echo ""
    echo "🔄 Re-executing in worktree: ${worktree_path}"
    echo "   IN_WORKTREE=1 bash scripts/workflow.sh ${MODE} ${SWEEP_ID:-""} ${EXTRA_ARGS}"
    echo ""

    # Use subshell so parent's CWD unchanged after return
    (
        cd "${worktree_path}"
        IN_WORKTREE=1 bash scripts/workflow.sh ${MODE} ${SWEEP_ID:-""} ${EXTRA_ARGS}
    )
    return $?
}

# 创建新 worktree (用于新 sweep)
create_worktree_for_new_sweep() {
    local mode="$1"
    local timestamp
    timestamp="$(get_timestamp)"

    # 读取 sweep_name (从 sweep config 的 name 字段)
    local sweep_name
    sweep_name="$(get_sweep_name "configs/sweep/mnist_sweep.yaml")"

    local proj_dirname
    proj_dirname="$(get_project_dirname)"

    local registry_key="${sweep_name}_${timestamp}"
    local branch_name="exp/${mode}_${timestamp}"
    local worktree_name="${proj_dirname}_${sweep_name}_${timestamp}"
    local worktree_path="${WORKTREE_PARENT_DIR}/${worktree_name}"

    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  🌿 Worktree 隔离模式                             ║"
    echo "╠══════════════════════════════════════════════════╣"
    echo "║  Sweep name:  ${sweep_name}"
    echo "║  Branch:      ${branch_name}"
    echo "║  Worktree:    ${worktree_path}"
    echo "║  Registry:    ${registry_key}"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""

    # Step 1: Snapshot 脏工作区
    snapshot_dirty_workspace "snapshot: ${mode} ${timestamp}"

    # Step 2: 记录当前分支
    local main_branch
    main_branch="$(git -C "${PROJECT_ROOT}" branch --show-current)"

    # Step 3: 创建实验分支
    git -C "${PROJECT_ROOT}" branch "${branch_name}" HEAD
    echo "✅ Created branch: ${branch_name}"

    # Step 4: 获取 commit hash
    local commit_hash
    commit_hash="$(get_head_commit)"
    echo "📌 HEAD commit: ${commit_hash:0:8}"

    # Step 5: 创建 worktree
    if [[ -d "${worktree_path}" ]]; then
        echo "❌ Worktree directory already exists: ${worktree_path}"
        exit 1
    fi
    git -C "${PROJECT_ROOT}" worktree add "${worktree_path}" "${commit_hash}"
    echo "✅ Worktree created: ${worktree_path}"

    # Step 6: 复制 extra files
    echo "📋 Copying extra files to worktree..."
    copy_extra_files "${worktree_path}"

    # Step 7: 注册到 registry
    python scripts/worktree_helper.py register \
        --registry "${WORKTREE_REGISTRY}" \
        --key "${registry_key}" \
        --sweep-id "" \
        --commit "${commit_hash}" \
        --worktree-path "${worktree_path}" \
        --branch "${branch_name}" \
        --sweep-name "${sweep_name}" \
        --timestamp "${timestamp}"

    # Step 8: 生成清理脚本
    generate_cleanup_script "${worktree_path}" "${branch_name}" "${registry_key}"

    # Step 9: 在 worktree 中执行
    reexec_in_worktree "${worktree_path}"
    local exit_code=$?

    # Step 10: sweep 完成后尝试获取 sweep_id 并更新 registry
    # (从 latest status 文件读取)
    local latest_status_dir="${worktree_path}/logs/sweeps"
    if [[ -d "${latest_status_dir}" ]]; then
        local latest_link
        latest_link="$(find "${latest_status_dir}" -name "*-latest.yaml" -type l 2>/dev/null | head -1)"
        if [[ -z "${latest_link}" ]]; then
            latest_link="$(find "${latest_status_dir}" -name "*-latest.yaml" -type f 2>/dev/null | head -1)"
        fi
        if [[ -n "${latest_link}" && -f "${latest_link}" ]]; then
            local new_sweep_id
            new_sweep_id="$(grep 'sweep_id:' "${latest_link}" | head -1 | awk '{print $2}' | tr -d '"' | tr -d "'")"
            if [[ -n "${new_sweep_id}" ]]; then
                echo "📝 Updating registry with sweep_id: ${new_sweep_id}"
                python scripts/worktree_helper.py update-sweep-id \
                    --registry "${WORKTREE_REGISTRY}" \
                    --key "${registry_key}" \
                    --sweep-id "${new_sweep_id}"
            fi
        fi
    fi

    return ${exit_code}
}

# 查找或创建 worktree (用于 eval/ablation/sensitivity)
find_or_create_worktree() {
    local mode="$1"
    local sweep_id="$2"

    if [[ -z "${sweep_id}" ]]; then
        echo "❌ find_or_create_worktree requires sweep_id"
        exit 1
    fi

    echo "🔍 Looking up worktree for sweep_id=${sweep_id}..."

    # 查询注册表
    local find_output
    find_output="$(python scripts/worktree_helper.py find-worktree \
        --registry "${WORKTREE_REGISTRY}" \
        --sweep-id "${sweep_id}" 2>/dev/null || true)"

    local found_worktree
    local found_commit
    local found_branch
    found_worktree="$(echo "${find_output}" | sed -n '1p')"
    found_commit="$(echo "${find_output}" | sed -n '2p')"
    found_branch="$(echo "${find_output}" | sed -n '3p')"

    # 场景1: worktree 存在且目录有效 → 直接复用
    if [[ -n "${found_worktree}" && -d "${found_worktree}" ]]; then
        echo "✅ Found existing worktree: ${found_worktree}"
        reexec_in_worktree "${found_worktree}"
        return $?
    fi

    # 场景2: worktree 被删或未找到 → 需要获取 commit 并重建
    local commit_to_use="${found_commit}"
    local branch_to_use="${found_branch}"

    # 如果没有 commit 信息，从 W&B 获取
    if [[ -z "${commit_to_use}" ]]; then
        echo "🔍 No commit in registry. Querying W&B for best_run.commit..."

        local wb_entity="${WANDB_ENTITY:-qitianliang}"
        local wb_project="${WANDB_PROJECT:-mnist}"

        # 从 .env 或环境变量获取 W&B 信息
        if [[ -f "${PROJECT_ROOT}/.env" ]]; then
            wb_entity="$(grep -E '^WANDB_ENTITY=' "${PROJECT_ROOT}/.env" 2>/dev/null | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'" || echo "${wb_entity}")"
            wb_project="$(grep -E '^WANDB_PROJECT=' "${PROJECT_ROOT}/.env" 2>/dev/null | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'" || echo "${wb_project}")"
        fi

        local wb_output
        wb_output="$(python scripts/worktree_helper.py get-commit \
            --entity "${wb_entity}" \
            --project "${wb_project}" \
            --sweep-id "${sweep_id}" 2>&1)" || {
            echo "❌ Failed to get commit from W&B: ${wb_output}"
            exit 1
        }

        commit_to_use="$(echo "${wb_output}" | sed -n '1p')"
        local sweep_name_from_wb
        sweep_name_from_wb="$(echo "${wb_output}" | sed -n '2p')"
    fi

    # 确保 commit 可达 (跳过 SSH host key 验证)
    echo "🔄 Ensuring commit ${commit_to_use:0:8} is reachable..."
    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git -C "${PROJECT_ROOT}" fetch origin 2>/dev/null || true

    # 验证 commit 存在
    if ! git -C "${PROJECT_ROOT}" cat-file -e "${commit_to_use}" 2>/dev/null; then
        echo "❌ Commit ${commit_to_use:0:8} not found in local repo. Trying fetch..."
        git -C "${PROJECT_ROOT}" fetch origin "${commit_to_use}" 2>/dev/null || {
            echo "❌ Cannot fetch commit ${commit_to_use:0:8}. Aborting."
            exit 1
        }
    fi

    # 重建 worktree
    local timestamp
    timestamp="$(get_timestamp)"
    local proj_dirname
    proj_dirname="$(get_project_dirname)"
    local sweep_name
    sweep_name="$(get_sweep_name "configs/sweep/mnist_sweep.yaml")"
    local registry_key
    if [[ -n "${found_branch}" ]]; then
        # 使用原有的 registry key (从 branch 推导)
        registry_key="${sweep_name}_${timestamp}"
    else
        registry_key="${sweep_name}_${timestamp}"
    fi
    local new_branch="exp/${mode}_${timestamp}"
    local worktree_name="${proj_dirname}_${sweep_name}_${timestamp}"
    local worktree_path="${WORKTREE_PARENT_DIR}/${worktree_name}"

    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  🌿 重建 Worktree                                  ║"
    echo "╠══════════════════════════════════════════════════╣"
    echo "║  Sweep ID:    ${sweep_id}"
    echo "║  Commit:      ${commit_to_use:0:8}"
    echo "║  Worktree:    ${worktree_path}"
    echo "║  Branch:      ${new_branch}"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""

    echo "⚠️  WARNING: Rebuilding worktree from commit ${commit_to_use:0:8}"
    echo "   .env and data/ will be copied from CURRENT workspace."

    # 创建 worktree
    if [[ -d "${worktree_path}" ]]; then
        echo "❌ Worktree directory already exists: ${worktree_path}"
        exit 1
    fi
    git -C "${PROJECT_ROOT}" worktree add "${worktree_path}" "${commit_to_use}"
    echo "✅ Worktree created: ${worktree_path}"

    # 创建分支标记
    git -C "${PROJECT_ROOT}" branch "${new_branch}" "${commit_to_use}" 2>/dev/null || true

    # 复制 extra files (从当前工作区)
    echo "📋 Copying extra files to worktree (from current workspace)..."
    copy_extra_files "${worktree_path}"

    # 注册到 registry
    python scripts/worktree_helper.py register \
        --registry "${WORKTREE_REGISTRY}" \
        --key "${registry_key}" \
        --sweep-id "${sweep_id}" \
        --commit "${commit_to_use}" \
        --worktree-path "${worktree_path}" \
        --branch "${new_branch}" \
        --sweep-name "${sweep_name}" \
        --timestamp "${timestamp}"

    # 生成清理脚本
    generate_cleanup_script "${worktree_path}" "${new_branch}" "${registry_key}"

    # 在 worktree 中执行
    reexec_in_worktree "${worktree_path}"
    return $?
}

# ============================================================================
# 主流程
# ============================================================================

# ── Worktree 隔离检查 ──────────────────────────────────────────────────────
if [[ "${IN_WORKTREE:-0}" != "1" ]] && [[ "${WORKTREE_ENABLED}" == "true" ]]; then
    # 不在 worktree 中 → 根据模式决定 worktree 流程

    # dry-run 模式不创建 worktree
    if [[ "${MODE}" == "dry-run" ]]; then
        : # 跳过 worktree 管理，直接执行
    elif [[ "${MODE}" == "notify" ]]; then
        : # notify 模式跳过
    elif [[ -z "${SWEEP_ID}" ]]; then
        # 新 sweep (无 sweep_id) → 创建新 worktree
        create_worktree_for_new_sweep "${MODE}"
        exit $?
    elif [[ -n "${SWEEP_ID}" ]]; then
        # 有 sweep_id (eval/ablation/sensitivity) → 查找或创建 worktree
        find_or_create_worktree "${MODE}" "${SWEEP_ID}"
        exit $?
    fi
fi

# ── 在 worktree 中 (或 WORKTREE_ENABLED=false) → 正常执行 ──────────────────

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
if [[ "${IN_WORKTREE:-0}" == "1" ]]; then
echo "║  Worktree: ✅ YES (isolated)"
fi
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