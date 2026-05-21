#!/usr/bin/env bash
# ============================================================================
# Four workflow modes integration smoke test
#
# Default layer:
#   Offline dry-run coverage for sweep / evaluate / ablation / sensitivity.
#
# Optional layer:
#   RUN_REAL_WANDB=1 REAL_SWEEP_ID=<id> bash tests/integration/test_four_modes.sh
#   Runs short W&B smoke checks for evaluate / ablation / sensitivity.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYENV="${PYENV:-cu129}"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/lhp_four_modes.XXXXXX")"

export WANDB_BASE_URL="${WANDB_BASE_URL:-http://127.0.0.1:19999}"
export WANDB_API_KEY="${WANDB_API_KEY:-local-12345}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0

activate_env() {
    if [[ -n "${SKIP_CONDA_ACTIVATE:-}" ]]; then
        echo "SKIP_CONDA_ACTIVATE is set; using current Python: $(command -v python)"
        return
    fi

    local conda_base=""
    if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base 2>/dev/null || true)"
    elif [[ -d "/root/miniforge3" ]]; then
        conda_base="/root/miniforge3"
    fi

    if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${conda_base}/etc/profile.d/conda.sh"
        conda activate "${PYENV}"
        echo "Active env: ${CONDA_DEFAULT_ENV:-unknown}"
    else
        echo -e "${YELLOW}⚠️  conda not found; using current Python: $(command -v python)${NC}"
    fi
}

join_common_opts() {
    local name="$1"
    local run_dir="${TEST_ROOT}/${name}/hydra"
    local log_dir="${TEST_ROOT}/${name}/logs"
    local output_dir="${TEST_ROOT}/${name}/outputs"

    printf "%s " \
        "workflow.dry_run=true" \
        "workflow.sweep_task.conda_env=${PYENV}" \
        "workflow.devices=[0]" \
        "hydra.run.dir=${run_dir}" \
        "paths.log_dir=${log_dir}" \
        "paths.output_dir=${output_dir}" \
        "workflow.general.status_dir=${log_dir}/sweeps" \
        "workflow.evaluate_task.report_path=${log_dir}/final_reports/optimized_results.json" \
        "workflow.ablation_task.report_path=${log_dir}/final_reports/ablation.json" \
        "workflow.sensitivity_task.report_path=${log_dir}/final_reports/sensitivity.json"
}

print_failure_context() {
    local name="$1"
    local cmd="$2"
    local log_file="$3"
    local reason="$4"

    echo -e "${RED}❌ ${name}: ${reason}${NC}"
    echo "Command:"
    echo "  ${cmd}"
    echo "Log:"
    echo "  ${log_file}"
    echo "---- tail -80 ${log_file} ----"
    tail -80 "${log_file}" || true
    echo "--------------------------------"
}

assert_patterns() {
    local name="$1"
    local cmd="$2"
    local log_file="$3"
    local expected_patterns="$4"
    local forbidden_patterns="$5"

    local pattern
    while IFS= read -r pattern; do
        [[ -z "${pattern}" ]] && continue
        if ! grep -qiE "${pattern}" "${log_file}"; then
            print_failure_context "${name}" "${cmd}" "${log_file}" "missing expected pattern: ${pattern}"
            return 1
        fi
    done <<< "${expected_patterns}"

    while IFS= read -r pattern; do
        [[ -z "${pattern}" ]] && continue
        if grep -qiE "${pattern}" "${log_file}"; then
            print_failure_context "${name}" "${cmd}" "${log_file}" "found forbidden pattern: ${pattern}"
            return 1
        fi
    done <<< "${forbidden_patterns}"
}

run_test() {
    local name="$1"
    local cmd="$2"
    local expected_patterns="$3"
    local forbidden_patterns="${4:-}"
    local log_file="${TEST_ROOT}/test_${name}.log"
    local exit_code=0

    echo ""
    echo "========== Testing: ${name} =========="
    echo "Log: ${log_file}"

    set +e
    bash -c "${cmd}" > "${log_file}" 2>&1
    exit_code=$?
    set -e

    if [[ "${exit_code}" -ne 0 ]]; then
        print_failure_context "${name}" "${cmd}" "${log_file}" "exit ${exit_code}"
        FAIL=$((FAIL + 1))
        return
    fi

    if ! assert_patterns "${name}" "${cmd}" "${log_file}" "${expected_patterns}" "${forbidden_patterns}"; then
        FAIL=$((FAIL + 1))
        return
    fi

    echo -e "${GREEN}✅ ${name}: passed${NC}"
    PASS=$((PASS + 1))
}

skip_test() {
    local name="$1"
    local reason="$2"
    echo -e "${YELLOW}↷ ${name}: skipped (${reason})${NC}"
    SKIP=$((SKIP + 1))
}

run_offline_dry_run_tests() {
    local base_cmd="python src/workflow.py"
    local common

    common="$(join_common_opts sweep)"
    run_test "sweep" \
        "${base_cmd} ${common} workflow.task_name=sweep" \
        "DRY-RUN
Stage 1
wandb sweep
dry-run-mock-sweep" \
        "tmux attach|create_workers_session|Traceback"

    common="$(join_common_opts evaluate)"
    run_test "evaluate" \
        "${base_cmd} ${common} workflow.task_name=pipeline workflow.target_sweep_id=dry-run-mock-sweep" \
        "DRY-RUN
Target Sweep ID \[dry-run-mock-sweep\] provided
Pipeline Stage 1.*Full_Model
Stage 2
Mock sweep.*skipping wandb lookup" \
        "Traceback"

    common="$(join_common_opts ablation)"
    run_test "ablation" \
        "${base_cmd} ${common} workflow=ablation workflow.target_sweep_id=dry-run-mock-sweep" \
        "DRY-RUN
task_name: ablation
Target Sweep ID \[dry-run-mock-sweep\] provided
Launching Ablation Task
Skipping ablation.*no valid sweep_id
no_lin1_bn
no_lin2_bn" \
        "Traceback"

    common="$(join_common_opts sensitivity)"
    run_test "sensitivity" \
        "${base_cmd} ${common} workflow=sensitivity workflow.target_sweep_id=dry-run-mock-sweep" \
        "DRY-RUN
task_name: sensitivity
Target Sweep ID \[dry-run-mock-sweep\] provided
Launching Sensitivity Task
Skipping sensitivity.*no valid sweep_id
width_sensitivity
lr_sensitivity" \
        "Traceback"
}

run_real_wandb_smoke_tests() {
    if [[ "${RUN_REAL_WANDB:-0}" != "1" ]]; then
        return
    fi

    if [[ -z "${REAL_SWEEP_ID:-}" ]]; then
        skip_test "real-wandb" "REAL_SWEEP_ID is required when RUN_REAL_WANDB=1"
        return
    fi

    if [[ -z "${WANDB_API_KEY:-}" || -z "${WANDB_BASE_URL:-}" ]]; then
        skip_test "real-wandb" "WANDB_API_KEY and WANDB_BASE_URL are required"
        return
    fi

    echo ""
    echo "========== Optional real W&B smoke tests enabled =========="
    echo "Sweep ID: ${REAL_SWEEP_ID}"
    echo "W&B URL:  ${WANDB_BASE_URL}"

    local base_cmd="python src/workflow.py"
    local common

    common="$(join_common_opts real_evaluate)"
    common="${common/workflow.dry_run=true /}"
    run_test "real_evaluate" \
        "${base_cmd} ${common} workflow.target_sweep_id=${REAL_SWEEP_ID} workflow.evaluate_task.num_seeds=1 workflow.evaluate_task.top_n=1 workflow.evaluate_task.timeout_secs=120" \
        "Target Sweep ID \[${REAL_SWEEP_ID}\] provided
Stage 2" \
        "Traceback"

    common="$(join_common_opts real_ablation)"
    common="${common/workflow.dry_run=true /}"
    run_test "real_ablation" \
        "${base_cmd} ${common} workflow=ablation workflow.target_sweep_id=${REAL_SWEEP_ID} workflow.ablation_task.num_seeds=1 workflow.ablation_task.timeout_secs=120" \
        "task_name: ablation
Launching Ablation Task" \
        "Traceback"

    common="$(join_common_opts real_sensitivity)"
    common="${common/workflow.dry_run=true /}"
    run_test "real_sensitivity" \
        "${base_cmd} ${common} workflow=sensitivity workflow.target_sweep_id=${REAL_SWEEP_ID} workflow.sensitivity_task.num_seeds=1 workflow.sensitivity_task.timeout_secs=120" \
        "task_name: sensitivity
Launching Sensitivity Task" \
        "Traceback"
}

activate_env
run_offline_dry_run_tests
run_real_wandb_smoke_tests

echo ""
echo "══════════════════════════════════════════════════"
echo "  Four workflow modes test result"
echo "══════════════════════════════════════════════════"
echo -e "  Passed:  ${GREEN}${PASS}${NC}"
echo -e "  Failed:  ${RED}${FAIL}${NC}"
echo -e "  Skipped: ${YELLOW}${SKIP}${NC}"
echo "  Logs:    ${TEST_ROOT}"
echo "══════════════════════════════════════════════════"

if [[ "${FAIL}" -gt 0 ]]; then
    exit 1
fi
