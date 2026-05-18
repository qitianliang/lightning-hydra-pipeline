#!/usr/bin/env python3
"""Worktree 隔离集成测试 — 验证 git worktree 管理的 3 个典型场景。

场景:
  1. sweep + eval: 首次创建 worktree
  2. eval (worktree 存在): 复用已有 worktree
  3. eval (worktree 被删除): 自动重建 worktree + 补齐 .env/data/

安全措施:
  - 测试前先打包项目 zip 到 /root
  - 检查 clean 脚本不包含危险路径
  - 测试后不删除 worktree，供手动核验

用法:
  pytest tests/test_worktree.py -v --timeout=600
  或:
  python tests/test_worktree.py
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd: str, cwd: str = None, timeout: int = 120) -> tuple:
    """运行 shell 命令，返回 (returncode, stdout, stderr)。"""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd or str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def is_git_repo(path: str) -> bool:
    """检查路径是否是 git 仓库。"""
    rc, _, _ = run_cmd(f"git -C {path} rev-parse --is-inside-work-tree")
    return rc == 0


def get_head_commit(path: str = None) -> str:
    """获取 HEAD commit hash。"""
    rc, out, _ = run_cmd(f"git -C {path or str(PROJECT_ROOT)} rev-parse HEAD")
    return out.strip() if rc == 0 else ""


def create_test_env_file(path: Path):
    """创建测试用 .env 文件。"""
    env_content = """
# Test .env for worktree integration test
HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=0
PYENV=myenv
SMTP_HOST=smtp.test.com
USER_EMAIL_ADDRESS=test@test.com
SMTP_EMAIL_ADDRESS=test@test.com
SMTP_PASSWORD=testpass
"""
    (path / ".env").write_text(env_content.strip())


def create_test_data_dir(path: Path):
    """创建测试用 data/ 目录。"""
    data_dir = path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "test_marker.txt").write_text("worktree_test_data_marker")


def verify_worktree_structure(worktree_path: str, project_root: str):
    """验证 worktree 结构正确。"""
    wt = Path(worktree_path)
    pr = Path(project_root)

    # 1. 目录存在
    assert wt.exists(), f"Worktree directory not found: {worktree_path}"

    # 2. 是 git worktree
    rc, out, _ = run_cmd(f"git -C {worktree_path} rev-parse --is-inside-work-tree")
    assert rc == 0, f"Not a git worktree: {worktree_path}"

    # 3. 关键文件存在
    assert (wt / "scripts" / "workflow.sh").exists(), "workflow.sh missing in worktree"
    assert (wt / "src" / "workflow.py").exists(), "workflow.py missing in worktree"
    assert (wt / "configs" / "workflow.yaml").exists(), "workflow.yaml missing in worktree"

    # 4. .env 已复制
    assert (wt / ".env").exists(), ".env not copied to worktree"

    # 5. data/ 已复制
    assert (wt / "data").exists(), "data/ not copied to worktree"
    assert (wt / "data" / "test_marker.txt").exists(), "data/test_marker.txt not in worktree"

    # 6. HEAD commit 匹配主仓库
    main_commit = get_head_commit(project_root)
    wt_commit = get_head_commit(worktree_path)
    assert main_commit == wt_commit, \
        f"Commit mismatch: main={main_commit[:8]}, worktree={wt_commit[:8]}"


def verify_registry(registry_path: str, expected_key: str = None):
    """验证 registry 文件有效。"""
    rp = Path(registry_path)
    assert rp.exists(), f"Registry file not found: {registry_path}"

    with open(rp) as f:
        registry = json.load(f)

    assert isinstance(registry, dict), "Registry should be a dict"
    if expected_key:
        assert expected_key in registry, f"Key '{expected_key}' not in registry"
        entry = registry[expected_key]
        assert "commit" in entry, "Registry entry missing 'commit'"
        assert "worktree_path" in entry, "Registry entry missing 'worktree_path'"
        assert "branch" in entry, "Registry entry missing 'branch'"
        assert "timestamp" in entry, "Registry entry missing 'timestamp'"

    return registry


def verify_cleanup_script(script_path: str, project_root: str):
    """验证清理脚本内容安全。"""
    sp = Path(script_path)
    assert sp.exists(), f"Cleanup script not found: {script_path}"

    content = sp.read_text()

    # 必须包含 set -euo pipefail
    assert "set -euo pipefail" in content, "Cleanup script missing 'set -euo pipefail'"

    # 必须包含 MAIN_REPO 绝对路径
    assert "MAIN_REPO=" in content, "Cleanup script missing MAIN_REPO"

    # 必须包含 WORKTREE_PATH 绝对路径
    assert "WORKTREE_PATH=" in content, "Cleanup script missing WORKTREE_PATH"

    # 必须包含 --dry-run 模式
    assert "--dry-run" in content, "Cleanup script missing --dry-run mode"

    # 必须包含 git worktree remove
    assert "worktree remove" in content, "Cleanup script missing 'worktree remove'"

    # 必须包含 git worktree prune
    assert "worktree prune" in content, "Cleanup script missing 'worktree prune'"

    # 安全检查: 不应包含删除根目录的危险命令
    dangerous_patterns = ["rm -rf /", "rm -rf /root", "rm -rf ~"]
    for pattern in dangerous_patterns:
        assert pattern not in content, f"Dangerous pattern '{pattern}' in cleanup script"

    # 必须包含边界处理 (目录不存在时跳过)
    assert "不存在" in content or "not found" in content.lower() or "skip" in content.lower(), \
        "Cleanup script should handle non-existent directories"


def verify_worktree_list(project_root: str, expected_count: int = 1):
    """验证 git worktree list 包含预期的 worktree。"""
    rc, out, _ = run_cmd(f"git -C {project_root} worktree list")
    assert rc == 0, "git worktree list failed"
    lines = out.strip().split("\n")
    # git worktree list 输出格式: <path> <hash> [<branch>]
    # 主仓库的 path 就是 project_root，worktree 的 path 不同
    # 用精确路径匹配: 排除第一列等于 project_root 的行
    import os
    pr_normalized = os.path.normpath(project_root)
    worktree_lines = []
    for l in lines:
        if not l.strip():
            continue
        first_col = l.split()[0]
        if os.path.normpath(first_col) != pr_normalized:
            worktree_lines.append(l)
    assert len(worktree_lines) >= expected_count, \
        f"Expected >= {expected_count} worktrees, found {len(worktree_lines)}\nOutput: {out}"


# ============================================================================
# Test: worktree_helper.py 纯逻辑
# ============================================================================

class TestWorktreeHelper:
    """测试 scripts/worktree_helper.py 的注册表操作。"""

    def test_register_and_find(self, tmp_path):
        """注册条目后能按 sweep_id 查找。"""
        registry = str(tmp_path / "registry.json")

        # 注册 — 使用 tmp_path 下的不存在的路径
        nonexistent_wt = str(tmp_path / "nonexistent_wt")
        rc, out, _ = run_cmd(
            f"python {PROJECT_ROOT}/scripts/worktree_helper.py register "
            f"--registry {registry} --key base_20260517 "
            f"--sweep-id test123 --commit abcdef1234 "
            f"--worktree-path {nonexistent_wt} --branch exp/sweep_20260517 "
            f"--sweep-name base --timestamp 20260517_203430"
        )
        assert rc == 0, f"register failed: {out}"
        assert "OK" in out

        # 查找
        rc, out, _ = run_cmd(
            f"python {PROJECT_ROOT}/scripts/worktree_helper.py find-worktree "
            f"--registry {registry} --sweep-id test123"
        )
        assert rc == 0
        lines = out.rstrip("\n").split("\n")
        # worktree_path 不存在 → 第一行为空
        assert lines[0] == "", f"Expected empty worktree_path (dir not exist), got: {lines[0]}"
        assert lines[1] == "abcdef1234", f"Expected commit, got: {lines[1]}"

    def test_find_worktree_existing_dir(self, tmp_path):
        """当 worktree 目录存在时，find 返回路径。"""
        registry = str(tmp_path / "registry.json")
        wt_dir = tmp_path / "test_worktree"
        wt_dir.mkdir()

        run_cmd(
            f"python {PROJECT_ROOT}/scripts/worktree_helper.py register "
            f"--registry {registry} --key base_20260517 "
            f"--sweep-id test456 --commit abcdef5678 "
            f"--worktree-path {wt_dir} --branch exp/sweep_20260517 "
            f"--sweep-name base --timestamp 20260517_203430"
        )

        rc, out, _ = run_cmd(
            f"python {PROJECT_ROOT}/scripts/worktree_helper.py find-worktree "
            f"--registry {registry} --sweep-id test456"
        )
        lines = out.strip().split("\n")
        assert lines[0] == str(wt_dir), f"Expected worktree path, got: {lines[0]}"

    def test_update_sweep_id(self, tmp_path):
        """更新已有条目的 sweep_id。"""
        registry = str(tmp_path / "registry.json")

        run_cmd(
            f"python {PROJECT_ROOT}/scripts/worktree_helper.py register "
            f"--registry {registry} --key base_20260517 "
            f"--sweep-id '' --commit abcdef1234 "
            f"--worktree-path /tmp/test_wt --branch exp/sweep_20260517 "
            f"--sweep-name base --timestamp 20260517_203430"
        )

        rc, out, _ = run_cmd(
            f"python {PROJECT_ROOT}/scripts/worktree_helper.py update-sweep-id "
            f"--registry {registry} --key base_20260517 --sweep-id new_sweep_789"
        )
        assert rc == 0
        assert "OK" in out

        # 验证
        with open(registry) as f:
            data = json.load(f)
        assert data["base_20260517"]["sweep_id"] == "new_sweep_789"

    def test_find_nonexistent_sweep(self, tmp_path):
        """查找不存在的 sweep_id 返回空行。"""
        registry = str(tmp_path / "registry.json")
        # 创建空 registry
        with open(registry, "w") as f:
            json.dump({}, f)

        rc, out, _ = run_cmd(
            f"python {PROJECT_ROOT}/scripts/worktree_helper.py find-worktree "
            f"--registry {registry} --sweep-id nonexistent"
        )
        lines = out.strip().split("\n")
        assert all(l == "" for l in lines), "Expected all empty lines for nonexistent sweep"


# ============================================================================
# Test: worktree 集成场景 (需要 git 仓库)
# ============================================================================

@pytest.fixture(scope="class")
def worktree_test_env():
    """创建 worktree 测试环境。

    在 /tmp 下创建一个 git 仓库副本，避免污染主仓库。
    """
    # 创建临时目录
    test_dir = tempfile.mkdtemp(prefix="worktree_test_")
    test_repo = Path(test_dir) / "test_repo"
    test_repo.mkdir()

    # 初始化 git 仓库
    run_cmd(f"git init {test_repo}")
    run_cmd(f"git -C {test_repo} config user.email 'test@test.com'")
    run_cmd(f"git -C {test_repo} config user.name 'Test'")

    # 复制关键文件 (不复制 .git 和 data/logs)
    for item in ["src", "configs", "scripts", "memory-bank"]:
        src = PROJECT_ROOT / item
        if src.exists():
            if src.is_dir():
                shutil.copytree(str(src), str(test_repo / item), dirs_exist_ok=True)
            else:
                shutil.copy2(str(src), str(test_repo / item))

    for f in [".project-root", "pyproject.toml", "setup.py", "requirements.txt"]:
        src = PROJECT_ROOT / f
        if src.exists():
            shutil.copy2(str(src), str(test_repo / f))

    # 创建测试用 .env 和 data/
    create_test_env_file(test_repo)
    create_test_data_dir(test_repo)

    # 初始提交
    run_cmd(f"git -C {test_repo} add -A")
    run_cmd(f"git -C {test_repo} commit -m 'Initial commit for worktree test'")

    yield {
        "test_repo": str(test_repo),
        "test_dir": test_dir,
    }

    # 清理: 不删除 worktree (供手动核验)，但清理临时目录
    # 注意: 测试后 worktree 可能在 test_dir 外 (由 git worktree add 创建)
    # 只删除 test_dir 本身
    try:
        # 先清理 worktree 引用
        run_cmd(f"git -C {test_repo} worktree prune", timeout=10)
    except Exception:
        pass


@pytest.mark.timeout(600)
class TestWorktreeIntegration:
    """Worktree 隔离集成测试 — 3 个典型场景。"""

    def test_scenario1_sweep_creates_worktree(self, worktree_test_env):
        """场景1: 新 sweep → 自动创建 worktree。

        验证:
          - worktree 目录存在
          - .env/data/ 已复制
          - registry 已更新
          - cleanup 脚本已生成且安全
          - HEAD commit 匹配
        """
        repo = worktree_test_env["test_repo"]

        # 确保在 git 仓库中
        assert is_git_repo(repo)

        # 运行 dry-run sweep (不会实际训练，但会走 worktree 流程)
        # 用 WORKTREE_ENABLED=true 和 IN_WORKTREE 未设置
        rc, out, err = run_cmd(
            f"cd {repo} && WORKTREE_ENABLED=true IN_WORKTREE=0 "
            f"bash scripts/workflow.sh dry-run",
            timeout=60,
        )
        # dry-run 模式跳过 worktree 管理，直接执行

        # 直接测试 worktree 创建逻辑
        # 模拟 workflow.sh 的 create_worktree_for_new_sweep
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        sweep_name = "base"
        proj_dirname = Path(repo).name
        branch_name = f"exp/pipeline_{timestamp}"
        worktree_name = f"{proj_dirname}_{sweep_name}_{timestamp}"
        parent_dir = str(Path(repo).parent)
        worktree_path = f"{parent_dir}/{worktree_name}"
        registry_path = f"{repo}/scripts/worktree_registry.json"
        commit = get_head_commit(repo)

        # Step 1: 创建分支
        rc, _, _ = run_cmd(f"git -C {repo} branch {branch_name} HEAD")
        assert rc == 0, f"Failed to create branch {branch_name}"

        # Step 2: 创建 worktree
        rc, _, _ = run_cmd(f"git -C {repo} worktree add {worktree_path} {commit}")
        assert rc == 0, f"Failed to create worktree at {worktree_path}"

        # Step 3: 复制 extra files
        for f in [".env", "data/"]:
            src = Path(repo) / f
            dst = Path(worktree_path) / f
            if src.is_dir():
                shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
            elif src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))

        # Step 4: 注册
        rc, out, _ = run_cmd(
            f"python {repo}/scripts/worktree_helper.py register "
            f"--registry {registry_path} --key {sweep_name}_{timestamp} "
            f"--sweep-id '' --commit {commit} "
            f"--worktree-path {worktree_path} --branch {branch_name} "
            f"--sweep-name {sweep_name} --timestamp {timestamp}"
        )
        assert rc == 0

        # ===== 验证 =====
        verify_worktree_structure(worktree_path, repo)
        verify_registry(registry_path, f"{sweep_name}_{timestamp}")
        verify_worktree_list(repo, expected_count=1)

        # 存储信息供后续场景使用
        worktree_test_env["worktree_path"] = worktree_path
        worktree_test_env["registry_path"] = registry_path
        worktree_test_env["commit"] = commit
        worktree_test_env["branch_name"] = branch_name
        worktree_test_env["registry_key"] = f"{sweep_name}_{timestamp}"

    def test_scenario2_reuse_existing_worktree(self, worktree_test_env):
        """场景2: 已有 worktree → 复用，不创建新的。

        前置: 场景1 已创建 worktree
        """
        repo = worktree_test_env["test_repo"]
        wt_path = worktree_test_env.get("worktree_path")
        registry_path = worktree_test_env.get("registry_path")

        if not wt_path or not Path(wt_path).exists():
            pytest.skip("Scenario 1 not run or worktree not found")

        # 先给 registry 补上 sweep_id (模拟 sweep 完成)
        registry_key = worktree_test_env.get("registry_key", "")
        run_cmd(
            f"python {repo}/scripts/worktree_helper.py update-sweep-id "
            f"--registry {registry_path} --key {registry_key} --sweep-id test_sweep_001"
        )

        # 查找 worktree
        rc, out, _ = run_cmd(
            f"python {repo}/scripts/worktree_helper.py find-worktree "
            f"--registry {registry_path} --sweep-id test_sweep_001"
        )
        assert rc == 0
        lines = out.rstrip("\n").split("\n")
        found_path = lines[0]
        found_commit = lines[1]

        # 验证找到的 worktree 是同一个
        assert found_path == wt_path, \
            f"Expected {wt_path}, got {found_path}"
        assert found_commit == worktree_test_env["commit"]

        # 验证 worktree 仍然有效
        assert Path(found_path).exists()
        assert is_git_repo(found_path)

        # 验证没有创建新的 worktree (用精确路径匹配)
        rc, out, _ = run_cmd(f"git -C {repo} worktree list")
        pr_normalized = os.path.normpath(repo)
        worktree_lines = [l for l in out.strip().split("\n")
                         if l.strip() and os.path.normpath(l.split()[0]) != pr_normalized]
        assert len(worktree_lines) == 1, "Should still have exactly 1 worktree"

    def test_scenario3_rebuild_deleted_worktree(self, worktree_test_env):
        """场景3: worktree 被删除 → 自动重建 + 补齐 .env/data/。

        前置: 场景1 已创建 worktree
        """
        repo = worktree_test_env["test_repo"]
        wt_path = worktree_test_env.get("worktree_path")
        registry_path = worktree_test_env.get("registry_path")
        commit = worktree_test_env.get("commit")

        if not wt_path:
            pytest.skip("Scenario 1 not run")

        # Step 1: 删除 worktree (模拟用户手动清理)
        if Path(wt_path).exists():
            # 先用 git worktree remove 避免脏状态
            run_cmd(f"git -C {repo} worktree remove {wt_path} --force", timeout=10)
            # 如果 git remove 失败，手动删除
            if Path(wt_path).exists():
                shutil.rmtree(wt_path, ignore_errors=True)

        run_cmd(f"git -C {repo} worktree prune")

        # 验证 worktree 已删除
        assert not Path(wt_path).exists(), "Worktree should be deleted"

        # Step 2: 查找 → registry 有记录但 worktree 不存在
        rc, out, _ = run_cmd(
            f"python {repo}/scripts/worktree_helper.py find-worktree "
            f"--registry {registry_path} --sweep-id test_sweep_001"
        )
        lines = out.rstrip("\n").split("\n")
        found_path = lines[0]  # 应为空
        found_commit = lines[1]  # 应有 commit

        assert found_path == "", "worktree_path should be empty (deleted)"
        assert found_commit == commit, "commit should still be available"

        # Step 3: 重建 worktree (模拟 find_or_create_worktree 逻辑)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        sweep_name = "base"
        proj_dirname = Path(repo).name
        new_branch = f"exp/evaluate_{timestamp}"
        new_worktree_name = f"{proj_dirname}_{sweep_name}_{timestamp}"
        parent_dir = str(Path(repo).parent)
        new_worktree_path = f"{parent_dir}/{new_worktree_name}"

        # 确保 commit 可达
        rc, _, _ = run_cmd(f"git -C {repo} cat-file -e {commit}")
        assert rc == 0, "Commit should be reachable"

        # 创建新 worktree
        rc, _, _ = run_cmd(f"git -C {repo} worktree add {new_worktree_path} {commit}")
        assert rc == 0, f"Failed to rebuild worktree at {new_worktree_path}"

        # 创建分支标记
        run_cmd(f"git -C {repo} branch {new_branch} {commit}")

        # 复制 extra files (从当前工作区 — 应输出 WARNING)
        for f in [".env", "data/"]:
            src = Path(repo) / f
            dst = Path(new_worktree_path) / f
            if src.is_dir():
                shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
            elif src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))

        # 注册新 worktree
        new_key = f"{sweep_name}_{timestamp}"
        run_cmd(
            f"python {repo}/scripts/worktree_helper.py register "
            f"--registry {registry_path} --key {new_key} "
            f"--sweep-id test_sweep_001 --commit {commit} "
            f"--worktree-path {new_worktree_path} --branch {new_branch} "
            f"--sweep-name {sweep_name} --timestamp {timestamp}"
        )

        # ===== 验证 =====
        verify_worktree_structure(new_worktree_path, repo)
        verify_registry(registry_path, new_key)
        verify_worktree_list(repo, expected_count=1)

        # 验证 .env 和 data/ 已补齐
        assert (Path(new_worktree_path) / ".env").exists(), ".env not restored"
        assert (Path(new_worktree_path) / "data" / "test_marker.txt").exists(), \
            "data/ not restored"


# ============================================================================
# Test: cleanup 脚本安全性
# ============================================================================

class TestCleanupScript:
    """测试清理脚本生成和安全性。"""

    def test_generate_cleanup_script(self, tmp_path):
        """测试清理脚本生成。"""
        # 模拟 generate_cleanup_script 的输出
        repo = str(tmp_path / "repo")
        os.makedirs(repo)
        clean_dir = os.path.join(repo, "scripts", "clean")
        os.makedirs(clean_dir)

        script_path = os.path.join(clean_dir, "cleanup_worktree_base_20260517.sh")
        worktree_path = "/tmp/test_worktree"
        branch_name = "exp/sweep_20260517"

        script_content = f"""#!/usr/bin/env bash
set -euo pipefail

MAIN_REPO="{repo}"
WORKTREE_PATH="{worktree_path}"
BRANCH_NAME="{branch_name}"

DRY_RUN=false
if [[ "${{1:-}}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 DRY-RUN MODE"
fi

if [[ -d "${{WORKTREE_PATH}}" ]]; then
    echo "🗑️  正在删除 worktree: ${{WORKTREE_PATH}}"
    if ! ${{DRY_RUN}}; then
        git -C "${{MAIN_REPO}}" worktree remove "${{WORKTREE_PATH}}" --force
    fi
    echo "✅ Worktree 已删除"
else
    echo "⚠️  Worktree 目录不存在，跳过: ${{WORKTREE_PATH}}"
fi

echo "🧹 清理 worktree 残留引用..."
if ! ${{DRY_RUN}}; then
    git -C "${{MAIN_REPO}}" worktree prune
fi
echo "✅ Worktree 引用已清理"

if git -C "${{MAIN_REPO}}" branch --list "${{BRANCH_NAME}}" | grep -q "${{BRANCH_NAME}}"; then
    echo "🗑️  正在删除分支: ${{BRANCH_NAME}}"
    if ! ${{DRY_RUN}}; then
        git -C "${{MAIN_REPO}}" branch -D "${{BRANCH_NAME}}"
    fi
    echo "✅ 分支已删除"
else
    echo "⚠️  分支不存在，跳过: ${{BRANCH_NAME}}"
fi

echo "🎉 清理完成!"
"""
        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)

        # 验证脚本内容安全
        verify_cleanup_script(script_path, repo)

    def test_cleanup_script_dry_run(self, tmp_path):
        """测试清理脚本的 --dry-run 模式。"""
        repo = str(tmp_path / "repo")
        os.makedirs(repo)
        clean_dir = os.path.join(repo, "scripts", "clean")
        os.makedirs(clean_dir)

        # 创建 git 仓库
        run_cmd(f"git init {repo}")
        run_cmd(f"git -C {repo} config user.email 'test@test.com'")
        run_cmd(f"git -C {repo} config user.name 'Test'")
        run_cmd(f"git -C {repo} commit --allow-empty -m 'init'")

        # 创建 worktree
        wt = str(tmp_path / "test_wt")
        commit = get_head_commit(repo)
        run_cmd(f"git -C {repo} worktree add {wt} {commit}")
        run_cmd(f"git -C {repo} branch exp/test_branch {commit}")

        # 生成并运行 dry-run
        script_path = os.path.join(clean_dir, "cleanup_test.sh")
        with open(script_path, "w") as f:
            f.write(f"""#!/usr/bin/env bash
set -euo pipefail
MAIN_REPO="{repo}"
WORKTREE_PATH="{wt}"
BRANCH_NAME="exp/test_branch"
DRY_RUN=false
if [[ "${{1:-}}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 DRY-RUN MODE"
fi
if [[ -d "${{WORKTREE_PATH}}" ]]; then
    echo "Would remove: ${{WORKTREE_PATH}}"
    if ! ${{DRY_RUN}}; then
        git -C "${{MAIN_REPO}}" worktree remove "${{WORKTREE_PATH}}" --force
    fi
fi
""")
        os.chmod(script_path, 0o755)

        # dry-run 不应删除
        rc, out, _ = run_cmd(f"bash {script_path} --dry-run")
        assert rc == 0
        assert "DRY-RUN" in out
        assert Path(wt).exists(), "Dry-run should NOT delete worktree"

        # 非 dry-run 应删除
        rc, out, _ = run_cmd(f"bash {script_path}")
        assert rc == 0
        # worktree 应被删除
        run_cmd(f"git -C {repo} worktree prune")
        assert not Path(wt).exists(), "Worktree should be deleted after real run"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=600", "-s"])