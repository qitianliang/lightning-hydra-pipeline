#!/usr/bin/env python3
"""Worktree 管理助手 — 供 workflow.sh 调用，提供 W&B 查询和 worktree 注册表操作。

子命令:
  get-commit   — 查询 sweep 的 best_run commit hash
  find-worktree — 在注册表中查找已有 worktree
  register     — 注册/更新 worktree 条目
  update-sweep-id — 补充 sweep_id 到已有条目

文件锁: 使用 fcntl.flock 防止多进程竞态写入 worktree_registry.json。
"""

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path
from typing import Optional


def _locked_read_registry(reg_path: Path) -> dict:
    """Acquire shared lock, read registry JSON. Returns {} if file missing."""
    if not reg_path.exists():
        return {}
    with open(reg_path) as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return json.load(f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _locked_write_registry(reg_path: Path, data: dict) -> None:
    """Acquire exclusive lock, write registry JSON atomically."""
    with open(reg_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def get_commit(entity: str, project: str, sweep_id: str) -> None:
    """从 W&B 获取 sweep best_run 的 git commit hash。"""
    import wandb

    api = wandb.Api()
    sweep_path = f"{entity}/{project}/{sweep_id}"
    try:
        sweep = api.sweep(sweep_path)
    except Exception as e:
        print(f"ERROR: Cannot fetch sweep '{sweep_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # 获取 best run
    runs = sorted(
        sweep.runs,
        key=lambda r: r.summary.get("val/acc_best", -1),
        reverse=True,
    )
    if not runs:
        print(f"ERROR: No runs found in sweep '{sweep_id}'", file=sys.stderr)
        sys.exit(1)

    best_run = runs[0]
    commit = getattr(best_run, "commit", None) or best_run.config.get("git_commit", None)
    if not commit:
        print(f"ERROR: best_run has no commit info", file=sys.stderr)
        sys.exit(1)

    # 输出格式: commit_hash\nsweep_name
    sweep_name = sweep.config.get("name", "unknown") if sweep.config else "unknown"
    print(commit)
    print(sweep_name)


def find_worktree(registry_path: str, sweep_id: str) -> None:
    """在注册表中查找 sweep_id 对应的 worktree。

    输出格式 (3 行):
      worktree_path (或空行)
      commit_hash (或空行)
      branch_name (或空行)
    如果未找到或 worktree 目录不存在，输出空行。
    """
    reg_file = Path(registry_path)
    if not reg_file.exists():
        print("")
        print("")
        print("")
        return

    registry = _locked_read_registry(reg_file)

    # 按 sweep_id 查找
    for key, entry in registry.items():
        if entry.get("sweep_id") == sweep_id:
            wt_path = entry.get("worktree_path", "")
            commit = entry.get("commit", "")
            branch = entry.get("branch", "")

            # 验证 worktree 目录仍存在
            if wt_path and Path(wt_path).exists():
                print(wt_path)
                print(commit)
                print(branch)
                return
            else:
                # worktree 已被删除，返回 commit 信息用于重建
                print("")  # worktree_path 为空 → 需要重建
                print(commit)
                print(branch)
                return

    # 未找到
    print("")
    print("")
    print("")


def register(
    registry_path: str,
    key: str,
    sweep_id: str,
    commit: str,
    worktree_path: str,
    branch: str,
    sweep_name: str,
    timestamp: str,
) -> None:
    """注册或更新 worktree 条目 (带文件锁)。"""
    reg_file = Path(registry_path)
    registry = _locked_read_registry(reg_file)

    registry[key] = {
        "sweep_id": sweep_id,
        "sweep_name": sweep_name,
        "commit": commit,
        "worktree_path": worktree_path,
        "branch": branch,
        "timestamp": timestamp,
    }

    _locked_write_registry(reg_file, registry)
    print(f"OK: Registered worktree '{key}'")


def update_sweep_id(registry_path: str, key: str, sweep_id: str) -> None:
    """补充 sweep_id 到已有注册条目 (带文件锁)。"""
    reg_file = Path(registry_path)

    if not reg_file.exists():
        print(f"ERROR: Registry not found: {registry_path}", file=sys.stderr)
        sys.exit(1)

    registry = _locked_read_registry(reg_file)

    if key not in registry:
        print(f"ERROR: Key '{key}' not found in registry", file=sys.stderr)
        sys.exit(1)

    registry[key]["sweep_id"] = sweep_id

    _locked_write_registry(reg_file, registry)
    print(f"OK: Updated sweep_id='{sweep_id}' for key '{key}'")


def main():
    parser = argparse.ArgumentParser(description="Worktree management helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # get-commit
    p_gc = subparsers.add_parser("get-commit", help="Get best_run commit from W&B sweep")
    p_gc.add_argument("--entity", required=True)
    p_gc.add_argument("--project", required=True)
    p_gc.add_argument("--sweep-id", required=True)

    # find-worktree
    p_fw = subparsers.add_parser("find-worktree", help="Find worktree in registry by sweep_id")
    p_fw.add_argument("--registry", required=True)
    p_fw.add_argument("--sweep-id", required=True)

    # register
    p_reg = subparsers.add_parser("register", help="Register worktree entry")
    p_reg.add_argument("--registry", required=True)
    p_reg.add_argument("--key", required=True)
    p_reg.add_argument("--sweep-id", default="")
    p_reg.add_argument("--commit", required=True)
    p_reg.add_argument("--worktree-path", required=True)
    p_reg.add_argument("--branch", required=True)
    p_reg.add_argument("--sweep-name", default="")
    p_reg.add_argument("--timestamp", required=True)

    # update-sweep-id
    p_usi = subparsers.add_parser("update-sweep-id", help="Update sweep_id for existing entry")
    p_usi.add_argument("--registry", required=True)
    p_usi.add_argument("--key", required=True)
    p_usi.add_argument("--sweep-id", required=True)

    args = parser.parse_args()

    if args.command == "get-commit":
        get_commit(args.entity, args.project, args.sweep_id)
    elif args.command == "find-worktree":
        find_worktree(args.registry, args.sweep_id)
    elif args.command == "register":
        register(
            args.registry, args.key, args.sweep_id, args.commit,
            args.worktree_path, args.branch, args.sweep_name, args.timestamp,
        )
    elif args.command == "update-sweep-id":
        update_sweep_id(args.registry, args.key, args.sweep_id)


if __name__ == "__main__":
    main()