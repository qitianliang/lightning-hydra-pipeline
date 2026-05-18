#!/bin/bash
user=${User:-"your_user"}
hostname=${HostName:-"your_hostname"}
port=${Port:-"22"}

find_project_root() {
  local current_dir="$PWD"
  local found_dir=""

  # 尝试解析真实路径（如果可用）
  if command -v realpath >/dev/null 2>&1; then
    current_dir=$(realpath "$current_dir")
  fi

  # 向上查找
  while [[ "$current_dir" != "/" && -n "$current_dir" ]]; do
    if [[ -e "$current_dir/.project-root" ]]; then
      found_dir="$current_dir"
      # 继续向上查找，找到最顶层的 .project-root
      while [[ "$current_dir" != "/" && -n "$current_dir" ]]; do
        current_dir=$(dirname "$current_dir")
        if [[ -e "$current_dir/.project-root" ]]; then
          found_dir="$current_dir"
        fi
      done
      echo "$found_dir"
      return 0
    fi
    current_dir=$(dirname "$current_dir")
  done

  echo "错误: 未找到 .project-root 标记文件" >&2
  return 1
}

project_dir=$(find_project_root)
echo "项目根目录: $project_dir"

# 安全获取父目录的函数
get_parent_dir() {
  local path="$1"

  # 如果未提供路径，使用当前目录
  if [[ -z "$path" ]]; then
    path="$PWD"
  fi

  # 处理根目录
  if [[ "$path" == "/" ]]; then
    echo "/"
    return 0
  fi

  # 移除末尾的斜杠（如果有）
  path="${path%/}"

  # 获取父目录
  echo "${path%/*}"
}
parent_dir=$(get_parent_dir "$project_dir")
echo "父目录：$parent_dir"
rsync -avz -e  "ssh -p ${port}" "${project_dir}" "${user}"@"${hostname}":"${parent_dir}" \
--exclude-from="${project_dir}"/scripts/exclude-list.txt \
--delete
