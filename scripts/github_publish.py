# -*- coding: utf-8 -*-
"""
GitHub 仓库创建与推送脚本（方案 B：任何网络环境可执行）

用法（在能访问 GitHub 的网络环境）：
  1. 先完成 gh 认证：  gh auth login        （或 gh auth login --with-token < token.txt）
  2. 运行本脚本：      python scripts/github_publish.py
     或手动执行脚本内打印的 git 命令

步骤：
  - 用 gh repo create 创建仓库（public/private 可选）
  - 添加 remote 并推送当前分支（仅推送被 .gitignore 允许的文件）
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REPO_NAME = "vsb-pd-foundation"   # 仓库名（可改）
VISIBILITY = "public"             # public / private（投稿评审期建议 private，录用后转 public）
BRANCH = "main"                   # 推送分支


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout)
    if r.stderr:
        print(r.stderr, file=sys.stderr)
    if r.returncode != 0:
        print(f"!! exit {r.returncode}")


def main() -> None:
    # 0. 检查 gh
    r = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if r.returncode != 0:
        print("gh 未登录。请先运行: gh auth login  或  gh auth login --with-token")
        sys.exit(1)

    # 1. 检查/重命名分支
    run(["git", "branch", "-M", BRANCH])

    # 2. 创建仓库（不存在时）
    run(["gh", "repo", "create", REPO_NAME, "--" + VISIBILITY,
         "--description", "Coverage-aware sampling and hierarchical weakly-supervised "
                          "PD detection (VSB dataset) - IEEE TIM submission",
         "--source", str(ROOT), "--push"])

    # 3. 若仓库已存在则仅添加 remote + push
    r = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
    if r.returncode != 0:
        run(["git", "remote", "add", "origin",
             f"https://github.com/YldkS123/{REPO_NAME}.git"])
    run(["git", "push", "-u", "origin", BRANCH])

    print("\nDone. 仓库: https://github.com/YldkS123/" + REPO_NAME)
    print("提示: 评审期建议 private；录用后再 gh repo edit <repo> --visibility public")


if __name__ == "__main__":
    main()
