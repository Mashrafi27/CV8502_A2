#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/git_push.sh "commit message"
# Commits all tracked/untracked (respecting .gitignore) and pushes to origin main.

msg="${1:-update}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

git status -sb
git add .
git commit -m "$msg"
git push -u origin main
