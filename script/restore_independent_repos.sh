#!/usr/bin/env bash
# 恢复 baseline 的独立仓库，并将 ce-new/.git 移到目录外备份。
# 用法：bash /home/shx/code/ce-new/restore_independent_repos.sh [原备份目录]
set -euo pipefail
export GIT_OPTIONAL_LOCKS=0

repo_root='/home/shx/code/ce-new'
backup_root="${1:-/home/shx/code/ce-new-git-backup.EcIqqM}"
methods=(RECE unified-concept-editing SPEED ESD AdaVD)

fail() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

[[ $# -le 1 ]] || fail '最多接受一个参数：原备份目录。'
[[ -d "$repo_root/baseline" ]] || fail '找不到 baseline 目录。'
[[ -d "$backup_root" ]] || fail "找不到原备份目录：$backup_root"
[[ ! -L "$repo_root/.git" ]] || fail 'ce-new/.git 是符号链接，停止处理。'
if [[ -e "$repo_root/.git" && ! -d "$repo_root/.git" ]]; then
    fail 'ce-new/.git 不是普通目录，停止处理。'
fi

# 若 casu-2 已改成依赖父仓库元数据的子模块，停止，避免破坏它。
casu_git_dir="$(git -C "$repo_root/casu-2" rev-parse --absolute-git-dir)" \
    || fail '无法读取 casu-2 的 Git 元数据。'
case "$casu_git_dir" in
    "$repo_root/.git"|"$repo_root/.git/"*)
        fail 'casu-2 依赖 ce-new/.git，需先将其恢复为独立仓库。'
        ;;
esac

# 先检查全部目标和备份，已有的 .git 一律不覆盖。
for method in "${methods[@]}"; do
    target="$repo_root/baseline/$method"
    source_git="$backup_root/baseline/$method/.git"
    [[ -d "$target" && ! -L "$target" ]] || fail "方法目录不存在或是符号链接：$target"
    [[ ! -e "$target/.git" && ! -L "$target/.git" ]] \
        || fail "目标已有 .git，不覆盖：$target/.git"
    [[ -d "$source_git" && ! -L "$source_git" ]] \
        || fail "找不到有效的 Git 备份目录：$source_git"
    git --git-dir="$source_git" rev-parse --verify 'HEAD^{commit}' >/dev/null \
        || fail "无法读取备份的提交：$source_git"
done

# 在 ce-new 外准备恢复副本，原始备份始终保留。
archive_root="$(mktemp -d /home/shx/code/ce-new-git-rollback.XXXXXX)"
printf '本次恢复的暂存和父仓库备份目录：%s\n' "$archive_root"
for method in "${methods[@]}"; do
    mkdir -p "$archive_root/staged/$method"
    cp -a -- "$backup_root/baseline/$method/.git" "$archive_root/staged/$method/.git"
    git --git-dir="$archive_root/staged/$method/.git" \
        rev-parse --verify 'HEAD^{commit}' >/dev/null
done

# 全部副本就绪后，逐个放回对应的方法目录。
for method in "${methods[@]}"; do
    target="$repo_root/baseline/$method"
    [[ ! -e "$target/.git" && ! -L "$target/.git" ]] \
        || fail "目标出现了 .git，停止覆盖：$target/.git"
    mv -T -- "$archive_root/staged/$method/.git" "$target/.git"
    printf '已恢复：%s/.git\n' "$target"
done

# 所有方法仓库验证通过后，才移走父仓库的 .git。
for method in "${methods[@]}"; do
    target="$repo_root/baseline/$method"
    [[ "$(git -C "$target" rev-parse --show-toplevel)" == "$target" ]] \
        || fail "仓库根目录不符合预期：$target"
    printf '\n%s\n' "$target"
    git -C "$target" status --short --branch --untracked-files=no
done

if [[ -d "$repo_root/.git" ]]; then
    mv -T -- "$repo_root/.git" "$archive_root/ce-new.git"
    printf '\n已取消 ce-new 的 Git 仓库，原 .git 备份到：%s/ce-new.git\n' "$archive_root"
else
    printf '\nce-new 已没有 .git，无需移除。\n'
fi
printf '恢复完成。casu-2、代码文件和原始备份均已保留。\n'
