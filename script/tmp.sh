# cd /home/shx/code/ce-new

# # 先初始化；如果失败，会停止，不会移动子仓库的 .git
# # git init
# # git branch -m main

# # 备份目录位于 ce-new 外面
# backup_dir="$(mktemp -d /home/shx/code/ce-new-git-backup.XXXXXX)"

# for repo in \
#     casu-2 \
#     baseline/RECE \
#     baseline/unified-concept-editing \
#     baseline/SPEED \
#     baseline/ESD \
#     baseline/AdaVD
# do
#     mkdir -p "$backup_dir/$repo"
#     mv "$repo/.git" "$backup_dir/$repo/.git"
# done

# printf '原子仓库的 Git 元数据已备份到：%s\n' "$backup_dir"
# 



##　还原git删除并删除父仓库
