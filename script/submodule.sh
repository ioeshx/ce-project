## 
cd /home/shx/code/ce-new
git -C casu-2 push origin main

# 登记并提交到父仓库：
git submodule add https://github.com/ioeshx/ce-new.git casu-2
git add .gitmodules casu-2
git diff --cached --submodule
git commit -m "Add casu-2 as a submodule"
# 以后在其他机器上一起克隆：
git clone --recurse-submodules '父仓库远程URL' ce-new
# 如果已经克隆了父仓库，在父仓库目录执行：
git submodule update --init --recursive