# Codex 本地开发约束说明

本文档用于约束 Codex 在当前项目中的工作方式。

当前项目的工作流是：

```text
GitHub 仓库 → 本地拉取代码 → 本地使用 Codex 修改代码 → 推送到 GitHub → 服务器拉取代码 → 服务器运行测试与评估
```

因此，本地不是测试环境。本地只用于代码修改，服务器才用于运行训练、测试、评估和可视化。

---

## 1. 核心约束

Codex 必须遵守以下规则：

1. 只能在本地阅读代码、分析代码、修改代码、生成脚本、整理文档。
2. 不要在本地运行训练、验证、评估、可视化、latent 导出、pytest 或任何 Python 测试脚本。
3. 不要在本地构造 fake data 进行测试，除非用户明确要求。
4. 不要在本地执行任何依赖真实数据、checkpoint、GPU、CUDA、服务器环境的命令。
5. 不要执行 `pip install`、`conda install`、`apt install` 等安装命令。
6. 可以使用只读命令理解项目结构，例如：

```bash
ls
find
rg
grep
sed
cat
git status
git diff
```

7. 每完成一个阶段后，不要声称“已经测试通过”。
8. 应该明确说明：代码未在本地测试，需要同步到服务器后运行。
9. 如果需要测试，请只给出服务器端应该执行的命令，不要在本地执行。
10. 所有新增代码应尽量保持最小侵入，不要大规模重构训练流程。

---

## 2. 服务器路径约定

服务器仓库路径暂定为：

```bash
/data1/Johnny/challenge/wrf/homework
```

如果需要在文档或输出中给出运行命令，请使用变量形式：

```bash
export SERVER_REPO=/data1/Johnny/challenge/wrf/homework
export CONFIG_PATH=<你的配置文件路径>
export CKPT_PATH=<你的checkpoint路径>
export LOG_PATH=<你的训练日志路径>
export OUT_DIR=$SERVER_REPO/tools/jepa_viz/output
```

路径中不确定的部分必须保留为占位符，不要自行猜测。

---

## 3. Codex 允许做的事情

Codex 可以做：

```text
阅读项目结构
查找训练入口
查找模型 forward
查找 loss 计算位置
查找日志记录位置
查找 checkpoint 保存位置
新增评估脚本
新增可视化脚本
新增 README / docs 文档
修改必要的模型接口
整理服务器运行命令
```

---

## 4. Codex 禁止做的事情

Codex 不可以做：

```text
本地运行训练
本地运行验证
本地运行评估
本地导出 latent
本地生成可视化图
本地运行 pytest
本地运行 Python 脚本测试
本地安装依赖
本地创建 fake data 测试
假装代码已经测试通过
```

---

## 5. 给 Codex 的通用提示词

每次让 Codex 执行任务前，可以先粘贴以下内容：

```text
重要工作流约束：

当前代码是在本地通过 GitHub 拉取的副本，我只在本地使用 Codex 修改代码；真实数据、checkpoint、GPU、训练环境都在服务器上。

因此：

1. 你只能在本地阅读代码、修改代码、整理文档、生成脚本。
2. 不要在本地运行训练、验证、评估、可视化、导出 latent、pytest、python 脚本或任何依赖真实数据/checkpoint/GPU 的命令。
3. 不要创建 fake data 后在本地测试，除非我明确要求。
4. 不要执行 pip install / conda install / apt install。
5. 你可以使用只读命令理解项目结构，例如 ls、find、rg、grep、sed、cat、git status、git diff。
6. 每完成一个阶段后，不要说“我已经测试通过”，而是说明“未在本地测试，需要同步到服务器后运行”。
7. 如果需要运行命令，只输出服务器端命令，不要在本地执行。
8. 所有服务器运行命令都要保留路径变量，方便我复制到服务器执行。

服务器仓库路径暂定为：

/data1/Johnny/challenge/wrf/homework

本地只改代码，服务器才运行。
```

---

## 6. 最重要原则

```text
本地只负责开发。
服务器负责运行。
Codex 不要本地测试。
Codex 不要假装测试通过。
所有需要运行的内容都留到服务器执行。
```
