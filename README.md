# AnimeFinder · 智能动漫推荐与发现系统

课程项目 MVP：知识推理 + 协同过滤 + 遗传算法 + RAG 聊天机器人，四类技术一体化。

## 技术栈

| 层 | 选型 |
|----|------|
| 后端 | Python + FastAPI + SQLite（零配置） |
| 前端 | 原生 HTML/CSS/JS（无框架） |
| 知识推理 | pandas 思路的属性加权匹配（`services/knowledge.py`） |
| 协同过滤 | Item-based CF，item-item 余弦相似度（`services/cf.py`） |
| 遗传算法 | 纯 Python 手写，Fitness=0.4R+0.3P+0.2D+0.1N（`services/ga.py`） |
| RAG 聊天 | 检索 + 模板生成（离线可用），可切换 LLM API（`services/rag.py`） |

## 启动

### 方式一：服务管理脚本（推荐）

**Windows** — 双击或命令行执行：

```bat
service.bat start      :: 启动服务(后台)
service.bat stop       :: 停止服务
service.bat restart    :: 重启服务
service.bat status     :: 查看运行状态
```

**Linux / macOS / Git Bash**：

```bash
./service.sh start     # 启动服务(后台)
./service.sh stop      # 停止服务
./service.sh restart   # 重启服务
./service.sh status    # 查看运行状态
```

脚本会自动挑选可用 Python、缺依赖时自动安装、写 PID 与日志（`backend/server.pid`、`backend/server.log`）。

### 方式二：手动启动

```bash
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

打开 http://localhost:8000 即可使用。
接口文档（Swagger）在 http://localhost:8000/docs。

## 演示账号

- 用户 1 `demo`：收藏/高分均为心理·智斗系（死亡笔记、石头门、鲁路修…）
- 用户 2 `visitor`：热血战斗系。右上角切换用户可看到推荐结果差异。

## 接入真实 MyAnimeList 数据集（Kaggle dbdmobile/myanimelist-dataset）

`backend/database.py` 已预留接口 `load_mal_dataset(anime_csv, score_csv, details_csv)`，
docstring 里写明了每列的映射与注意事项（过滤无评分/成人内容、分块读取、Jikan API 补 themes/demographics/characters）。
实现该函数后删除 `anime.db` 重新启动即可无缝替换种子数据。

## 开启 LLM 生成（可选）

默认为模板回复（离线可演示）。配置环境变量后自动切换为 LLM 生成：

```bash
export LLM_API_BASE=https://api.deepseek.com/v1
export LLM_API_KEY=sk-xxx
export LLM_MODEL=deepseek-chat
```

## 语义检索升级（可选）

`services/rag.py` 的 `_score()` 是演示级词面匹配；替换为
`sentence-transformers(BGE) + ChromaDB` 只需改这一个函数，其余接口不变。
