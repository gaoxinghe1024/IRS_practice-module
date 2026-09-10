# AnimeFinder 开发日志（DEVLOG）

> **⚠️ 给所有协作者 / Agent 的强制规则：**
> 1. 对本项目的**任何改动**（新增文件、修改逻辑、修 bug、调数据、改接口）完成后，必须在本文件**追加一条带日期的记录**。
> 2. 记录格式：`## [YYYY-MM-DD] 改动标题` + 变更内容、涉及文件、原因、验证方式。
> 3. 追加到文件**末尾**，不要改写历史记录；如需纠错，另起一条注明。
> 4. 改动前先通读本日志，避免与他人/已有方案冲突。
> 5. 未记录的改动视为未完成。
> 6. **前端页面必须一个页面一个文件（强制）**：一个页面 = 一个独立 HTML 文件 + 一个同名页面级 JS（如 `chat.html` + `js/chat.js`）；跨页面共享的逻辑/组件放 `js/common.js`，共享样式放 `css/style.css`。禁止把多个页面塞进同一个 HTML（SPA 单文件、全塞 index 等）。新增页面时照此结构扩展。

---

## [2026-09-09] 项目初始版本 v1.0

**状态：** 基础系统完成，接口全部验证通过。

**技术栈：** Python 3.13 + FastAPI + SQLite 后端；原生 HTML/CSS/JS 前端（无框架）；算法纯 Python 实现（无重型依赖）。

**已完成：**

1. **数据库层 `backend/database.py`**
   - SQLite 五张表：anime / users / ratings / favorites / scenes。
   - 演示种子数据：24 部知名动漫（含 genres、themes、demographics、franchise、characters、synopsis）、62 个偏好聚类用户评分、8 条名场面。
   - 预留真实数据集接口 `load_mal_dataset()`（Kaggle dbdmobile/myanimelist-dataset），列映射与注意事项写在 docstring。

2. **四大算法模块 `backend/services/`**
   - `knowledge.py`：知识推理相似推荐（Similarity=w1·Genre+w2·Theme+w3·Demo+w4·Studio+w5·Length）+ 轻量知识图谱（Anime/Character/Studio/Franchise 四类节点、三/四类边）。
   - `cf.py`：Item-based 协同过滤，item-item 余弦相似度（评分中心化，共同用户≥3 抑噪），支持缓存失效重建；冷启动回退全局高分。
   - `ga.py`：遗传算法列表级重排，Fitness=0.4·Relevance+0.3·Personalization+0.2·Diversity+0.1·Novelty，锦标赛选择+精英保留+两点交叉+变异，输出最优解与贪心基线对比。
   - `rag.py`：RAG 聊天（检索→上下文拼装→生成），意图分流（名场面匹配/需求推荐），自然语言硬约束抽取（N集内/评分≥X/“不要太长”）；默认模板生成离线可演示，配置 `LLM_API_BASE/LLM_API_KEY` 环境变量后自动切换 OpenAI 兼容接口。

3. **后端 API `backend/main.py`**（Swagger 见 /docs）
   - `GET /api/anime/list`（搜索/类型过滤/分页）、`GET /api/anime/{id}`、`GET /api/anime/{id}/similar?offset=`（换一换）、`GET /api/anime/{id}/kg`
   - `GET /api/recommend/home`（ICF召回→GA重排）、`GET /api/recommend/personal`（规则筛选+收藏画像混合推理）
   - `POST /api/rate`、`POST /api/favorite`、`GET /api/user/{id}/favorites|ratings`
   - `POST /api/chat`
   - 静态托管 `frontend/`，单端口启动。

4. **前端 `frontend/`**（原生三件套）
   - 三个视图：主页推荐（含 GA 适应度对比展示）、浏览全部（搜索+类型筛选）、聊天找番（气泡聊天+快捷提问+作品卡片跳转详情）。
   - 详情弹窗：信息/标签/收藏/星级评分/相似推荐（换一换分页）/知识图谱视图。
   - 封面为渐变占位（数据集接入后换成真实封面图接口即可）。

**已修复：**
- `rag.py` 中文检索分词 bug：中文整句被当作单个 token 无法匹配 → 改为「CJK 单字 + 邻接二元组」轻量分词。

**待办 / 交接给后续协作者：**
- [ ] 实现 `load_mal_dataset()` 接入 Kaggle 真实数据（过滤无评分/成人内容，分块读取）。
- [ ] 用 Jikan API 补 themes / demographics / characters（优先级最高，见项目文档缺口分析）。
- [ ] RAG 检索升级为 BGE embedding + ChromaDB（只需替换 `rag.py` 的 `_tokenize/_score`，接口不变）。
- [ ] 名场面库 `scenes` 目前仅 8 条演示数据，需自行扩充。
- [ ] 观看计划优化（GA 组合优化：时间预算约束）、截图识番、冷启动问卷等附加功能。
- [ ] 前端封面接入真实图片 URL（anime 表可加 image_url 列）。

## [2026-09-09] 修复 rag.py 检索层三个 bug（回归 4/4 通过）

**涉及文件：** `backend/services/rag.py`

1. **中文分词失效**：`_tokenize()` 原实现无中文分词，整句被当成一个 token 导致检索永远 0 分 → 改为「CJK 单字 + 邻接二元组」轻量分词，拉丁词保持完整。
2. **检索分覆盖评分字段**：`retrieve()` 里 `{**a, "score": sc}` 把作品评分覆盖成检索分，导致"评分≥9"硬约束失效、回复中评分显示为小数 → 检索分改用独立字段 `ret_score`。
3. **硬条件查询空结果**：纯约束查询（"评分8.8以上 20集以内"）时检索命中的 14 条全被硬条件淘汰且不触发兜底 → 改为**过滤后为空才用全目录兜底**，另补 `评分X.X以上` 语序的约束抽取正则。

**验证：** 四类聊天场景（名场面匹配 / 模糊需求 / 硬条件 / 硬条件边界）全部通过；KG 接口 8 节点 7 边正常；GA 适应度 0.5694 > 基线 0.5647；评分/收藏闭环正常。

## [2026-09-09] v1.1：前端按页拆分 + 观看计划优化 + Jikan 数据补全

**状态：** 全部完成并回归通过；Jikan 补全 15/24（上游抖动，余 9 部稍后重跑）。

### 1. 前端重构：一个页面一个文件（响应团队要求，规则已写入本手册第 6 条）

- `frontend/index.html` + `js/home.js` —— 主页推荐 + 观看计划面板
- `frontend/explore.html` + `js/explore.js` —— 浏览全部
- `frontend/chat.html` + `js/chat.js` —— 聊天找番
- `frontend/js/common.js` —— 共享组件（API 封装 / 用户状态 localStorage 持久化 / 卡片 / 详情弹窗）
- `frontend/css/style.css` —— 共享样式；删除旧 `app.js` 单文件
- 用户选择跨页面保持（localStorage）；封面有真图显示真图，无则渐变占位。

### 2. 观看计划优化（附加功能 6.3，GA 组合优化）

- 新增 `backend/services/watchplan.py`：染色体 = 变长子集，时长约束内 max Preference+Quality+Diversity+Novelty+Utilization；交叉=并集+按性价比贪心裁剪回预算；变异=随机增删。
- 新增 `POST /api/watchplan`（参数 user_id / hours / keywords）。
- 修复：多样性的两两配对为空时取中性 0.5（否则单部作品白拿满分，GA 收敛到"一部超长番"）；加入预算利用率项鼓励填满时间。
- 已知局限：种子库 24 部太少，15 小时预算的心理/悬疑池填不满——真实数据集接入后自然解决。

### 3. Jikan API 数据补全（数据 pipeline 第一步）

- 新增 `backend/jikan_enrich.py`：回填 mal_id / 官方封面图 / themes（英文标签内置中英映射）/ demographics / 主要角色。
- `anime` 表新增 `mal_id`、`image_url` 两列（含旧库幂等迁移）；`SCENE_SEED` 名场面库扩到 12 条。
- **结果：15/24 成功**。Death Note 现有官方标签 Psychological(心理) / Shounen(少年向)——即项目缺口分析里"最痛的缺口"已部分补上；同时拿到 15 张官方封面。
- **未完成：9 部失败**（Monster、Mushishi、NGE、Vinland、Higurashi、Your Name、四月、工作细胞、L外传）。原因：Jikan 上游 MyAnimeList 当时拒绝连接（HTTP 504），缓存外的条目全部失败。**处理：稍后重跑 `python jikan_enrich.py` 即可（幂等，只处理 mal_id=0 的条目）。**
- 经验记录：本机环境下 urllib 与 curl 对 Jikan 的表现不一致，脚本已做 urllib→curl 双通道兜底 + 429/5xx 退避重试；未翻译标签已补 Childcare(育儿)、Parody(戏仿恶搞)。

**回归结果：** 详情/相似推荐/主页推荐/聊天/观看计划接口全部正常；三个页面与静态资源 200。

**交接待办（更新）：**
- [ ] **重跑 `python jikan_enrich.py` 补齐剩余 9 部**（上游恢复后即可）。
- [ ] 实现 `load_mal_dataset()` 接入 Kaggle 真实数据（过滤无评分/成人内容，分块读取）。
- [ ] RAG 检索升级为 BGE embedding + ChromaDB（替换 `rag.py` 的 `_tokenize/_score` 即可）。
- [ ] 截图识番、冷启动问卷、推荐解释等附加功能。

## [2026-09-10] Git 仓库初始化并推送 GitHub

**涉及文件：** 新增 `.gitignore`；解决 `README.md` 合并冲突。

### 1. Git 仓库初始化

- `git init`，主分支命名 `main`，完成首次提交。
- 新增 `.gitignore`：排除 `*.db`（SQLite 可重建）、`__pycache__`、虚拟环境、`.env` 等。
- 远程仓库：`https://github.com/gaoxinghe1024/IRS_practice-module.git`。
- 远程建仓时自带一个占位 README 初始提交，与本地无共同历史，已通过 `git pull --allow-unrelated-histories` 合并并解决 `README.md` 冲突（保留本地完整文档），随后推送成功。

### 2. 环境说明（重要）

- 受管 venv（`~/.workbuddy/.../envs/default`）曾被中断的 pip 安装损坏，已弃用。
- **本项目使用专用 conda 环境 `animedev`（Python 3.12），与科研环境（base/new_env）完全隔离。**
  - 解释器路径：`E:\anaconda3\envs\animedev\python.exe`
  - 激活方式：`conda activate animedev`
  - 已安装：fastapi / uvicorn[standard] / pandas / scikit-learn / requests / networkx / python-multipart
- `requirements.txt` 已补全为项目实际依赖（含可选的 sentence-transformers / chromadb 注释项）。
- **后续协作者请统一使用 `animedev` 环境**，不要再往 base 或其他科研环境安装本项目依赖。

**交接待办（更新）：**
- [ ] 重跑 `python jikan_enrich.py` 补齐剩余 9 部（上游恢复后即可）。
- [ ] 实现 `load_mal_dataset()` 接入 Kaggle 真实数据（过滤无评分/成人内容，分块读取）。
- [ ] RAG 检索升级为 BGE embedding + ChromaDB（替换 `rag.py` 的 `_tokenize/_score` 即可）。
- [ ] 截图识番、冷启动问卷、推荐解释等附加功能。
