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

## [2026-09-18] v1.2：注册登录 + 收藏夹体系 + 收藏夹混合推荐 + 真实评分与个人中心

**状态：** 全部完成，17 项接口测试 + 原有功能回归通过；测试数据已清理还原。

### 1. 注册 / 登录（新增 `backend/services/auth.py`）

- 密码 PBKDF2-HMAC-SHA256 + 随机盐哈希存储（不存明文）；会话 token（secrets 随机）存 `sessions` 表。
- `users` 表新增 `password_hash` / `created_at` 列（旧库幂等迁移）；demo/visitor 种子用户无密码，仍走演示模式。
- 接口：`POST /api/auth/register | login | logout`、`GET /api/auth/me?token=`。
- 鉴权约定：**带 token 的请求以 token 对应用户为准**（`_resolve_uid`），无 token 退回 `user_id`（演示模式兼容旧前端）。
- 前端新增 `login.html + js/login.js`（登录/注册双 Tab）；侧栏用户区改为动态渲染：已登录显示用户名+退出，未登录显示演示选择器+登录入口。

### 2. 收藏夹体系（点击收藏进入收藏夹）

- 新表 `fav_folders`；`favorites` 表新增 `folder_id` 列（**0 = 默认收藏夹**，虚拟夹不可删）。
- 接口：`GET /api/user/{id}/folders`（含条目数+封面）、`POST /api/folders`、`DELETE /api/folders/{id}`（条目自动移回默认夹）、`GET /api/folders/{id}`（详情+内含作品）、`POST /api/favorites/remove`（从指定夹移除单个）。
- `POST /api/favorite` 增加 `folder_id` 参数：对指定夹收藏/取消。
- 前端新增 `favorites.html + js/favorites.js`：收藏夹卡片网格 → 点击进入夹详情（内含作品可移除、可删夹、支持 `?folder=id` 直达）。

### 3. 收藏夹混合推荐（新增 `backend/services/hybrid.py`）

- 三层信号融合：知识层（夹内 genre/theme 画像标签匹配，0.45）+ 协同层（夹内作品的 ICF 近邻累加，0.35）+ 质量层（MAL 全局分，0.20）→ `HybridScore`。
- 融合分作为相关性喂给 GA（复用 `ga.optimize`）做列表级重排——四大课程算法在收藏场景串联。
- 接口：`GET /api/folders/{id}/recommend`；`cf.py` 新增公开函数 `item_neighbors()` 供复用。
- 前端在夹详情页展示推荐结果、画像标签、GA 适应度 vs 贪心基线，卡片上标注知识/协同分与推荐理由。

### 4. 真实评分系统 + 个人详情页

- `POST /api/rate` 升级：写入带时间戳的真实评分、同步用户均分、重建 ICF 矩阵；`ratings` 表新增 `created_at`（种子数据回填了演示时间）。
- `GET /api/user/{id}/ratings` 返回完整历史（标题/封面/评分/时间倒序）；`GET /api/user/{id}/profile` 返回用户信息+统计（评分数/均分/收藏数/夹数）+ 偏好 genre Top5。
- 前端新增 `profile.html + js/profile.js`：信息卡+统计+偏好标签+历史评分列表（点击进详情弹窗）。

### 修复与遗留

- 修复：`auth.py` 中 `pbkdf2_hmac` 返回 bytes 误用 `.hexdigest()` → `.hex()`；`main.py` 列表内 `**` 解包语法错误。
- 观察到（非本次引入，`rag.py` 未改动）："评分9分以上，20集以内的心理战动漫" 带逗号的句式命中兜底回复，空格版同样；待排查意图识别正则。

**验证：** 注册/重复注册400/登录/错密码400/me/建夹/收藏进夹/夹列表/夹详情/混合推荐/评分/历史/个人详情/移除/删夹回退/登出401/静态页 200 共 17 项通过；主页推荐 GA、聊天、静态资源回归正常。测试用户与测试收藏夹已清理，演示库还原。

**交接待办：**
- [ ] 重跑 `python jikan_enrich.py` 补齐剩余 9 部（上游恢复后即可）。
- [ ] 实现 `load_mal_dataset()` 接入 Kaggle 真实数据。
- [ ] 排查聊天意图识别："评分X分以上，Y集以内" 句式命中兜底。
- [ ] RAG 检索升级为 BGE embedding + ChromaDB。

## [2026-09-18] v1.3：四算法重组 —— 一算法一文件、命名对位、功能各就各位

**背景：** 团队反馈算法呈现混杂。按课程四条技术线重新划分，界面上以「算法①②③④」标签可见可辨。

### 文件与功能对位（services/）

| 算法 | 文件 | 界面位置 |
|------|------|----------|
| ① Knowledge-Based Reasoning | `knowledge.py` | 动漫详情页「相似推荐」（五维加权 + KG） |
| ② Rule × Knowledge 混合推理 | `hybrid.py` | 收藏夹详情页「基于此收藏夹的推荐」：规则层（关键词/评分/集数硬筛选）× 知识层（收藏夹画像标签匹配，match_score = 评分/10 + 命中×0.15）；规则筛空自动降级纯知识层并标记 relaxed |
| ③ ICF + GA Fusion（核心） | `icf_ga_fusion.py`（新） | 主页「为你推荐」：②ICF召回（cf.py）+ ③GA重排（ga.py），融合编排独立成文件 |
| ④ Cognitive LLM + KB + RAG | `rag.py` | 聊天找番页 |
| 组件/附加 | `cf.py` / `ga.py` / `watchplan.py` | cf/ga 为算法②③组件；watchplan 为 GA 附加应用 |

- **移除** `GET /api/recommend/personal`（逻辑并入算法②收藏页混合推理）；**新增** `GET /api/recommend/new`（新剧速递：aired 年份+热度数据库直读，不经过算法，用于对比）。
- `GET /api/folders/{id}/recommend` 参数改为 keywords / min_score / max_episodes / limit（规则层输入）。

### 算法④聊天属性强化

- 命中名场面时后端返回结构化 `scene_focus`（作品/集数/描述/关键词/anime_id），前端渲染成**「回忆片段」对话气泡**（金边引用样式+关键词+「去补这部番→」），与推荐位卡片刻意区分。
- 打字提示分阶段（翻名场面库→比对知识库→组织语言），强化"它在翻知识库"的对话感；候选作品以「📎 顺手捞出来的」聊天附件行呈现，不再铺卡片网格。
- 修复硬条件全灭时只会说"没理解"：新增 near_miss 逻辑，返回恰好满足一个条件、按另一条件差距排序的「最接近」候选和话术（如"9分以上20集以内"→推荐最接近的）。
- "类似X"查询把被点名的作品本身挪到候选末尾。

### 修复

- **默认收藏夹跨用户泄漏**：folder_id=0 为全用户共享虚拟夹，夹详情/混合推荐未按 user 过滤，demo 画像混入了 visitor 收藏 → 所有夹查询一律加 `f.user_id=?`。
- 修复 `",".join(字符串)` 按字符拼接导致画像标签变单字的 bug。
- 修复 rate 接口 `cf` 未导入的 NameError。

### 验证

①相似推荐 / ②混合推理（无规则+带规则+降级）/ ③主页ICF+GA（GA 0.5449）/ 新剧直读 / ④名场面命中(scene_focus)、硬条件落空话术、类似X排除本体 / rate / profile / 6 个静态页 全部 200 通过；visitor 与 demo 默认夹数据隔离验证正确；回归测试数据已清理。

## [2026-09-18] v1.3.1：四个口味画像演示账号 + ICF 信号修复

**新增演示账号（密码均为 123456，均带收藏夹+评分历史，登录即可看到算法效果）：**

| 账号 | 口味画像 | 收藏夹 | 金标准验证（主页算法③应推荐） |
|------|----------|--------|------------------------------|
| tom | 悬疑推理控 | 心理神作墙(1,7,5) / 补番中(19) | EVA / Re:0 / 寒蝉 ✓ |
| mike | 热血战斗党 | 热血神番(3,4,8) / 周末爆肝(9,11) | 银魂 / 灵能 ✓ |
| lily | 治愈日常系 | 治愈充电站(14,15,23) / 轻松下饭(17,24,21) | 你的名字 / 灵能 / 一拳 ✓ |
| jack | 科幻经典迷 | 科幻必看榜(12,13,5) / 老番补完(6,20) | L / 怪物 / 死亡笔记 ✓ |

设计原则：每个账号只评 10-11 部，**刻意留 2-4 部同口味作品不评分**，作为验证「算法能否猜中你喜欢什么」的金标准。

**算法修复：**

1. **cf.py 中心化方式修正**：`rating - 作品全局MAL分` → `rating - 该用户均分`（教科书式 mean-centered ICF）。原方式会把手松用户的口味信号中心化归零。
2. **种子数据孤儿评分修复**：ratings 表中 uid 7-62 的种子评分在 users 表无对应行（孤儿），新注册用户 id 自增到 7 起即与孤儿评分撞 UNIQUE(user_id, anime_id)，导致注册后写评分报 500。已为孤儿评分补建 `synth_07`~`synth_62` 合成用户行（保留其口味聚类信号）。
3. **新增 24 个强对比口味聚类用户**（`cluster_sus/hot/hea/sci 01-06`，各 6 人）：口味内作品评 8.3-9.9，口味外评 4.0-6.3，放大簇内 item-item 相似度。当前库：86 用户 / 1012 评分。

**验证：** 4 账号登录通过、错误密码 400；算法③主页推荐四账号口味对位全部命中金标准且数量 4-5 部；算法②收藏夹混合推理画像/推荐理由正常；算法①相似推荐正常；算法④聊天（含硬条件落空话术）正常。

## [2026-09-18] v1.4：六项体验修复（评分/收藏夹选择/标签下拉/登录卡/登出/登录门）

1. **评分静默失败修复（bug①根因）**：后端 /api/rate 一直正常；问题在前端 token 过期（历史清理过 sessions 表）后 onRate/toggleFav 无 catch，401 被吞掉、界面无任何反应。修复：onRate/toggleFav 全部 try/catch + toast(errMsg)；新增 checkAuth() 在每页入口校验 token，过期自动清凭据并回到登录门。
2. **收藏夹选择与移动（bug②）**：详情弹窗收藏按钮旁新增收藏夹下拉（默认夹+自定义夹，收藏时选择去处；已收藏时显示所在夹、切换即移动）；新增后端 `POST /api/favorites/move`（夹间移动）；收藏夹详情页每个条目卡片新增「移动到…」下拉。
3. **标签改选择（bug③）**：新增 `GET /api/tags`（全库 genre+theme）；收藏页算法②的规则筛选关键词输入框改为标签下拉（按 genre/theme 分组）；主页观看计划的关键词输入同步改为下拉。
4. **左下角登录卡 + 快速切换（bug④）**：侧栏底部改为用户登录卡（头像+用户名+▾），点击弹出面板：历史登录账号列表（localStorage af_accounts，最多8个）一键切换（切换前用 /api/auth/me 校验，过期自动移除）、登录其他账号、退出登录。
5. **登出（bug⑤）**：登录卡弹出面板内「退出登录」（调 /api/auth/logout + 清凭据 + 回登录门）。
6. **主界面登录门（bug⑥）**：主页/浏览/聊天/收藏/个人中心全部 requireAuth()；未登录时主界面整体变为登录/注册窗口（login/register 双 Tab），登录成功原地刷新进入；login.html 已登录则自动跳回主页。原演示用户选择器移除（demo/visitor 无密码不可登录，仅作数据保留）。

**验证：** /api/tags 15 genre+44 theme；收藏→指定夹→夹间移动→移回→取消收藏全链路通过；7 个前端脚本 node --check 全过；6 个静态页 200。

## [2026-09-18] v1.4.1：评分功能真根因修复（e.currentTarget 异步置空）

**症状：** 点星星后弹「请求失败，请重试」、星星不亮——但评分实际已写入数据库（个人中心历史里能看到）。

**根因（自 v1.2 起一直存在，v1.4 的 try/catch 修复只是把静默失败变成了报错提示，没碰到根因）：**

`onRate`（common.js）在 `await api("/api/rate",...)` 之后才访问 `e.currentTarget` 来更新星星。浏览器的事件对象在**派发结束（handler 在 await 处挂起）后会把 currentTarget 置为 null**，于是续延执行到 `e.currentTarget.querySelectorAll(...)` 时抛 TypeError，被 catch 后弹错误 toast——请求其实成功了。

**修复：** 在 await 之前把 `e.currentTarget` 捕获到局部变量 `starsEl`，await 之后用 `starsEl` 更新星星；`currentAnime.id` 同样提前捕获为 `animeId`。

**验证：**
- 后端 /api/rate 登录态实测通过（tom → anime 21 → 8.5 分写入 → 历史可查 → 已还原）。
- 用 Node 忠实模拟浏览器事件时序（handler 在 await 挂起 → 派发结束 currentTarget 置 null → 续延恢复）：修复前抛 TypeError，修复后正常返回。
- `node --check common.js` 语法通过。

**注意事项：** 前端是静态 JS，浏览器可能缓存旧版 common.js——**测试前请强制刷新（Ctrl+F5）**。

## [2026-09-30] AniLogic 完整重建（分支 anilogic-rebuild，替换 AnimeFinder v1.4.1）

**状态：** 新系统在模拟数据与真实数据（Kaggle MAL 2023 全量，23,363 部在库作品、2,396 万条评分，未抽样）上走通；旧模板已从本分支移除（历史保留在 `main` 与提交 `426db20`）。按重建提示词执行，未推送远程、未调用付费接口。

**变更内容：**

1. **新架构**（`app/`）：FastAPI 模块化单体 + SQLite（WAL、外键、Alembic 迁移含 FTS5）+ 服务端渲染页面骨架 + 原生 ES 模块前端。继续遵守本日志规则 6：一个页面 = `app/web/templates/<page>.html` + `app/web/static/js/pages/<page>.js`，共享逻辑 `js/common.js`，共享样式 `css/style.css`。
2. **账户与安全**：argon2id；HttpOnly 会话 cookie（库中只存哈希）+ CSRF 双提交；登录节流；无默认管理员；所有私有资源服务端校验归属（旧模板的 `user_id` 回退与越权读取已消除，见 `docs/legacy-feature-audit.md` §3）。
3. **四个核心算法**：①相似作品（加权五维属性 A 为默认，逻辑回归 B / BGE 属性向量 C / 融合 D 为对照）；②收藏夹规则层 + 偏好评分（严格为空时只给放宽建议，由用户选择）；③首页 ICF（共评分母的精确 adjusted cosine）召回 Top-20 + GA 列表优化 Top-10（MF / NCF / greedy / MMR / 穷举 oracle 对照，冷启动分 cold / partial / warm）；④聊天找番（确定性意图解析、多轮条件、FTS5 + BGE 混合检索 RRF、带来源回答，LLM 可选且经后验证，无 LLM 时本地模式）。
4. **观看计划**：时间预算、完整/部分模式、观看进度、前置作品、每日时段、保存/编辑/重新优化/导出（JSON / Markdown / HTML），每次返回与编辑都独立校验可行性。
5. **数据管线**（`pipelines/`）：模拟世界生成；真实数据 validate → preprocess → split → train → build-index → activate，版本化产物与 SHA-256 manifest，原子激活；可选 Jikan 离线补全（本次上游 MAL 大面积 504，仅 33 部取得 /full 数据）；补充数据（relations / characters / scenes / 帧图 manifest）按 `docs/data-contracts.md` 校验后入库。
6. **评估**（`evaluation/`）：9 个实验（similar、collection、homepage、explicit、ga_ablation、chat、planner、system、multimodal），每次运行独立目录（config / manifest / metrics / per_case / failures / comparison / report），`make export-evaluation` 汇总到 `docs/evaluation-results-{real,synthetic}.md`。
7. **本次修正的缺陷（均有回归测试或评估证据）**：收藏放宽请求带多余字段导致 422；编辑后的计划导出 500；关系文档超出 BGE 512 token 被截断；Jikan「已补全」统计把失败请求也算入；逻辑回归在仅 2 条弱标签时被标记为已训练；默认替换规则在 1 条验证查询上给出结论；ICF 对「全部打 9 分」的新用户预测全部相等（新增按验证集调参的用户均值先验 k，k = 0 即原公式）；冷启动纯标签画像几乎推不中（新增按验证集调参的热度权重 g）；真实模式补充数据只校验不入库；关系覆盖不足却显示 ready；聊天中仅标点不同的续作（如 Working'!!）与含逗号/感叹号的标题被截断识别错（标题识别 0.917 → 0.968）；评估自动查询误用被内容规则排除的类型。
8. **旧模板处理**：删除 `backend/*.py`、`backend/services/`、`frontend/`、含明文演示密码的 `account for test.txt`；`requirements.txt` 改为由 `uv.lock` 生成的锁定版本；`backend/anime.db`（未入库的本地用户数据）原样保留，供 `migrate-legacy` 使用。迁移 dry-run 结果见审计文档 §4（1 个注册账号；其 11 条评分符合旧种子回填时间模式，默认不迁移；`Hataraku Saibou` 的旧 MAL ID 35860 错误，迁移按唯一精确标题更正为 37141 并报告）。

**涉及文件：** `app/`、`pipelines/`、`evaluation/`、`configs/`、`migrations/`、`tests/`、`docs/`、`data/README.md`、`data/evaluation/`、`Makefile`、`Dockerfile`、`compose.yaml`、`.env.example`、`README.md`（重写）、`requirements.txt`（重新生成）；删除旧 `backend/`、`frontend/` 源码与 `account for test.txt`。

**原因：** 按《AniLogic 完整重建：Code Agent 实施提示词》与两份 Group 10 材料从零重建；旧模板存在越权与身份回退问题，且只在 24 部手写种子数据上运行，无法承载真实数据与评估要求。

**验证方式：** 单元 + 集成 131 项全部通过（含真实数据与 BGE 模型测试），浏览器旅程 8 个测试覆盖旅程 1–12 全部通过，ruff 无告警；详见 `docs/test-report.md`。真实数据 9 个实验结果见 `docs/evaluation-results-real.md`，要点与未决事项见 `docs/delivery-status.md`（例如：在该留出协议下热度基线 Hits@10 约为 ICF 的两倍，报告中如实呈现）。未验证：`docker compose up --build`（交付机器未安装 Docker）；Plain LLM vs RAG（未配置 LLM 凭证）；人工标注评估（协议见 `data/evaluation/human-evaluation-protocol.md`）。

**注意事项：**
- 演示账号只存在于模拟模式（`make seed`），密码 `anilogic-demo`，见 `docs/user-guide.md`；真实模式没有任何预置账号。
- 运行方式见 `README.md`；各功能状态见 `docs/delivery-status.md`。
- ICF 的 damping、用户均值先验 k 与首页冷启动热度权重 g 由 `make apply-tuning` 写入 `configs/<mode>.yaml`，注释中记录对应评估 run id。

## [2026-10-02] 算法②：五组模型比较、预注册选型与收藏夹接入（分支 yyc-algorithm2-selection）

**状态：** 按《AniLogic 算法②：测试、验证、模型选择与系统接入执行提示词》完成。正式运行 `real-main-01`（真实数据，代理标签）选出 B1（LightGBM 八特征），测试确认通过，网站真实模式默认评分器改为 B1；合成模式仍为 A0。未推送远程、未调用付费接口。

**变更内容：**

1. **算法② 包** `app/recommendation/algorithm2/`：只用 MAL id 的推荐目录（可服务、非露骨、有公开评分，14,182 部；theme / demographic 按 MAL id 并入 `algorithm1_benchmark/data/catalog.jsonl.gz`）与目录哈希；画像（频率 = 计数 / |H|，top-10 命名空间标签）；固定八特征；评分器 A0 / A1（公式）、B0 / B1（LightGBM）、C0（PyTorch 内容双塔）及旧评分器的辅助对照；模型包校验（模型 SHA-256、目录哈希、特征版本、编码器），不兼容时退回公式并记录状态。
2. **LightGBM 服务端求值** `gbdt_tree.py`：macOS 上 LightGBM（Homebrew libomp）与 PyTorch（自带 libomp）同进程会卡死或崩溃，服务改为用 NumPy 读取 LightGBM 文本模型；按位掩码逐树求值（QuickScorer 思路），原始分数与原生 LightGBM 逐位一致（freeze 时在 56 万行上核对，最大差 0），完整目录一次请求约 0.1 s。
3. **规则层** `constraints.py`：新增 `pool.scored`（无公开评分不进入推荐池）；关键词掩码缓存；`build_engine` 供服务与离线评估共用。
4. **收藏夹接口与页面**：从当前收藏读取 H，走与离线完全相同的代码路径得到 Top-8；严格结果不足不补齐，无结果只给放宽选项；放宽后响应带 `relaxed`、`relaxed_rules`、`strict_result_count`，页面显示原条件下的结果数；`scorer` 请求字段仅供实验 / 运维；降级提示不向普通用户暴露运行目录或异常。
5. **实验流程** `evaluation/algorithm2/`：audit / prepare / split / tune / train / freeze / evaluate-test / benchmark / verify-integration / report 与 run-all；用户级 70/15/15 切分（seed 42，按规模分层）、留一目标、R0–R3 严格规则、完整目录候选、1:4 共用弱负例、三个训练 seed、用户级配对 bootstrap；`selection_protocol.json` 在任何训练之前写入，冻结后不能再调参；LightGBM 与 PyTorch 分别在独立子进程中运行。
6. **运行记录**：`evaluation/algorithm2/runs/real-main-01`（24,000 用户，正式结论）与 `real-smoke-01`（1,200 用户，流程检查；该级别保留 A1，说明小样本结论不稳定）；大文件放在各运行的 `work/`（已加入 `.gitignore`）。
7. **配置与文档**：`configs/base.yaml`（scorer / algorithm2_run / supplement）、`configs/real.yaml`（B1 + 冻结运行）、`docs/algorithm2/`（README、仓库审计）、`docs/algorithms.md`、`docs/api.md`、`docs/delivery-status.md`、`docs/feature-traceability.md`、`docs/requirements-resolution.md`、`docs/evaluation-protocol.md`、`docs/evaluation-results-real.md`（生成器加入算法② 指引）、`README.md`、`Makefile`（`alg2-run` / `alg2-stage` / `alg2-fixture`）；`pyproject.toml` 的 `ml` extra 新增 `lightgbm`，`requirements.txt` 重新导出。

**结果（测试集 3,000 个主组用户，完整目录，R0，用户级 HR@8）：** A0 0.021、A1 0.032、B0 0.053、**B1 0.132**、C0 0.113（B、C 为三 seed 均值）。B1 相对 A0 +11.1 个百分点 [+9.8, +12.3]、相对 A1 +10.0 [+8.7, +11.2]，AUC 0.953，规则违反 0；B1 − B0 = +7.9 [+6.8, +9.0]，C0 − B1 = −1.9 [−3.0, −0.7]。完整报告：`evaluation/algorithm2/runs/real-main-01/final_report.md`。

**涉及文件：** `app/recommendation/algorithm2/`（新）、`app/recommendation/constraints.py`、`app/services/recommend.py`、`app/services/serving.py`、`app/schemas/requests.py`、`app/web/static/js/pages/collection.js`、`app/web/i18n/en.json`、`evaluation/algorithm2/`（新）、`evaluation/reports.py`、`configs/`、`tests/unit/test_algorithm2.py`（新）、`tests/integration/test_algorithm2_pipeline.py`（新）、`tests/e2e/test_journeys.py`、`docs/`、`README.md`、`Makefile`、`.gitignore`、`pyproject.toml`、`uv.lock`、`requirements.txt`。

**原因：** 原收藏夹默认评分器 `collection_profile_v1` 没有经过固定协议的比较，PPT 中的 GBDT / 双塔方案也没有在真实数据上训练与确认；本次按提示词用同一切分、候选、规则与评估器真实比较五组方案，并按预注册规则决定网站默认值。

**验证方式：** `pytest tests` 共 160 项全部通过（含新增算法② 测试 20 项与 8 个浏览器旅程），`make lint` 无告警；`verify-integration` 在真实数据库副本上 14 项检查全部通过（接口 Top-8 与离线预测 1,000 / 1,000 一致、收藏变化改变输出、严格不足不补齐、无结果不静默放宽、显式放宽契约、推荐 id 可打开详情页、模型包缺失或被改动时退回公式且筛选仍严格）；`evaluate-test` 重复执行结果完全一致。

**注意事项：**
- 标签是代理标签（评分 ≥ 8 的集合），应用数据库中还没有真实收藏，也没有线上实验，网站上的实际效果尚未验证。
- macOS 需要 `brew install libomp` 才能导入 LightGBM（只影响训练与离线评估；网站服务不加载原生 LightGBM）。
- 补记：提交 `fee53c9` 起，`app/core/config.py` 会从项目根目录 `.env` 读取 `ANILOGIC_*` 设置（进程环境变量优先），当时未写日志，在此补记。

## [2026-10-02] 算法② 实验材料集中到 `algorithm2_benchmark/`（与 `algorithm1_benchmark/` 同样的组织方式）

**变更内容：**
1. 实验流程代码 `evaluation/algorithm2/` → `algorithm2_benchmark/pipeline/`；运行记录 `evaluation/algorithm2/runs/` → `algorithm2_benchmark/results/`；说明与仓库审计 `docs/algorithm2/` → `algorithm2_benchmark/README.md`、`algorithm2_benchmark/docs/repo_audit.md`（`git mv`，保留历史）。
2. 入口改为 `python -m algorithm2_benchmark <stage> --run <id>`；新增 `run_checks.py`（运行算法② 测试并把实际输出写入 `results/tests.txt`）、`verify_delivery.py` 与 `DELIVERY_MANIFEST.json`（交付文件 SHA-256 清单）、`.gitattributes`（禁止换行符转换，保证清单按字节校验）。
3. 测试按保护对象拆分：实验自身的测试（指标定义、fixture 全流程）移到 `algorithm2_benchmark/tests/`（13 项）；保护网站行为的测试留在主测试集，改名为 `tests/unit/test_collection_algorithm2.py`（6 项）与 `tests/integration/test_collection_algorithm2.py`（2 项）。
4. `configs/real.yaml` 的 `algorithm2_run` 改为 `algorithm2_benchmark/results/real-main-01`；Makefile（`alg2-run` / `alg2-stage` / `alg2-fixture`，新增 `alg2-checks`、`alg2-verify`，`make lint` 包含新文件夹）、`.gitignore`、`evaluation/reports.py`、README 与各文档的路径同步更新；两次运行的报告重新生成。
5. 修正：报告与接入阶段在没有 `work/`（例如新克隆的仓库）时读取运行目录中已发布的 query / pair 清单；复现说明里的运行路径改为相对仓库根目录计算。
6. 网站运行用的代码 `app/recommendation/algorithm2/` 位置不变（网站启动时导入）。

**涉及文件：** `algorithm2_benchmark/`（新）、`evaluation/algorithm2/` 与 `docs/algorithm2/`（移走）、`tests/unit/test_collection_algorithm2.py`、`tests/integration/test_collection_algorithm2.py`、`configs/base.yaml`、`configs/real.yaml`、`Makefile`、`.gitignore`、`evaluation/reports.py`、`app/recommendation/algorithm2/__init__.py`、`README.md`、`docs/`。

**原因：** 按用户要求，像算法① 一样把算法② 的尝试过程、测试与结论集中在一个单独的文件夹，方便查找。

**验证方式：** `pytest tests` 148 项通过；`algorithm2_benchmark/run_checks.py` 21 项通过；`make lint` 无告警；本地真实模式网站从新路径加载冻结模型（`collection_scorer: ready`，默认 B1）；`verify_delivery.py` 核对通过。

**注意事项：** 两次运行在整理之前完成，运行记录（`run_metadata.json`、`audit_facts.json`、`freeze_manifest.json`）里的 `evaluation/algorithm2/` 是当时的代码位置，作为历史记录保持原样。

## [2026-10-03] 算法④：聊天找番（LLM + 知识库 + RAG）评测包 `algorithm4_benchmark/`（分支 algorithm-4-benchmark）

**变更内容：**
1. 新增 `algorithm4_benchmark/`（263 个文件，约 3.0 MB，纯标准库 + unittest，出图可选 matplotlib；离线可复现，无需 app / 数据库服务 / LFS）：80 条查询 × 6 个 LLM × {R：检索增强，N：无检索，T：模板}；3 个 LLM 评审（按各评审自己的均值居中后的分数）、按问题重采样的 bootstrap 置信区间、按题型分层。表 1–2、`scored.csv` 和评审聚合由包内代码从保存的原始回复与评审输出离线重算；其余表和图以保存下来的输出随包提供。
2. 被评测对象是 AnimeFinder v1.4.1 页④的聊天管线（`backend/services/rag.py`，git blob `3b5f6dc`，提交 `426db20`），冻结在 `algo/rag_snapshot.py`，知识库为它的 24 部作品 + 12 条名场面种子数据。本次不包含 AniLogic 的 `app/chat`：方法（R/N/T 三种条件、评分、置信区间）可以迁移，数字不能。
3. 新增 `algorithm4_benchmark/.gitattributes`（`* -text`，保持字节不变；包内有逐文件 SHA-256 清单和逐字节比对的测试）与 `.gitignore`（根目录的 `*.db` 规则会忽略 `data/kb/anime.db`，这里放行；并忽略脚本生成的 `results/replay/`）。
4. `README.md` 目录表、`docs/delivery-status.md` 第 4 节、`docs/evaluation-protocol.md` 的 Plain LLM vs RAG 一行各加一句指向该目录的说明。

**涉及文件：** `algorithm4_benchmark/`（新）、`README.md`、`docs/delivery-status.md`、`docs/evaluation-protocol.md`、`DEVLOG.md`。未改 `app/`、`evaluation/`、`pipelines/`、Makefile、根目录 `.gitignore` / `.gitattributes`。

**原因：** `docs/delivery-status.md` 与 `docs/evaluation-protocol.md` 记录 Plain LLM vs RAG 对比“尚未实现”；本包用 AnimeFinder v1.4.1 的管线给出该对比的方法与结果，针对 AniLogic `app/chat` 的同类对比仍待实现。

**验证方式：** 在 `algorithm4_benchmark/` 下运行 `python -m unittest discover -s tests`（100 项通过，约 2 分钟）与 `python tools/presubmit_check.py`（无告警）；`MANIFEST.txt` 的逐文件 SHA-256 与磁盘一致；提交内容与测试时的文件逐字节相同。

**注意事项：**
- 单次运行；LLM 评审没有人工标注；6 个模型通过 CLI 通道调用，延迟不是端到端结果；知识库只含 24 部知名作品，存在天花板效应（均已写入 `algorithm4_benchmark/REPORT.md`）。
- 只在 Windows 11 + Python 3.10 上测过。
- `algorithm4_benchmark/` 下的 `.md` 全部为英文（包内测试限制）。
- `data/kb/anime.db` 只保留 `anime` 与 `scenes` 两张表，行内容不变。

## [2026-10-04] 接入团队数据包（补充层）+ 聊天年份 / 工作室 / 场景能力 + 相似作品默认改为 B（分支 yyc-data-package）

**状态：** 真实数据新版本 `real-034ebd6a74` 已激活；算法② 冻结模型 `real-main-01`（默认 B1）在新版本上校验通过，目录哈希不变（`24c83192…`）。全部评估在本地模式下运行（`ANILOGIC_LLM__PROVIDER=none`），没有调用付费模型接口；未推送远程。

**变更内容：**

1. **数据包导入** `pipelines/package_import.py`（新）与命令 `python -m pipelines.cli import-package`、`import-similarity-labels`：只读取 `DATASET_INDEX.json`、`data/canonical/anime_all.jsonl`、`data/rag/scenes_verified.jsonl`、`data/rag/evaluation_queries.jsonl`，**从不读取含用户名和性别的 `users.csv.gz`**（有单元测试检查）。写出 `data/enriched/{tags,synopses,relations,characters,scenes}.jsonl`、`data/evaluation/rag_queries.jsonl`（`label_source = team_package_<版本>`）和导入报告 `data/enriched/team_package_import.json`（版本、文件哈希、数量、规则）。类型、工作室、集数、时长、评分**不导入**，仍以 2023 快照为准；受众每部作品只保留字母序第一个（与算法① 投影规则一致），因此算法② 的目录哈希保持不变。只有 MAL 身份已确认的记录被使用。
2. **预处理**（`REAL_PREPROCESS_VERSION = 3`）：合并 `tags.jsonl` / `synopses.jsonl` 并逐值记录来源；可推荐规则改为"有评分，且有类型或主题"（12,429 → 13,798 部）；`missing_fields` 增加 themes。数据契约新增 tags / synopses / similarity_labels_mal；`rag_queries` 的 intent 允许为空，`expected_outcome` 做枚举校验。
3. **数据结果**：主题覆盖 14,242 部、受众 10,395 部、角色表 11,494 部、关系 19,478 条（10,883 部作品，之前 33 部）、简介缺失率 18.2 % → 14.8 %、场景 300 条（100 部作品）；知识库文档 42,515 → 64,914。
4. **算法①**：389 条 MAL 用户推荐弱标签（保留基准预先登记的 train / validation / test 划分，并报告跨划分共享系列的查询数）。替换规则首次能做出判断：验证集上 B（逻辑回归）与 C（BGE 属性向量）都通过且两者打平（C − B +1.0 [−6.9, +8.8]），**真实模式默认改为 B**（`configs/real.yaml` 的 `recommendation.similar.scorer`），因为 B 用的就是 A 的可解释特征，推荐理由不变；聊天里的"类似 X"也改用同一评分器。状态页显示实际默认值，配置的评分器不可用时降级并回退到 A。
5. **约束引擎与聊天解析**：新增开播年份 `year_from` / `year_to`（年份未知不满足条件）与评分上限 `max_score`；解析器支持年份（英文、中文、年代、范围；剧情中的年代、年龄、计数不当作年份）、两位数评分和句末句号、"no more than"、"TV anime"、工作室（强提示后找不到的工作室会如实说明）、数据中没有的信息（视频时间码、评分日期、版权费、播放平台、票房、"收藏夹"）以及"哪部动画有<剧情>"的识别类问题；"I would like" 不再当作"like X"，短的小写词组和类型词不再当作作品名，剧情问题里引号中的昵称不再当作作品名。没有结果时说明是哪个条件排除了全部作品，并给出目录的实际范围。
6. **场景检索**：场景记录与简介在同一个全局排名中比较，只有排第一的作品由一条在两个检索通道中都排进全局前 3 的场景记录支撑时，才回答"verified scene record"；否则交替列出剧情层面的匹配和最接近的场景记录，并说明不确定；点名作品时也按全局排名判断。修正：集数显示为整数（之前是"episode 1.0"）、电影不再说"第 1 集"、证据按检索排名排序。
7. **关系与观看计划**：前传只在不晚于本作开播时才算"需先看"（按播出顺序）；较晚推出的前传（如《咒术回战 0》《游戏人生 Zero》）单独列出。热门 500 部中有 28 部受此影响。问续作时先列直接续作；没有记录时明确说"没有记录到直接续作（不代表没有）"。详情页显示"较晚推出的前传"。
8. **发现页与搜索接口**：增加开播年份筛选；聊天的放宽按钮支持"any year"。
9. **评估**：聊天实验按行内 `label_source` 命名标注集并按类别拆分；按期望结果计算任务成功（matches / no_match / insufficient_information）；新增场景归因 scene@1 / scene@3、场景语料消融，以及两组自动场景题（场景库外的作品、点名作品但描述不符）；续作题改为随机抽样且任一直接续作都算对；检测"编造集数"前先去掉本身含"Episode N"的作品名。运行记录新增 `git_dirty` 字段。

**结果（真实数据，本地模式）：**
- 团队 100 道题：场景 40/40（全部引用到对应场景记录）、语义 30/30、约束 20/20、无法回答 10/10。场景题只考"在 300 条场景里找对那一条"；场景库外的 60 道剧情题任务成功 0.97，**没有一次**把别的作品说成"已验证场景"（修改场景规则之前是 13/60）；点名作品但描述不符的 40 题没有一次声称场景。只用简介时场景题前 5 命中 0.43。
- 相似作品（验证集 102 条）：A 0.049、B 0.167、C 0.176；测试集（只报告）A 0.081、B 0.197、C 0.162，B − A +11.5 [+7.3, +16.2]。
- 首页：ICF Hits@10 0.062，热度基线 0.137（与之前一致）；GA 相对贪心 −1.5 个百分点（之前 −0.9），原因是主题变全后"新鲜度"项有了更多信号；验证集消融中新颖度项和新鲜度项各占约三分之一个百分点，MMR(0.7) 准确率 0.040 且多样性更高。调参在新版本上复核后取值不变（20 / 5 / 0.9）。
- 观看计划 0 个不可行，GA 在小池上 98.7 % 达到精确最优；系统故障注入全部正确回退。

**涉及文件：** `pipelines/package_import.py`（新）、`pipelines/cli.py`、`pipelines/real_ingest.py`、`pipelines/validate_data.py`、`pipelines/fit.py`、`app/chat/intent.py`、`app/chat/conversation.py`、`app/services/chat.py`、`app/services/catalog.py`、`app/services/plans.py`、`app/services/serving.py`、`app/recommendation/constraints.py`、`app/recommendation/similarity/logistic.py`、`app/graph/relations.py`、`app/retrieval/{evidence,documents,titles}.py`、`app/api/v1/anime.py`、`app/web/`（发现页、聊天页、详情页、i18n）、`evaluation/experiments/chat.py`、`evaluation/common.py`、`evaluation/reports.py`、`configs/real.yaml`、`tests/`（新增 `test_package_import.py`、`test_chat_years_studios.py` 等）、`docs/`、`README.md`、`data/enriched/`、`data/evaluation/`、新版本数据与模型、`var/real/anilogic.sqlite3`、`evaluation/runs/`。

**原因：** 按上一轮给出的推荐方案，接入团队在原始数据基础上整理的数据包（补充主题、受众、关系、角色、简介和带来源的场景，并提供评估题），同时保持现有架构、数据契约和评估协议；新数据暴露出的问题（场景误认、晚出前传被当作"需先看"、相似作品 A 的稀疏元数据偏差）一并修正。

**验证方式：** 单元 + 集成 + 真实数据 + 浏览器旅程测试全部通过（数量见 `docs/test-report.md`），`ruff` 无告警；真实数据 9 个实验与合成数据全部实验已重跑，结果见 `docs/evaluation-results-{real,synthetic}.md` 与 `docs/delivery-status.md`。

**注意事项：**
- 场景由 AI 代理按引用来源核对（核对人、日期和方法记录在每条 `source_ref` 中），不是人工核对，建议人工抽查。
- 团队的语义 / 约束题把所有满足筛选条件的作品都算作相关，这两类的 Hits@5 主要衡量约束处理，不是人工相关性判断。
- 项目根目录的 `.env` 配置了 LLM，本地网站的聊天会调用它；本次所有评估都在本地模式下运行，没有产生费用。
- 本次评估运行时代码尚未提交，运行记录中的 `git_commit` 是 `e9df5bf`；对应代码见本条记录所在的提交。之后的运行会在 `git_dirty` 字段中注明。
- 回退：把 `configs/real.yaml` 的 `recommendation.similar.scorer` 改为 `weighted_attributes` 即恢复 A；`python -m pipelines.cli activate --mode real --version real-89bfdea00b` 可回到导入前的数据版本（该版本目录保留在本地）。

## [2026-10-19] 系统运行数据统一放进 SQLite（训练数据与模型权重除外）（分支 yyc-sqlite-serving-data）

**状态：** 本地 real（`real-034ebd6a74`）与 synthetic（`synthetic-7a7d5d6a74`）数据库已按新结构重新激活；未提交、未推送。

**变更内容：**

1. **迁移 `0003`**：`anime` 增加 `catalog_position`（目录序号，即模型使用的行号；唯一索引）、`train_interactions`（训练集评分数）、`exclusion_reasons`；新表 `anime_embeddings`、`document_embeddings`、`scene_image_embeddings`（每行一个向量，float32 原始字节）、`vector_sets`（维度、类型、数量、来源信息与整块矩阵的 SHA-256）、`supplement_tags`（算法② 的主题 / 受众补充表）。迁移只用 `ALTER TABLE ADD COLUMN`，不重建 `anime`（它是用户数据的外键父表）。
2. **激活** `pipelines/activate.py`：写入**整个目录**（未上架的露骨作品以 `in_catalog = 0` 保存，用于模型行号对齐，接口仍只返回可服务作品）及其标签、别名、训练计数；属性向量、文档向量、截图向量；real 模式下的 `algorithm1_benchmark/data/catalog.jsonl.gz` 补充表（来源路径与 SHA-256 记在激活记录里）。仍在同一个事务里完成。
3. **服务** `app/services/serving.py`：目录快照、标签矩阵、别名、训练计数、三组向量和算法② 补充表全部从数据库读取，向量按版本、行数和 SHA-256 校验，不一致时该能力报 `artifact_mismatch`；只有模型权重（`models/`、LSA 编码器）仍从 artifact 文件读取并按 manifest 校验。`build_service` 增加 `supplement` 参数，离线实验仍读文件。
4. **测试**：新增数据库读取与 artifact 文件逐项一致、移走目录 / 向量文件后服务仍正常、篡改数据库向量被拒绝三项检查。

**结果：** real 模式快照（键顺序、数值、四个标志、五个标签矩阵、封面、英文名）、训练计数、属性向量、文档向量与 artifact 文件完全一致；算法② 冻结模型 `real-main-01` 校验通过（目录哈希不变，默认 B1）。real 数据库 164 MB → 362 MB。

**涉及文件：** `migrations/versions/0003_serving_data_in_db.py`（新）、`app/db/models.py`、`app/db/vectors.py`（新）、`pipelines/activate.py`、`app/services/serving.py`、`app/recommendation/algorithm2/service.py`、`tests/integration/test_pipeline_integrity.py`、`docs/architecture.md`、`docs/operations.md`、`var/{real,synthetic}/anilogic.sqlite3`。

**原因：** 按用户要求，除训练数据和模型权重外，系统运行需要的数据（RAG 文档与向量、相似度向量、目录与标签、算法② 补充表）都放在数据库里，并去掉 parquet 与数据库之间的重复。

**验证方式：** `pytest tests/unit tests/integration` 179 项通过；`pytest -m "real or ml"` 通过；`make lint` 无告警；real 模式 `load_state` 各能力状态与改动前一致。

**注意事项：**
- 旧结构的数据库（迁移 `0003` 之前激活的）需要重新执行一次 `python -m pipelines.cli activate --mode <mode>`，否则状态页显示 `artifact_mismatch`。
- 本分支合并了 `origin/main`（截至 `51c55db`）：相似作品按 main 的决定在两种模式下都默认使用期末报告路线 A（`algorithm1_a`），去掉了 2026-10-04 条目里 `configs/real.yaml` 改为 B 的设置（B 仍可通过 `scorer=logistic_similarity` 选用）；配置的评分器不可用时，状态页与聊天都退回 `algorithm1_a`。
- artifact 中的 `catalog/`、`embeddings/`、`index/`、`stats/` 文件仍保留：它们是激活的输入，离线评估和 `algorithm2_benchmark` 也直接读取它们。

## [2026-10-19] 本地小组名称更正为 Group 18

**变更内容：** 将小组显示配置及代码默认值统一改为 `Group 18`，同步更新 README 和需求决策说明。页面页脚与系统状态接口通过现有共享配置显示正确的小组名称。

**涉及文件：** `configs/base.yaml`、`app/core/config.py`、`README.md`、`docs/requirements-resolution.md`、`DEVLOG.md`。

**原因：** 团队确认实际为 Group 18；仅修改本地项目，不上传 GitHub。

**验证方式：** 重启本地服务，检查 `/api/v1/system/status` 的 `app.group` 和 `/discover` 页脚，确认返回或显示 `Group 18`，并检查健康接口。

## [2026-10-19] 配置本地 ModelScope LLM

**变更内容：** 在本地 `.env` 中配置 `openai_compatible` 提供方、ModelScope 推理服务地址、`deepseek-ai/DeepSeek-V4.1-Flash` 模型和用户提供的 API 密钥，保留真实数据模式。

**涉及文件：** `.env`（Git 忽略的本地文件）、`DEVLOG.md`。

**原因：** 按用户要求仅更新本地 LLM 配置；密钥不写入日志，不上传 GitHub。

**验证方式：** 检查配置加载及密钥存在性，重启服务并检查健康接口和 LLM 配置状态；不调用外部推理接口。

## [2026-10-19] 更新 GitHub 最新 main 并运行 Group 18 本地版本

**变更内容：** 快进更新至 `9dfe8ed`，重新应用本地 Group 18 配置和文档更正，保留以上本地变更历史。用仓库的 `.env.example` 替换旧 `.env`，不再沿用本地 API 配置；当前 main 未跟踪 `.env`，默认使用本地聊天模式。

**数据处理：** 更新前以 SQLite 备份 API 保存原业务数据库至 `var/local/backups/before-9dfe8ed-20261019.sqlite3`，本地受版本控制的改动另存 Git stash。恢复原业务数据库后，使用新版本的激活流程迁移到 `0003` 并激活 `real-034ebd6a74`，保留账号、收藏、评分和聊天记录。

**涉及文件：** `configs/base.yaml`、`app/core/config.py`、`README.md`、`docs/requirements-resolution.md`、`DEVLOG.md`、`.env`、`var/real/anilogic.sqlite3`、`artifacts/real/active.json`。

**原因：** 按用户要求拉取最新项目、改为 Group 18 并运行；所有定制仅保留本地，不上传 GitHub。

**验证方式：** 核对 Git 版本、用户数据与备份逐表一致、数据库完整性及迁移版本；启动服务后检查健康接口、Group 18 页脚、搜索、相似推荐、语义检索及新版场景数据。

## [2026-10-19] 本地配置 DeepSeek V4.1 Flash 并精简首页卡片

**变更内容：** 在本地 `.env` 中配置用户提供的 API 密钥，使用 ModelScope 的 OpenAI 兼容接口及 `deepseek-ai/DeepSeek-V4.1-Flash`。首页 Recommended for you 与 Explore 的卡片统一为发现页样式，仅显示封面、标题、基本信息和收藏按钮，去掉算法来源标签、推荐理由及额外操作。

**涉及文件：** `.env`（Git 忽略的本地文件）、`app/web/static/js/pages/home.js`、`DEVLOG.md`。

**原因：** 按用户要求选择 DeepSeek V4.1 Flash，并将首页两处卡片调整为参考截图的展示；只修改本地项目，不上传 GitHub。

**验证方式：** 检查 JavaScript 语法、配置加载及密钥存在性；重启服务后检查健康接口、Group 18 与 LLM 配置状态；使用带算法标签和推荐理由的响应检查桌面及手机视口下两处卡片的显示、收藏和详情入口。未调用外部推理接口，密钥不写入日志。

## [2026-10-19] 按最新授权将本地定制发布到 GitHub main

**变更内容：** 按用户本次授权，将此前仅保留本地的 Group 18 配置、文档更正及首页卡片精简一并提交到 `main`。更新 `.env.example` 的非敏感示例为 ModelScope / DeepSeek V4.1 Flash，并在 README 中说明本地密钥配置。此前日志中的“不上传”描述的是当时的操作范围，本次以最新授权为准。

**涉及文件：** `.env.example`、`app/core/config.py`、`app/web/static/js/pages/home.js`、`configs/base.yaml`、`README.md`、`docs/requirements-resolution.md`、`DEVLOG.md`。

**原因：** 用户要求整体更新 GitHub 的 `main`。实际 `.env` 与 API 密钥不提交；本机数据库中的账号、评分、收藏和聊天记录，以及仅有激活时间和本机路径差异的 `artifacts/real/active.json` 保留本地。

**验证方式：** 核对远端 `main`、仓库写入权限、提交文件范围和密钥忽略规则；检查配置加载、JavaScript 语法、差异格式及服务健康状态。首页已使用包含完整推荐理由的响应通过 1440px / 390px Playwright 检查，封面、收藏和详情入口正常且无横向溢出；提交前复查暂存内容不包含实际密钥，推送后核对远端提交。
