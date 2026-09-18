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
