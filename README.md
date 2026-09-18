# AnimeFinder · 智能动漫推荐与发现系统

课程项目：一个纯前端 + FastAPI + SQLite 的智能动漫推荐与发现系统，**四大核心算法一体化、一个算法一个文件、界面可见算法标签**。

> 当前版本 v1.4.1（完整迭代记录见 [DEVLOG.md](DEVLOG.md)）

## 一、功能总览

### 1. 账号体系（v1.4 登录门）
- 注册 / 登录 / 登出，密码 PBKDF2-HMAC-SHA256 哈希存储，会话用随机 token
- **主界面即登录门**：主页 / 浏览 / 聊天 / 收藏 / 个人中心全部需要登录，未登录时整体呈现登录/注册窗口（双 Tab），登录成功原地进入
- **左下角登录卡**：头像 + 用户名 + 下拉面板，支持历史登录账号一键快速切换（最多记 8 个）、切换前自动校验 token、退出登录
- token 过期自动清理凭据并回到登录门，杜绝"静默失败"

### 2. 主页 · 核心推荐（算法③ ICF + GA）
- 「为你推荐」：基于**你的真实评分**做 Item-based 协同过滤召回 → 遗传算法对整张列表做最优重排
- 「新剧速递」：直接读库按开播时间展示
- 「观看计划」：按标签（下拉选择）+ MAL 分区间生成片单
- 冷启动：无评分/少评分时由人气分兜底，GA 的多样性(D)与新颖度(N)项依然生效

### 3. 动漫详情页 · 相似推荐（算法① Knowledge-Based）
- 五维加权相似度：`Similarity = 0.35·Genre + 0.30·Theme + 0.15·Demographic + 0.10·Studio + 0.10·Length`
- 附轻量知识图谱（Anime / Character / Studio / Franchise 四类节点）
- 详情弹窗内可直接**打分**（1-10 星）、**收藏**（可选目标收藏夹）

### 4. 我的收藏 · 收藏夹体系（算法② Rule + Knowledge 混合）
- 多收藏夹管理：默认夹 + 自定义夹（新建 / 删除 / 夹间移动），收藏时可下拉选择去处
- **基于收藏夹的混合推理推荐**：
  - 规则层：关键词 / 最低 MAL 分 / 集数上限 等硬约束筛选（关键词为标签下拉选择）
  - 知识层：用夹内作品的 genre/theme 建立偏好画像，对候选打分排序 `match_score = MAL分/10 + 标签命中数 × 0.15`
  - 规则层筛空时自动降级为纯知识画像排序（relaxed 标记），保证永远有结果

### 5. 聊天找番 · RAG（算法④）
- 自然语言找番：「想看一部短一点的悬疑番」这类口语输入 → 检索知识库 → 回复
- **名场面回忆**：命中名场面时返回结构化 scene_focus，以「回忆片段」对话气泡渲染，与推荐卡片刻意区分
- 检索不到结果时有兜底话术；「类似 X」查询会自动排除 X 本体
- 离线可用（检索 + 模板生成），可一行配置切换 LLM 生成（见下文）

### 6. 个人中心
- 口味画像（根据收藏夹 + 评分历史聚合的标签）
- 完整历史评分列表、全部收藏条目

### 7. 其他体验细节
- 全站 Toast 错误提示，评分 / 收藏失败不再静默
- 探索页支持按标签筛选浏览全部番剧
- 评分修复（v1.4.1）：解决了 `e.currentTarget` 在 await 后被浏览器置空导致的假报错，评分写入与星星点亮全链路正常

## 二、四大算法与文件对位

| 界面位置 | 算法 | 文件 | 核心方法 |
|----------|------|------|----------|
| 详情页「相似推荐」 | ① 基于知识的推理 | `backend/services/knowledge.py` | 五维标签 Jaccard 加权相似度 + 轻量知识图谱 |
| 收藏夹页「混合推理」 | ② 规则 + 知识混合 | `backend/services/hybrid.py` | 规则层硬筛 × 知识层画像打分，筛空自动降级 |
| 主页「为你推荐」 | ③ ICF + GA 融合（核心） | `backend/services/icf_ga_fusion.py`（编排）+ `cf.py` + `ga.py` | ICF 以**用户均分中心化**的 item-item 相似度召回 Top-20 → GA 以 Fitness=0.4·Relevance+0.3·Personalization+0.2·Diversity+0.1·Novelty 重排 |
| 聊天找番页 | ④ Cognitive LLM + KB + RAG | `backend/services/rag.py` | TF-IDF 思想词面检索 + 模板/LLM 生成，名场面结构化输出 |

> 算法③说明：ICF 建相似度矩阵用的是**全库评分**，查推荐用的是**你的个人高分种子（≥8 分）**；评分既参与建矩阵、又是查你口味的钥匙。

## 三、技术栈

| 层 | 选型 |
|----|------|
| 后端 | Python 3.12 + FastAPI + SQLite（零配置，自带种子数据） |
| 前端 | 原生 HTML/CSS/JS（无框架，一页一 JS 文件） |
| 认证 | PBKDF2-HMAC-SHA256 密码哈希 + 随机 token 会话（`services/auth.py`） |
| 数据 | MyAnimeList 风格种子库 + 86 用户 / 1012 条评分（含 24 个强对比口味聚类用户） |



```bash
conda activate animedev
cd backend
uvicorn main:app --reload --port 8000
```

打开 http://localhost:8000 即可使用；接口文档（Swagger）在 http://localhost:8000/docs。

> 注意：前端是静态 JS，浏览器可能缓存旧版 —— 更新后请 **Ctrl+F5 强制刷新**。

> 也可在 PyCharm 中把解释器指向 `E:\anaconda3\envs\animedev\python.exe`，然后直接运行 `main.py`。

## 演示账号

每个账号 10-11 条评分 + 4~6 部收藏，**刻意留了同口味作品不评分**，作为验证「算法能否猜中你喜欢什么」的金标准（四个账号实测全部命中）：

| 账号 | 口味画像 | 收藏夹 | 登录后主页算法③会推荐（实测） |
|------|----------|--------|------------------------------|
| `tom` | 悬疑推理控 | 心理神作墙 / 补番中 | EVA、寒蝉、Re:0 |
| `mike` | 热血战斗党 | 热血神番 / 周末爆肝 | 银魂、灵能百分百 |
| `lily` | 治愈日常系 | 治愈充电站 / 轻松下饭 | 你的名字、灵能、一拳 |
| `jack` | 科幻经典迷 | 科幻必看榜 / 老番补完 | L、怪物、死亡笔记 |

切换不同账号登录，主页推荐结果会随口味明显变化——这是展示算法个性化的最直接方式。

## 六、主要接口一览

| 分类 | 接口 | 说明 |
|------|------|------|
| 认证 | `POST /api/auth/register` `/login` `/logout`、`GET /api/auth/me` | 注册 / 登录 / 登出 / 当前用户 |
| 番剧 | `GET /api/anime/list`、`GET /api/anime/{id}`、`GET /api/anime/{id}/similar`、`GET /api/anime/{id}/kg` | 列表 / 详情 / 算法①相似 / 知识图谱 |
| 推荐 | `GET /api/recommend/home`、`GET /api/recommend/new`、`POST /api/watchplan` | 算法③ / 新剧速递 / 观看计划 |
| 收藏 | `GET /api/user/{id}/favorites`、`GET /api/user/{id}/folders`、`POST /api/folders`、`DELETE /api/folders/{id}`、`GET /api/folders/{id}`、`GET /api/folders/{id}/recommend`、`POST /api/favorite`、`POST /api/favorites/move`、`POST /api/favorites/remove` | 收藏夹 CRUD / 算法② / 收藏与移夹 |
| 互动 | `POST /api/rate`、`GET /api/user/{id}/ratings`、`GET /api/user/{id}/profile`、`GET /api/tags` | 打分 / 历史 / 个人中心 / 标签下拉 |
| 聊天 | `POST /api/chat` | 算法④ RAG |

## 七、可选升级

### 接入真实 MyAnimeList 数据集（Kaggle dbdmobile/myanimelist-dataset）
`backend/database.py` 已预留接口 `load_mal_dataset(anime_csv, score_csv, details_csv)`，docstring 写明了列映射与注意事项（过滤无评分/成人内容、分块读取、Jikan API 补 themes/demographics/characters）。实现后删除 `anime.db` 重启即可无缝替换种子数据。

### 开启 LLM 生成（默认离线模板回复）
```bash
export LLM_API_BASE=https://api.deepseek.com/v1
export LLM_API_KEY=sk-xxx
export LLM_MODEL=deepseek-chat
```

### 语义检索升级
`services/rag.py` 的 `_score()` 是演示级词面匹配，替换为 `sentence-transformers(BGE) + ChromaDB` 只需改这一个函数，其余接口不变。

## 八、项目结构

```
anime-recommender/
├── backend/
│   ├── main.py              # FastAPI 入口 + 全部路由
│   ├── database.py          # SQLite 建表 + 种子数据（users/sessions/ratings/favorites/fav_folders…）
│   ├── jikan_enrich.py      # Jikan API 数据补全
│   ├── load_mal.py          # MAL 数据集接入脚本
│   └── services/
│       ├── knowledge.py     # 算法①：知识推理（详情页）
│       ├── hybrid.py        # 算法②：规则×知识混合（收藏页）
│       ├── icf_ga_fusion.py # 算法③：ICF+GA 编排（主页，核心）
│       ├── cf.py            #   算法③子模块：Item-based CF（均分中心化）
│       ├── ga.py            #   算法③子模块：遗传算法重排
│       ├── rag.py           # 算法④：RAG 聊天
│       ├── auth.py          # 注册登录（PBKDF2 + token）
│       └── watchplan.py     # 观看计划
├── frontend/
│   ├── index/explore/chat/favorites/profile/login.html
│   ├── css/
│   └── js/                  # 一页一 JS：home/explore/chat/favorites/profile/login/common
├── requirements.txt
├── DEVLOG.md                # 完整开发迭代记录
└── README.md
```
