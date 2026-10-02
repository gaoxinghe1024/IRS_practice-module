# 算法② 仓库核验（repo audit）

核验时间：2026-10-02。所有数字均为查询或统计得到，不来自 PPT。机器可读的核验结果由
`python -m evaluation.algorithm2.cli audit --run <run_id>` 写入运行目录（`audit_facts.json`）。

## 1. 仓库与分支

| 项目 | 值 |
|---|---|
| 仓库 | `origin` = https://github.com/gaoxinghe1024/IRS_practice-module.git |
| 目标分支 | `yyc-algorithm2-selection`（用户指定名称；从 `origin/main` 新建并已推送到 origin） |
| 起始提交 | `3cda03a`（Merge pull request #4 from gaoxinghe1024/yyc-v1） |
| 远程核验 | 已执行 `git fetch origin --prune`；`origin/main` = `3cda03a`，与起始提交一致 |
| 额外本地改动 | `var/real/anilogic.sqlite3`、`var/synthetic/anilogic.sqlite3`：本机运行中的服务写入的运行时数据库，与本任务无关，保留为未提交改动，不会提交 |
| 项目约定 | `DEVLOG.md`（每次改动追加带日期记录；前端一个页面 = 一个 HTML + 同名页面 JS）；仓库中没有其他代理说明文件（如 AGENTS.md） |

## 2. 算法② 的实际代码位置

| 环节 | 文件 / 函数 |
|---|---|
| 收藏夹页面 | `app/web/templates/collection.html`、`app/web/static/js/pages/collection.js`（`loadRecs`、放宽按钮） |
| API | `app/api/v1/collections.py::recommendations` → `POST /api/v1/collections/{id}/recommendations`（请求体 `CollectionRecRequest`：keywords、include_tags、min_score、max_episodes、types、k、scorer、relaxation_applied） |
| 规则筛选 | `app/recommendation/constraints.py::ConstraintEngine.evaluate`（关键词、评分、集数、类型、标签、去重、账户排除）；`suggest_relaxations`（放宽建议） |
| 偏好画像 | `app/recommendation/profiles.py::build_collection_profile`、`affinity` |
| 评分 | `app/recommendation/collection/manual.py`：`CollectionProfileScorer`（id `collection_profile_v1`，**当前默认**）、`ManualSlidesScorer`（id `manual_slides_v0`，PPT 公式）；`gradient_boosting.py`（sklearn HistGB，`gradient_boosting_v1`）；`two_tower.py`（`two_tower_v1`） |
| 排序 / Top-K | `app/services/recommend.py::collection`：`stable_order(scores, item_key)`，取前 `k`（默认 8，`configs/base.yaml: recommendation.collection.k`） |
| 放宽 / fallback | 同上：严格结果为 0 时返回 `status: no_match` + `relaxation_options`，由用户点击后带 `relaxation_applied` 重新请求；**不存在静默放宽** |
| 评分器注册 | `app/services/serving.py`（`state.collection_scorers`） |
| 数据导入 | `pipelines/real_ingest.py`（Kaggle CSV → `data/processed/real/<version>/`）、`pipelines/activate.py`（写入业务库） |
| 旧离线实验 | `evaluation/experiments/collection.py`（代理基准：高评分历史当收藏夹） |
| 旧训练入口 | `pipelines/fit.py::_train_gbdt / _train_two_tower`：只在存在 `collection_events.jsonl` 时训练；真实模式下为 `limited_by_data` |

## 3. 当前推荐是否真实计算

- 真实计算：每次请求读取收藏夹、构建画像、规则筛选、评分并排序；没有固定列表、手写结果或 mock。
- 历史背景中提到的“24 部手写作品、675 条种子评分”和“空结果静默放宽”属于已删除的旧模板 AnimeFinder；当前 `main` 已无该代码（旧模板删除于提交 `e0f1e27`），当前实现在空结果时返回用户可选的放宽建议。
- 学习型评分器（GBDT、双塔）在真实模式下**没有训练过**：`artifacts/real/real-89bfdea00b/manifest.json` 中 `gbdt` 与 `two_tower` 状态均为 `limited_by_data`（无收藏事件）。

## 4. 实际数据（统计结果）

| 项目 | 数量 | 来源 |
|---|---|---|
| 当前激活版本 | `real-89bfdea00b` | `artifacts/real/active.json` |
| 目录行数（快照） | 24,905 | `data/processed/real/real-89bfdea00b/catalog.parquet` |
| 在库（已排除 Hentai/Erotica/Rx） | 23,363 | `in_catalog` |
| 在库且有公开评分 | **14,182** | 本实验的候选目录 |
| 在库且有评分与类型（当前收藏推荐池 `recommendable`） | 12,429 | |
| 评分交互（处理后） | 23,963,380 | `ratings/part-*.parquet`（列 `offline_user_id, item_idx, rating`） |
| 落在 14,182 部目录上的评分 | 23,954,329 | |
| 其中个人评分 ≥ 8 | 13,580,272 | |
| 有 ≥1 个高评分作品的用户 | 264,020 | |
| 高评分集合大小 1 / 2–5 / ≥6 的用户 | 23,055 / 38,763 / 202,202 | |
| 重复 user-item 对 | 0（预处理已去重；冲突评分对 0） | `reports/preprocess_report.json` |
| 业务库中的真实收藏夹 | **0**（真实库 1 个用户、0 收藏、0 评分） | `var/real/anilogic.sqlite3` |

结论：**没有真实收藏数据**。算法② 的离线效果评估只能使用“个人评分 ≥ 8 的高评分偏好集合”作为代理标签，报告中一律称为“高评分偏好集合代理评估”。

### 4.1 字段覆盖率（14,182 部目录）

| 字段 | Kaggle 处理数据 | 队友目录投影（见 4.2） | 本实验采用 |
|---|---|---|---|
| genre | 87.6%（21 个标签） | 90.1% | Kaggle |
| theme | 0.2%（仅 33 部 Jikan 补全） | **68.9%**（53 个标签） | 两者并集 |
| demographic | 0.1% | **37.2%**（5 个标签，多值时仅保留字典序第一个） | 两者并集 |
| studio | 76.2%（多值，1,042 个） | 78.8%（仅第一个工作室） | Kaggle |
| 集数 | 99.6% | — | Kaggle |
| 公开评分 | 100%（目录定义要求） | — | Kaggle |

### 4.2 theme / demographic 的第二来源

`algorithm1_benchmark/data/catalog.jsonl.gz`（队友提交 `d0824db`，已合并进 `main`）提供 14,202 部作品的 MAL theme / demographic 投影，`algorithm1_benchmark/data/provenance.json` 记录其上游为团队正式数据包 `AnimeFinder_system_dev_20260918_v1`（包本身不在仓库内，SHA-256 已记录）。核验结果：

- 我们的 14,182 部目录 **全部** 能按 MAL ID 对上；
- 集数一致 14,130/14,130，工作室一致 10,799/10,812，类型集合一致 12,356/12,780（差异来自新旧 MAL 标签）；
- 同一 MAL ID 体系，未混入 AniList ID。

处理：只把 theme 与 demographic 作为算法② 的补充元数据（与 Kaggle/Jikan 已有值取并集），在 `data_manifest.json` 中记录文件哈希、覆盖率和“多值 demographic 被投影为单值”的限制。**不改动共享目录和业务库**，因此算法①、首页、聊天的行为不受影响。离线实验与线上服务使用同一份补充表，并用目录哈希绑定。

## 5. 当前公式与语义（核验结果）

| 项目 | 当前实现 |
|---|---|
| 当前默认评分器 | `collection_profile_v1`：`0.7 × 画像匹配 + 0.3 × score/10`（不是 PPT 公式） |
| PPT 公式实现 | `manual_slides_v0`：`score/10 + 0.15 × hits` |
| hits 定义 | 候选作品的 genre 与 theme 标签中，**在画像里出现过（频率 > 0）**的个数；按命名空间分开计数。**没有用到频率排序**，因此不是“标签频次偏好 + 命中”的明确实现 |
| 画像频率 | 标签出现次数 ÷ **该命名空间有标签的作品数**（不是 ÷ \|H\|）；低分 / 不感兴趣的作品不进入画像 |
| 标签解析 | Kaggle 多值字段按逗号拆分、保留公司后缀、去重；标签匹配用 `norm()`（NFKC + casefold + 合并空白） |
| 关键词 | `split_keywords`：按 `, ， 、 ; ；` 拆分、规范化、去重；每个关键词需**包含**于类型文本或标题/别名文本（子串匹配），多关键词 AND |
| 评分 / 集数 | `min_score` 为 ≥，`max_episodes` 为 ≤；未知值一律不满足 |
| 排除 | 内容规则（explicit）、`recommendable` 池（评分且有类型）、收藏夹内作品、已完成作品、用户“不感兴趣”的作品与标签 |
| 同分排序 | 分数降序，`item_key` **字符串**升序（`mal:10` 排在 `mal:9` 前），不是 anime_id 数值升序 |
| Top-K | 8；不足 8 个时返回实际数量并给出提示，不补齐 |
| 空结果 | `status: no_match` + 放宽建议（不自动应用） |

## 6. 本任务复用、修改和协议差异

复用：

- `ConstraintEngine` 作为离线与线上**唯一**的规则实现（关键词子串语义、评分、集数、去重、内容规则）；离线候选生成与线上调用同一实现，并用测试核对等价。
- Kaggle 处理数据（版本 `real-89bfdea00b`）与 MAL `item_key` 体系；`norm` / `split_keywords`。
- 现有放宽建议机制（用户触发）。

新增 / 修改：

- 新包 `app/recommendation/algorithm2/`（目录、画像、特征、评分器、双塔、服务）与 `evaluation/algorithm2/`（准备、切分、训练、调参、冻结、测试、基准、报告、接入核验、run-all）。
- `ConstraintEngine` 增加 `pool.scored` 规则（在库且有公开评分）和关键词掩码缓存；收藏夹推荐改用该目录。
- 收藏夹服务改为调用算法② 服务；响应补充放宽状态字段；评分器由冻结结论配置。

旧行为与新实验协议的差异（新协议依照执行提示词，旧实现保留为辅助对照）：

| 项目 | 旧行为 | 新协议 |
|---|---|---|
| 候选目录 | `recommendable`（评分 + 类型，12,429 部） | 在库且有评分（14,182 部）；类型缺失的作品保留并报告 |
| 画像频率分母 | 有该命名空间标签的作品数 | \|H\| |
| hits | 候选标签在画像中出现过的个数 | 画像中频率最高的 10 个带命名空间标签（genre:/theme:，并列按标签字典序）与候选标签的交集大小 |
| 同分排序 | `item_key` 字符串 | anime_id 数值升序 |
| theme / demographic | 几乎全部缺失 | 加入队友 MAL 投影（覆盖 68.9% / 37.2%） |
| 默认评分器 | `collection_profile_v1` | 由冻结的选型协议决定（A0/A1/B0/B1/C0） |

旧的 `manual_slides_v0`（旧 hits）和 `collection_profile_v1`（旧默认）作为辅助对照 `AUX_slides_v0` / `AUX_profile_v1` 一并评估，不参与选型。

## 7. 环境

- Python 3.12.14（uv 管理的 `.venv`），PyTorch 2.14.0（MPS 可用），scikit-learn 1.9.1。
- 新增依赖：`lightgbm 4.7.0`（加入 `pyproject.toml` 的 `ml` extra 并锁定在 `uv.lock`）。macOS 上 LightGBM 需要 OpenMP 运行库，已用 Homebrew 安装 keg-only 的 `libomp 23.1.2`（不链接到全局路径；可用 `brew uninstall libomp` 移除）。Windows / Linux 的 LightGBM wheel 自带或使用系统 OpenMP。
- 硬件：Apple M1 Pro，8 核 CPU，16 GB 内存。
