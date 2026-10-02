# 算法②实验包：收藏夹推荐的五组方案比较、预注册选型与系统接入

对应收藏夹页面的推荐流程 **收藏夹内容 H → 偏好画像 → 固定规则层 → 可替换评分器 → Top-8**，比较 PPT 中的
**A0 原手工公式、A1 调权公式、B0 两特征 GBDT、B1 八特征 GBDT、C0 内容双塔**。所有方案在同一真实目录
（14,182 部）、同一用户切分、同一候选与规则、同一评估器下比较；选型规则在任何训练之前写好并冻结。

先看 [正式运行报告](results/real-main-01/final_report.md)、[英文简表](results/real-main-01/summary_en.md)、
[失败案例](results/real-main-01/case_analysis.md)、[配对比较](results/real-main-01/paired_comparisons.csv)、
[时延](results/real-main-01/latency.csv)、[系统接入检查](results/real-main-01/integration_report.json) 和
[测试输出](results/tests.txt)。

标签是**代理标签**：把用户评分 ≥ 8 的番剧集合当作“收藏夹”，隐藏其中一部，看各方案能否把它排进 Top-8。
应用数据库里还没有真实收藏，所以结论都是“高评分偏好集合代理评估”，不是收藏行为本身。

正式运行 `real-main-01`：24,000 个用户，测试集 3,000 个主组用户，完整目录排序，规则 R0，用户级 Macro 指标。

| 模型 | HR@8 | NDCG@8 | AUC | ΔHR@8 vs A0（百分点，95% 区间） | ΔHR@8 vs A1（百分点，95% 区间） |
|---|---|---|---|---|---|
| A0 原手工公式 | 0.0210 | 0.0107 | 0.8699 | — | −1.10 [−1.60, −0.57] |
| A1 验证集调权公式（w = 0.05） | 0.0320 | 0.0179 | 0.9082 | +1.10 [+0.57, +1.60] | — |
| B0 LightGBM 两特征（seed 42） | 0.0543 | 0.0316 | 0.9213 | +3.33 [+2.50, +4.17] | +2.23 [+1.53, +2.97] |
| **B1 LightGBM 八特征（seed 42，部署）** | **0.1317** | **0.0723** | **0.9527** | **+11.07 [+9.83, +12.30]** | **+9.97 [+8.73, +11.17]** |
| C0 内容双塔（seed 42） | 0.1083 | 0.0621 | 0.9640 | +8.73 [+7.53, +9.93] | +7.63 [+6.53, +8.77] |

- 三个训练 seed 的均值：B0 0.0532、B1 0.1320、C0 0.1133（标准差 0.0008 / 0.0010 / 0.0037）。
- 关键对照：A1 − A0 = +1.10 [+0.57, +1.60]（调权有小幅收益）；B0 − A1 = +2.12 [+1.48, +2.84]（同两特征下学习
  评分函数有收益）；**B1 − B0 = +7.88 [+6.78, +8.99]**（六个画像特征带来主要收益）；C0 − B1 = −1.87
  [−3.03, −0.69]（双塔低于八特征 GBDT；C0 的 AUC 最高，但前 8 名不如 B1）。
- **选型结论**：B1 在验证集预选（HR@8 0.1301），测试确认全部通过（AUC ≥ 0.65、比 A0 高 ≥ 2 个百分点、规则
  违反 0、相对 A0 与 A1 的配对区间下界都 > 0）。网站真实模式的默认评分器设为 B1，合成模式仍为 A0。
- 冒烟运行 `real-smoke-01`（150 个测试用户）预选了 C0，但相对 A1 的区间包含 0，按协议保留 A1：小样本下结论
  不稳定，正式结论只看主运行。
- 服务时延（本机单线程，完整目录一次请求的服务处理 P50）：B1 约 120 ms，公式约 7 ms，C0 约 1 ms。

## 1. 直接运行

以下命令在**仓库根目录**执行。与算法①的独立包不同，本实验直接调用网站代码（目录、规则层、特征、评分器都在
`app/` 中），这样离线评估与网站走的是同一条代码路径；运行前需要仓库的 Python 环境与已激活的真实数据。

```bash
uv sync --extra ml                 # Python 3.12 环境（LightGBM 4.7、PyTorch）
brew install libomp                # 仅 macOS：LightGBM 依赖的 OpenMP 运行库

.venv/bin/python algorithm2_benchmark/verify_delivery.py      # 先核对交付文件（SHA-256 清单）
.venv/bin/python algorithm2_benchmark/run_checks.py           # 测试，实际输出写入 results/tests.txt

.venv/bin/python -m algorithm2_benchmark report --run real-main-01              # 由已有记录重新生成报告（秒级）
.venv/bin/python -m algorithm2_benchmark verify-integration --run real-main-01  # 接口与离线一致性（约 1 分钟）
.venv/bin/python -m algorithm2_benchmark benchmark --run real-main-01           # 时延（约 5 分钟）

.venv/bin/python -m algorithm2_benchmark run-all --run my-smoke --level smoke   # 新建一次完整运行（约 5 分钟）
.venv/bin/python -m algorithm2_benchmark run-all --run my-main --level main     # 正式规模（本机约 3 小时）
```

Makefile 也提供同样的入口：`make alg2-run RUN=... LEVEL=...`、`make alg2-stage STAGE=... RUN=...`、
`make alg2-checks`、`make alg2-verify`、`make alg2-fixture`（生成的小数据上检查流程，不是效果结论）。

阶段依次为 `audit`、`prepare`、`split`、`tune`、`train`、`freeze`、`evaluate-test`、`benchmark`、
`verify-integration`、`report`；`run-all` 按顺序执行并跳过已完成的阶段，每个阶段在独立进程中运行。新运行需要
`--level fixture|smoke|main`，已有运行沿用自己的 `config.json`；`--threads N` 设定 LightGBM / PyTorch 线程数。

交付清单记录原始交付文件；重跑阶段会改写运行元数据、报告和测试输出，所以请先校验再重跑。确认新的交付后用
`verify_delivery.py --write` 更新清单。

## 2. 文件结构

| 路径 | 用途 |
|---|---|
| `pipeline/cli.py`、`config.py`、`common.py` | 命令行入口与阶段编排、运行配置（所有影响结果的取值）、哈希 / 日志 / 运行元数据 |
| `pipeline/audit.py` | 仓库与数据事实（对应 `docs/repo_audit.md`） |
| `pipeline/data.py` | 读取并校验处理后的真实数据、清洗评分、生成代理正例；fixture 小数据 |
| `pipeline/split.py` | 用户级切分、隐藏目标、R1 关键词、1:4 弱负例采样、预注册的 `selection_protocol.json` |
| `pipeline/train.py` | A1 调权、B0 / B1 LightGBM 与 C0 双塔的调参和三个 seed 的最终训练（LightGBM 与 PyTorch 分进程） |
| `pipeline/select.py` | 验证集预选、冻结（模型哈希、原生 LightGBM 与服务端求值的一致性核对）、测试确认 |
| `pipeline/evaluate.py`、`metrics.py`、`final_eval.py` | 完整目录评估器、指标定义、测试集评估、分组与配对比较 |
| `pipeline/benchmark.py` | 纯评分 / 服务处理 / 接口三层时延与加载成本 |
| `pipeline/integration.py`、`webapp.py` | 在真实数据库副本上运行网站，核对接入（14 项检查） |
| `pipeline/report.py` | 中文报告、英文简表、失败案例、复现说明 |
| `results/real-main-01/` | 正式运行：配置、数据 / 切分 / 选型协议 / 冻结清单、`models/`（9 个模型）、`predictions.parquet`（逐 query × 规则 × 模型）、指标、分组、配对比较、时延、接入检查、报告 |
| `results/real-smoke-01/` | 冒烟运行（同样的记录，规模小，只检查流程） |
| `results/tests.txt` | 最近一次 `run_checks.py` 的实际输出 |
| `docs/repo_audit.md` | 开始前的仓库审计 |
| `tests/` | 实验自身的测试：指标定义、fixture 全流程（切分与泄漏、真实拟合、候选一致、离线 = 服务、冻结后不能再调参、记录齐全） |
| `run_checks.py`、`verify_delivery.py`、`DELIVERY_MANIFEST.json` | 测试入口、交付校验与清单 |

每个运行目录的 `work/` 存放可再生的中间文件（正例、特征、分进程预测、日志），不进入 Git；报告与接入阶段在没有
`work/` 时读取运行目录里已发布的 `query_manifest.parquet`、`sampled_pairs.parquet`。

**网站运行用的代码不在本文件夹**：`app/recommendation/algorithm2/`（目录、画像、八特征、评分器、LightGBM 文本
模型的 NumPy 求值、模型包校验、服务），规则层 `app/recommendation/constraints.py`，收藏夹接口
`app/services/recommend.py`。保护网站行为的测试在 `tests/unit/test_collection_algorithm2.py` 与
`tests/integration/test_collection_algorithm2.py`（`run_checks.py` 会一并运行）。

两次运行是在整理成本文件夹之前完成的：运行记录里 `run_metadata.json`、`audit_facts.json`、
`freeze_manifest.json` 出现的 `evaluation/algorithm2/` 是当时的代码位置，作为历史记录保持原样。

## 3. 方案与指标约定

| ID | 评分 |
|---|---|
| A0 | `rating_norm + 0.15 × hits`（hits = 候选的 genre / theme 标签落在 H 的 top-10 命名空间标签中的个数） |
| A1 | `rating_norm + w × hits`，w 只在验证集从 [0.05, 0.10, 0.15, 0.20, 0.30] 中选 |
| B0 | LightGBM 二分类（objective = binary），输入 `rating_norm`、`hits`，按原始分数排序 |
| B1 | 同上，输入八个固定特征：`rating_norm`、`hits`、genre / theme / demographic / studio 亲和度、集数接近度、与收藏中最相似作品的标签相似度 |
| C0 | PyTorch 内容双塔：收藏塔与番剧塔各 128 → 64 → 32/64，多热标签、制作公司、数值特征与缺失 mask；点积 logits + BCEWithLogitsLoss，不使用 in-batch 负例 |

- 画像频率 = 计数 / |H|；缺失特征保持 NaN，不补造。B0 / B1 / C0 共用同一组 1:4 训练样本（`negative_seed = 42`）。
- 指标：HR@8、NDCG@8（单正例，IDCG = 1）、MRR@8、AUC（同分记 0.5；候选数 < 2 为 null）。先按用户平均再对用户
  平均；同分按 MAL id 升序决定名次，但不计入 AUC 的区分度。
- 主集合（预注册）：规则 R0、主组（可见历史 ≥ 5）、候选数 > 8、目标满足规则。
- 规则场景：R0 无筛选；R1 关键词 = H 中最常见 genre；R2 公开评分 ≥ 7.0 且集数 ≤ 26（未知集数不满足）；
  R3 = R1 + R2。严格模式：不满足规则的目标不插回，单独报告端到端结果。
- 测试确认（相对 A0）：AUC ≥ 0.65、HR@8 提升 ≥ 2 个百分点、规则违反 0、用户配对 bootstrap（2,000 次）95%
  区间下界 > 0；学习模型还需相对 A1 的区间下界 > 0，否则保留 A1。

## 4. 实验协议与冻结保护

- 数据：Kaggle MyAnimeList 2023 快照（处理版本 `real-89bfdea00b`，`prepare` 按清单核对文件哈希），只用 MAL id；
  目录 = 可服务（排除 Hentai / Erotica / Rx，保留 Ecchi 与 R+）且有公开评分；theme / demographic 按 MAL id 并入
  `algorithm1_benchmark/data/catalog.jsonl.gz`。
- 切分：用户级 70 / 15 / 15，`split_seed = 42`，按偏好集合规模分层；主组（≥ 6）与稀疏组（2–5）分开报告；验证 /
  测试每个用户隐藏 1 部，训练最多 3 部；目标从不进入画像输入。
- 选型：`selection_protocol.json` 在任何训练之前写入；参数与预选只用验证集；B / C 用训练 seed [42, 2026, 7]，
  部署 seed 42；`freeze` 之后 `tune` / `train` 拒绝运行，`evaluate-test` 重跑只复现同一结果。
- 进程隔离：macOS 上 LightGBM（Homebrew libomp）与 PyTorch（自带 libomp）在同一进程会卡死或崩溃，所以训练与
  评估把两者放在不同子进程；网站用 NumPy 直接读取 LightGBM 文本模型求值（按位掩码逐树计算，原始分数与原生
  LightGBM 完全一致，`freeze` 时在 56 万行上核对，最大差 0）。

## 5. 接入网站

`configs/real.yaml`：

```yaml
recommendation:
  collection:
    scorer: B1
    algorithm2_run: algorithm2_benchmark/results/real-main-01
```

服务启动时校验模型文件 SHA-256、目录哈希、特征版本与双塔编码器；任何不一致都改用公式 A0，状态页显示
`collection_scorer: degraded`，用户的筛选条件仍严格执行。严格无结果时只给出放宽选项，用户选择后响应带
`relaxed`、`relaxed_rules` 与 `strict_result_count`，页面显示放宽了哪个条件。接口的 `scorer` 字段只供实验与
运维使用，页面不发送。`verify-integration` 在数据库副本上核对：接口 Top-8 与离线预测 1,000 / 1,000 一致、
收藏变化改变输出、严格不足不补齐、无结果不静默放宽、推荐 id 可打开详情页、模型包缺失或被改动时退回公式。

## 6. 限制

- 代理标签不等于收藏夹；未评分条目是弱负例。没有真实收藏、也没有线上实验，网站上的实际效果尚未验证；有了
  真实收藏数据后可以用同一流程重新评估。
- 静态快照、用户留出：评估“同期其他高评分条目能否找回”，不是预测未来行为。
- 目标多为热门高分番剧，带有流行度偏差；冷门目标普遍难以进入前 8。
- C0 能编码没有交互的新番，但没有 item-disjoint 实验，不能声称冷启动更好。
- LightGBM 的 seed 只影响直方图分箱的行抽样（网格没有子采样），seed 间差异不代表完整的训练随机性。
- 时延是单台笔记本上的观测值，项目没有既定 SLO。
