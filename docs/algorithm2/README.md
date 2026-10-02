# 算法②：收藏夹推荐的效果比较、模型选择与系统接入

收藏夹页面的推荐流程：**收藏夹内容 H → 偏好画像 → 固定规则层 → 可替换评分器 → Top-8**。本目录说明这次
实验与接入做了什么、结论是什么、记录在哪里、如何复现。完整的中文报告由实验流程生成，位于运行目录：

- 正式运行：[`evaluation/algorithm2/runs/real-main-01/final_report.md`](../../evaluation/algorithm2/runs/real-main-01/final_report.md)
  （英文简表 `summary_en.md`，失败案例 `case_analysis.md`，复现说明 `README_reproduce.md`）
- 冒烟运行：[`evaluation/algorithm2/runs/real-smoke-01/`](../../evaluation/algorithm2/runs/real-smoke-01/)（只检查流程，不作选型依据）
- 仓库审计：[`repo_audit.md`](repo_audit.md)

## 结论（正式运行 `real-main-01`）

测试集 3,000 个主组用户，完整目录 14,182 部，规则 R0，用户级 Macro 指标；标签是**代理标签**（用户评分 ≥ 8
的番剧集合当作“收藏夹”，隐藏其中一部），不是真实收藏行为。

| 模型 | HR@8 | NDCG@8 | AUC | ΔHR@8 vs A0（百分点，95% CI） | ΔHR@8 vs A1（百分点，95% CI） |
|---|---|---|---|---|---|
| A0 原手工公式 | 0.0210 | 0.0107 | 0.8699 | — | −1.10 [−1.60, −0.57] |
| A1 验证集调权公式（w = 0.05） | 0.0320 | 0.0179 | 0.9082 | +1.10 [+0.57, +1.60] | — |
| B0 LightGBM 两特征（seed 42） | 0.0543 | 0.0316 | 0.9213 | +3.33 [+2.50, +4.17] | +2.23 [+1.53, +2.97] |
| **B1 LightGBM 八特征（seed 42，部署）** | **0.1317** | **0.0723** | **0.9527** | **+11.07 [+9.83, +12.30]** | **+9.97 [+8.73, +11.17]** |
| C0 内容双塔（seed 42） | 0.1083 | 0.0621 | 0.9640 | +8.73 [+7.53, +9.93] | +7.63 [+6.53, +8.77] |

- 三个训练 seed 的均值：B0 0.0532、B1 0.1320、C0 0.1133（标准差 0.0008 / 0.0010 / 0.0037）。
- 关键对照：B1 − B0 = +7.88 [+6.78, +8.99] 个百分点（六个画像特征的贡献，B0 已含公开评分与 hits）；
  C0 − B1 = −1.87 [−3.03, −0.69]（双塔低于八特征 GBDT）；A1 − A0 = +1.10 [+0.57, +1.60]。
- **选型**：按训练前写好的 `selection_protocol.json`，B1 在验证集预选（HR@8 0.1301），测试确认全部通过
  （AUC ≥ 0.65、比 A0 高 ≥ 2 个百分点、规则违反 0、相对 A0 与 A1 的配对区间下界都 > 0）。网站真实模式的默认
  评分器因此设为 B1（`configs/real.yaml`），合成模式仍为 A0。
- 冒烟运行（150 个测试用户）预选了 C0，但相对 A1 的区间包含 0，按协议保留 A1——说明小样本下的结论不稳定，
  正式结论只看主运行。
- 服务时延（本机单线程，完整目录请求的服务处理 P50）：B1 约 120 ms，公式约 7 ms，C0 约 1 ms；详见运行目录的
  `latency.csv` 与报告第 5.5 节。

## 代码位置

| 内容 | 位置 |
|---|---|
| 目录、画像、八特征、评分器、LightGBM 文本模型的 NumPy 读取器、模型包校验、服务 | `app/recommendation/algorithm2/` |
| 规则层（与离线评估共用，新增 `pool.scored`） | `app/recommendation/constraints.py` |
| 收藏夹接口（读取当前收藏 → 同一代码路径 → Top-8；严格 / 放宽契约） | `app/services/recommend.py`、`app/api/v1/collections.py`、`app/web/static/js/pages/collection.js` |
| 实验流程（十个阶段的命令行） | `evaluation/algorithm2/` |
| 测试 | `tests/unit/test_algorithm2.py`、`tests/integration/test_algorithm2_pipeline.py`、`tests/e2e/test_journeys.py`（旅程 4） |
| 运行记录 | `evaluation/algorithm2/runs/<run_id>/`（`work/` 为可再生的中间文件，不入库） |

## 实验协议摘要

- **数据**：Kaggle MyAnimeList 2023 快照（处理版本 `real-89bfdea00b`），只用 MAL id；目录 = 可服务（排除 Hentai /
  Erotica / Rx，保留 Ecchi 与 R+）且有公开评分的 14,182 部；theme / demographic 按 MAL id 并入队友的
  `algorithm1_benchmark/data/catalog.jsonl.gz`；不臆造字段，未知集数在集数上限规则下视为不满足。
- **切分**：用户级 70 / 15 / 15，`split_seed = 42`，按偏好集合规模分层；主组（集合 ≥ 6）与稀疏组（2–5）分开；
  验证 / 测试每个用户隐藏 1 部，训练最多 3 部；目标从不进入画像输入。
- **规则场景**：R0 无筛选；R1 关键词 = H 中最常见 genre；R2 评分 ≥ 7.0 且集数 ≤ 26；R3 = R1 + R2。严格模式，
  目标不满足规则时不插回，另报端到端结果。
- **候选**：对通过规则的完整目录排序，不采样候选；同分按 MAL id 决定名次，AUC 同分记 0.5。
- **训练样本**：每个训练 query 1 正 4 负（`negative_seed = 42`），B0 / B1 / C0 共用。
- **选型**：参数与预选只用验证集；冻结后测试只运行一次；B0 / B1 / C0 用训练 seed [42, 2026, 7]，部署 seed 42；
  用户级配对 bootstrap 2,000 次。

## 复现命令

```bash
uv sync --extra ml                 # LightGBM 4.7、PyTorch
brew install libomp                # 仅 macOS：LightGBM 依赖的 OpenMP 运行库

make alg2-run RUN=real-main-01 LEVEL=main      # 全部阶段（已完成的阶段会跳过）
make alg2-stage STAGE=report RUN=real-main-01  # 单个阶段
make alg2-fixture                              # 生成的小数据上检查流程（不是效果结论）
```

阶段：`audit`、`prepare`、`split`、`tune`、`train`、`freeze`、`evaluate-test`、`benchmark`、`verify-integration`、
`report`（`python -m evaluation.algorithm2.cli <stage> --run <id> [--level fixture|smoke|main] [--threads N]`）。
`freeze` 之后 `tune` / `train` 拒绝运行，`evaluate-test` 只复现同一结果。

LightGBM（Homebrew libomp）与 PyTorch（自带 libomp）在 macOS 同一进程中会卡死或崩溃，所以实验流程把两者放在
不同子进程中运行；网站服务用 NumPy 读取 LightGBM 文本模型（按位掩码逐树求值，原始分数与原生 LightGBM 完全
一致，`freeze` 时核对并记录在 `freeze_manifest.json`）。

## 接入方式

`configs/real.yaml`：

```yaml
recommendation:
  collection:
    scorer: B1
    algorithm2_run: evaluation/algorithm2/runs/real-main-01
```

服务启动时校验模型文件 SHA-256、目录哈希、特征版本与双塔编码器；任何不一致都会改用公式 A0，并在状态页显示
`collection_scorer: degraded`，用户的筛选条件仍严格执行。严格无结果时只给出放宽选项，用户选择后响应带
`relaxed`、`relaxed_rules` 与 `strict_result_count`，页面显示放宽了哪个条件。接口的 `scorer` 字段只供实验与
运维使用，页面不发送。`verify-integration` 在数据库副本上核对了离线与接口 Top-8 一致（1,000 / 1,000）。

## 限制

- 代理标签不等于收藏夹；未评分条目是弱负例。应用数据库中还没有真实收藏，也没有线上实验，网站上的实际效果
  尚未验证。
- 静态快照、用户留出：评估“同期其他高评分条目能否找回”，不是预测未来行为。
- 目标多为热门高分番剧，带有流行度偏差；冷门目标普遍难以进入前 8。
- C0 能编码没有交互的新番，但没有 item-disjoint 实验，不能声称冷启动更好。
- 时延是单台笔记本上的观测值，项目没有既定 SLO。
