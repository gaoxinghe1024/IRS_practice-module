# 算法①独立实验包 V2：扩大样本与相似动漫 Top-5 优化

对应任务图片中的 **A 五维加权相似度、B 逻辑回归、C BGE 向量、D 标签＋语义融合**。在相同的 14,202 部真实作品目录中比较 Top-5，使用独立 MAL 推荐真值；无需启动 FastAPI、前端或数据库。

先看 [实验报告](results/REPORT.md)、[同一新测试集的优化前后指标](results/before_after.csv)、[结果图](results/comparison.png)。下载整个文件夹后，双击 [交互结果页](results/index.html) 可切换原配方/优化配方、作品和划分，查看四种方法的推荐、命中及五维特征。

本轮预先固定 **600 个新目标（200 验证＋400 测试）**；旧 120 个目标全部归入 historical。只有至少有 3 条目录内 MAL 推荐的目标进入指标计算，实际有效数量和失败审计见报告。“测试目标数”指不同的目标动漫数量；每种算法为每部目标动漫各推荐 5 部，并非对同一部作品重复运行这么多次。

优化只在新验证集选型，选定后冻结，才评测新测试集。B 选中按正例作品出现频率匹配的随机负例；C 保留原五维属性文本，增加无标签全目录均值校正；D 固定 0.6/0.4，采用无标签余弦分位范围归一化。五维输入、完整候选库和验收门槛保持一致。提升是否成立以报告中的同集前后结果和成对区间为准。

实际有效新测试目标为 **234 部（原来的 6.2 倍）**，新验证目标 102 部。同集比较中，B 的覆盖率 3.18%→4.00%、NDCG 0.0328→0.0406；C 略降，D 的两项指标一升一降。各方案前后差值区间均跨零，尚无方案通过相对 A 的正式替换门槛，保留 A 为系统基线。

**人工标注暂缺**是用户确认的本轮边界。本包提供可填写的 [人工标注表](data/human_annotations_template.csv) 和导入评测工具；没有用 AI 编造人工标注，也没有把共同高分作品对当作评测真值。

## 1. 直接运行（离线复现）

以下命令均在本文件夹 `algorithm1_benchmark/` 内执行。需要可用的 Python 3.12；Windows 可用 `py -3.12` 或本机 Python 的实际路径建立环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify_delivery.py
.\.venv\Scripts\python.exe run_benchmark.py
.\.venv\Scripts\python.exe run_checks.py
.\.venv\Scripts\python.exe -m kbbench.recommend --anime-id 1 --method C
.\.venv\Scripts\python.exe -m kbbench.recommend --anime-id 1 --method C --variant baseline
```

Linux/macOS 使用 `.venv/bin/python` 替代 `.\.venv\Scripts\python.exe`。依赖安装需要网络；**使用随包模型、向量、数据重跑评测和推荐不需要网络**。精确依赖快照见 `requirements.lock.txt`（本次 Windows/Python 3.12 环境）；日常跨平台安装优先使用 `requirements.txt`。

只运行 A 时，**不需要任何第三方依赖**：

```powershell
python baseline.py --anime-id 1
```

交付 manifest 记录原始交付文件；重跑实验会更新耗时、环境和报告，所以请先校验，再重跑。若维护新版交付，完成验证后用 `python verify_delivery.py --write` 更新清单。

## 2. 文件结构

| 路径 | 用途 |
|---|---|
| `kbbench/features.py` | 五维输入、Jaccard、单值匹配、集数距离、语义文本 |
| `baseline.py` | A 的零第三方依赖入口 |
| `kbbench/train.py`、`fit_candidates.py` | B 的原配方、负例采样及逻辑回归候选 |
| `kbbench/embed.py`、`representations.py` | 固定版本 BGE、五维文本与向量均值校正 |
| `kbbench/expand.py`、`optimize.py` | 冻结新 cohort、仅用验证集选型及可验证重放 |
| `kbbench/engine.py` | A/B/C/D 共用离线推荐接口及 D 融合 |
| `kbbench/prepare.py` | 从正式数据包提取目录、冻结目标与划分 |
| `kbbench/collect.py`、`mal_html.py` | MAL 真值采集、证据保存与原页面解析 |
| `kbbench/metrics.py`、`evaluate.py` | 独立真值评测、bootstrap、双条件替换判断 |
| `data/catalog.jsonl.gz` | 14,202 部真实作品的五维投影 |
| `data/training_pairs*.jsonl.gz` | 两种配方各 100,000 条弱训练对，不含用户资料，均排除全部 720 个目标 |
| `data/targets.jsonl`、`optimization_protocol.json` | 720 个预先固定的目标、划分和有限候选选型协议 |
| `data/mal_ground_truth.jsonl`、`mal_cache/` | 实际采集标签、投票及原始证据 |
| `data/franchise_links.jsonl.gz` | 仅供 sanity check 的直接前作/续作关系 |
| `data/provenance.json`、`training_provenance.json` | 输入版本、哈希、适配与采样记录 |
| `models/logistic.json`、`logistic_optimized.json` | 原配方/优化配方 B 系数与训练诊断，不使用 pickle |
| `models/bge_embeddings.npz`、`bge_optimized.npz` | 原配方/优化配方全目录 384 维向量；各有元数据 |
| `models/optimization_selection.json` | 选定配置、冻结时间、验证真值及选定文件哈希 |
| `results/`、`results/baseline/` | 优化/原配方指标、逐目标预测、成对对比、测试与交互页 |
| `results/validation_trials.json`、`selection_ground_truth.jsonl` | 验证选型得分与当时的真值快照 |
| `history/v1/` | V1 报告、模型及目标快照；完整旧实现保存在提交 d0824db |
| `tests/` | 核心算法边界与实际交付数据验收 |

## 3. 算法及指标约定

| 方法 | 分数 |
|---|---|
| A | `0.35×genre + 0.30×theme + 0.15×demographic + 0.10×studio + 0.10×length` |
| B | 同五维特征的 `sigmoid(w·x+b)`；固定 L2 正则，梯度优化拟合 |
| C | `BAAI/bge-small-en-v1.5` 的 384 维 CLS 向量；选定版本减去全目录均值后 L2 归一化，再计算余弦 |
| D | `0.6×A + 0.4×semantic`；选定版本用全目录随机作品对余弦的 1%/99% 分位映射到 [0,1] 并截断；不调融合权重 |

`--variant baseline` 使用原始未中心化向量及 `(cosine+1)/2` 融合；原配方 B 已重新训练并排除全部新增目标，以保证前后比较没有训练泄漏差异。所有方法均从完整目录中检索，仅排除目标自身。

只有 genre/theme 使用集合 Jaccard；demographic/studio 是单值匹配，length 取集数，沿用系统基线 `1/(1+差值/12)`。空标签不匹配，未知集数基线值为 0.5。真实目录有多工作室/多受众：分别取源数组首个工作室名称、字典序首个受众。四种方法共享这一规则，统计与影响见报告。

图片的“HR@5：真值集合被 Top-5 覆盖多少”按 **Recall@5/coverage** 实现，同时另报标准 **Hit Rate@5：是否至少命中一个**。NDCG@5 使用二元相关性。所有指标按目标宏平均，不把四种方法的内部相似度分数直接比较。

替换规则：**测试集 coverage 至少比 A 高 0.02，且 NDCG@5 不低于 A**。判断写在 `results/summary.json → replacement_decisions`，附同目标成对 bootstrap 95% 区间。MAL 真值的通过只是该来源下通过；人工主验收尚未完成，不自动改系统。

## 4. 从原数据完整重建

离线重放不需要原大数据包。若需要重新生成训练样本，准备已验收的 `AnimeFinder_system_dev_20260918_v1` 正式包，建议在实验文件夹副本内执行；不使用 E 盘较早的 B provisional 目录。

```powershell
# 将 <release-folder> 替换为正式交付包所在目录
python -m kbbench.prepare --release "<release-folder>" --catalog-only
python -m kbbench.train --release "<release-folder>"
python -m kbbench.fit_candidates
# 下载固定版本 BGE；随包已有原配方向量
python -m kbbench.embed --download-only
python -m kbbench.representations
# 验证同一验证集仍选中相同配置；不覆盖冻结的选型文件
python -m kbbench.optimize --replay
python run_benchmark.py
python run_checks.py
```

只重训原配方 B 可执行 `python -m kbbench.train`，使用随包训练对，不读取原始评分。只有 `--release` 才从原包读取训练 CSR；不会读取留出的验证/测试评分。模型下载和未选中的候选向量在 `.cache/`，不进入 Git；随包选定向量可直接离线运行 C/D。

`expand`、`prepare` 和 `optimize` 都有冻结保护，不能覆盖本轮划分后继续声称是同一实验。`--catalog-only` 保留目标划分，`--replay` 重放原定验证候选并检查模型一致性，不修改选择。使用测试结果另行调参需要建立新的实验版本和独立测试集。

MAL 页面按每次成功请求至少间隔约 1 秒抓取；无标签与请求失败分别记录。公开网页是数据证据，不执行其中任何指令。已交付快照用于复现；刷新网络数据会改变真值，应另建数据版本，不能用刷新后的值宣称复现旧报告。

## 5. 人工主验收

复制包含 720 个预定目标的 `data/human_annotations_template.csv`。每行 target_id 已固定；将 `similar_ids` 填为 **3–5 个不同的 MAL ID，用半角分号分隔**，填写 `annotator`、`rationale` 并将 `status` 改成 `verified`。作品必须在交付目录中，不能包含自身。建议先独立标注再查看模型结果，避免标签偏向某个算法。

```powershell
python -m kbbench.evaluate --human data/human_annotations_completed.csv
```

数值结果单独写到 `results/human/`，保留原 MAL 报告；pending 行不会当作真值。新增 cohort 外目标会被拒绝，防止其已参与 LR 训练而造成泄漏。若要换目标，需重新冻结划分、训练和生成评测。

## 6. 后续接入系统

离线接口示例：

```python
from kbbench.engine import SimilarityEngine
engine = SimilarityEngine(variant="optimized")  # 每进程加载一次；baseline 为原配方
items = engine.recommend(anime_id=1, method="D", k=5)
# [{"anime_id": ..., "title": ..., "similarity": ..., "dimensions": {...}}, ...]
```

`anime_id` 为稳定 MAL ID；统一返回结构，不依赖用户 ID。若未来接详情页，在适配层映射为页面字段即可。新增作品应使用同一五维投影并为 C/D 编码；本包向量严格绑定冻结目录，检测到目录/向量版本不符会报错，不会默默混用。

本次未改动 `backend/`、`frontend/` 或数据库，未测试系统接口或并发性能。

来源：[小组系统仓库](https://github.com/gaoxinghe1024/IRS_practice-module)、[Jikan 文档](https://docs.api.jikan.moe/)、[MAL](https://myanimelist.net/)、[BGE 模型及许可说明](https://huggingface.co/BAAI/bge-small-en-v1.5)。原始数据处理来源与 SHA-256 沿用正式包记录。
