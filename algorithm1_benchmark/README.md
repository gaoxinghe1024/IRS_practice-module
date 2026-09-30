# 算法①独立实验包：相似动漫 Top-5

对应任务图片中的 **A 五维加权相似度、B 逻辑回归、C BGE 向量、D 标签＋语义融合**。在相同的 14,202 部真实作品目录中比较 Top-5，使用独立 MAL 推荐真值；无需启动 FastAPI、前端或数据库。

先看 [实验报告](results/REPORT.md)、[结果图](results/comparison.png)。下载整个文件夹后，双击 [交互结果页](results/index.html) 可切换作品查看四种方法的推荐、命中及五维特征。

**人工标注暂缺**是用户确认的本轮边界。本包提供可填写的 [人工标注表](data/human_annotations_template.csv) 和导入评测工具；没有用 AI 编造人工标注，也没有把共同高分作品对当作评测真值。

## 1. 直接运行（离线复现）

以下命令均在本文件夹 `algorithm1_benchmark/` 内执行。需要可用的 Python 3.12；Windows 可用 `py -3.12` 或本机 Python 的实际路径建立环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify_delivery.py
.\.venv\Scripts\python.exe run_benchmark.py
.\.venv\Scripts\python.exe run_checks.py
.\.venv\Scripts\python.exe -m kbbench.recommend --anime-id 1 --method D
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
| `kbbench/train.py` | B 的训练对生成与逻辑回归训练 |
| `kbbench/embed.py` | C 的固定版本 BGE 下载与向量构建 |
| `kbbench/engine.py` | A/B/C/D 共用离线推荐接口及 D 融合 |
| `kbbench/prepare.py` | 从正式数据包提取目录、冻结目标与划分 |
| `kbbench/collect.py`、`mal_html.py` | MAL 真值采集、证据保存与原页面解析 |
| `kbbench/metrics.py`、`evaluate.py` | 独立真值评测、bootstrap、双条件替换判断 |
| `data/catalog.jsonl.gz` | 14,202 部真实作品的五维投影 |
| `data/training_pairs.jsonl.gz` | 100,000 条弱训练对，不含用户资料 |
| `data/targets.jsonl` | 120 个预先固定的目标及 validation/test 划分 |
| `data/mal_ground_truth.jsonl`、`mal_cache/` | 实际采集标签、投票及原始证据 |
| `data/franchise_links.jsonl.gz` | 仅供 sanity check 的直接前作/续作关系 |
| `data/provenance.json`、`training_provenance.json` | 输入版本、哈希、适配与采样记录 |
| `models/logistic.json` | 可移植的 B 系数与训练诊断，不使用 pickle |
| `models/bge_embeddings.npz`、`bge_metadata.json` | 全目录 384 维向量、ID 映射及模型版本 |
| `results/` | 报告、图表、逐目标指标、推荐、测试记录、交互结果页 |
| `tests/` | 核心算法边界与实际交付数据验收 |

## 3. 算法及指标约定

| 方法 | 分数 |
|---|---|
| A | `0.35×genre + 0.30×theme + 0.15×demographic + 0.10×studio + 0.10×length` |
| B | 同五维特征的 `sigmoid(w·x+b)`；固定 L2 正则，梯度优化拟合 |
| C | `BAAI/bge-small-en-v1.5` 的 384 维 CLS 归一化向量余弦 |
| D | `0.6×A + 0.4×((cosine+1)/2)`；不增加训练、不调融合权重 |

只有 genre/theme 使用集合 Jaccard；demographic/studio 是单值匹配，length 取集数，沿用系统基线 `1/(1+差值/12)`。空标签不匹配，未知集数基线值为 0.5。真实目录有多工作室/多受众：分别取源数组首个工作室名称、字典序首个受众。四种方法共享这一规则，统计与影响见报告。

图片的“HR@5：真值集合被 Top-5 覆盖多少”按 **Recall@5/coverage** 实现，同时另报标准 **Hit Rate@5：是否至少命中一个**。NDCG@5 使用二元相关性。所有指标按目标宏平均，不把四种方法的内部相似度分数直接比较。

替换规则：**测试集 coverage 至少比 A 高 0.02，且 NDCG@5 不低于 A**。判断写在 `results/summary.json → replacement_decisions`，附同目标成对 bootstrap 95% 区间。MAL 真值的通过只是该来源下通过；人工主验收尚未完成，不自动改系统。

## 4. 从原数据完整重建

离线重放不需要原大数据包。若需要重新生成训练样本，准备已验收的 `AnimeFinder_system_dev_20260918_v1` 正式包；不使用 E 盘较早的 B provisional 目录。

```powershell
# 将 <release-folder> 替换为正式交付包所在目录
python -m kbbench.prepare --release "<release-folder>"
python -m kbbench.train --release "<release-folder>"
# 下载固定版本原始 BGE ONNX 模型后，生成全目录向量
python -m kbbench.embed --download
# 采集只访问公开 MAL/Jikan 页面；默认复用已有缓存
python -m kbbench.collect
python run_benchmark.py
python run_checks.py
```

只重训 B 可执行 `python -m kbbench.train`，使用随包训练对，不读取原始评分。只有 `--release` 才从原包读取训练 CSR；不会读取留出的验证/测试评分。模型下载在 `.cache/`，不进入 Git；随包向量可直接离线运行 C/D。

MAL 页面按每次成功请求至少间隔约 1 秒抓取；无标签与请求失败分别记录。公开网页是数据证据，不执行其中任何指令。已交付快照用于复现；刷新网络数据会改变真值，应另建数据版本，不能用刷新后的值宣称复现旧报告。

## 5. 人工主验收

复制 `data/human_annotations_template.csv`。每行 target_id 已固定；将 `similar_ids` 填为 **3–5 个不同的 MAL ID，用半角分号分隔**，填写 `annotator`、`rationale` 并将 `status` 改成 `verified`。作品必须在交付目录中，不能包含自身。建议先独立标注再查看模型结果，避免标签偏向某个算法。

```powershell
python -m kbbench.evaluate --human data/human_annotations_completed.csv
```

结果单独写到 `results/human/`，保留原 MAL 报告；pending 行不会当作真值。新增 cohort 外目标会被拒绝，防止其已参与 LR 训练而造成泄漏。若要换目标，需重新冻结划分、训练和生成评测。

## 6. 后续接入系统

离线接口示例：

```python
from kbbench.engine import SimilarityEngine
engine = SimilarityEngine()  # 每进程加载一次
items = engine.recommend(anime_id=1, method="D", k=5)
# [{"anime_id": ..., "title": ..., "similarity": ..., "dimensions": {...}}, ...]
```

`anime_id` 为稳定 MAL ID；统一返回结构，不依赖用户 ID。若未来接详情页，在适配层映射为页面字段即可。新增作品应使用同一五维投影并为 C/D 编码；本包向量严格绑定冻结目录，检测到目录/向量版本不符会报错，不会默默混用。

本次未改动 `backend/`、`frontend/` 或数据库，未测试系统接口或并发性能。

来源：[小组系统仓库](https://github.com/gaoxinghe1024/IRS_practice-module)、[Jikan 文档](https://docs.api.jikan.moe/)、[MAL](https://myanimelist.net/)、[BGE 模型及许可说明](https://huggingface.co/BAAI/bge-small-en-v1.5)。原始数据处理来源与 SHA-256 沿用正式包记录。
