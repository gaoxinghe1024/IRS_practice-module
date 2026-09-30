"""Generate the Chinese report, portable charts and an offline comparison viewer."""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ROOT, read_json, read_rows
from .engine import METHODS


def report(output=None):
    output = Path(output) if output else ROOT / "results"
    summary = read_json(output / "summary.json")
    source = summary["source"]
    rows = {row["method"]: row for row in summary["summaries"] if row["source"] == source and row["split"] == "test"}
    data = read_json(ROOT / "data/provenance.json")
    training = read_json(ROOT / "models/logistic.json")
    embedding = read_json(ROOT / "models/bge_metadata.json")
    audit = read_json(ROOT / "data/collection_audit.json")
    counts = Counter(row["status"] for row in audit)
    mal_labels = list(read_rows(ROOT / "data/mal_ground_truth.jsonl"))
    labels = [row for row in read_rows(output / "ground_truth_used.jsonl") if row["source"] == source]
    counts_by_group = Counter(row["sampling_group"] for row in mal_labels)
    predictions = list(read_rows(output / "predictions.jsonl"))
    colors = ["#183e7a", "#3c78bb", "#45a69d", "#d5a42e"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, metric, title in zip(axes, ["coverage_at_5", "ndcg_at_5"], ["Ground-truth coverage @5 (slide HR@5)", "NDCG @5"]):
        values = [rows[method][metric] for method in METHODS]
        bars = axis.bar(list(METHODS), values, color=colors, width=0.6)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4)
        axis.set_ylim(0, max(0.05, max(values) * 1.35))
        axis.set_title(title, fontsize=11)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Algorithm 1 | {source} | test queries n={rows['A']['n_targets']}", fontsize=13)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    svg_path = output / "comparison.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    lines = ["# 算法①：四种相似动漫算法的独立实验报告", "",
             f"运行时间（UTC）：{summary['created_at']}。数据版本：`{data['catalog_version']}`。",
             "", "## 1. 结论", "",
             ("本次已完成 A/B/C/D 四种算法的真实计算、同候选库 Top-5 对比和独立来源评测。用户确认暂时没有人工标注，因此本轮依据 MAL 推荐做评测，人工主验收仍待补齐。未修改或启动应用系统。" if source != "human" else "本报告使用导入表中已验证的人工标注，比较相同目标的 A/B/C/D Top-5 结果。未修改或启动应用系统。"), "",
             f"测试集有效目标 **{rows['A']['n_targets']}** 个，每个目标从 **{summary['candidate_catalog_size'] - 1:,}** 个非自身作品中选择 Top-5。结果只代表本次有标签的目标，不外推为全站效果。", "",
             "| 方法 | 图片 HR@5＝覆盖率 | NDCG@5 | 标准 Hit Rate@5 | 相对 A 覆盖率变化 | 达到图片双条件 |",
             "|---|---:|---:|---:|---:|---|"]
    for method in METHODS:
        row = rows[method]
        decision = summary["replacement_decisions"].get(method)
        delta = f"{decision['coverage_delta_pp']:+.2f} pp" if decision else "基线"
        passed = ("是（仅本轮来源）" if decision["passes"] else "否") if decision else "—"
        lines.append(f"| {method} | {row['coverage_at_5']:.2%} | {row['ndcg_at_5']:.4f} | {row['hit_rate_at_5']:.2%} | {delta} | {passed} |")
    passing = [method for method, decision in summary["replacement_decisions"].items() if decision["passes"]]
    if passing:
        lines += ["", f"本轮来源下达到双条件的候选：{', '.join(passing)}。仍需结合人工主验收与置信区间决定是否接入。"]
    else:
        best = max(("B", "C", "D"), key=lambda method: (rows[method]["coverage_at_5"], rows[method]["ndcg_at_5"]))
        delta = summary["replacement_decisions"][best]["coverage_delta_pp"]
        lines += ["", f"**本轮建议保留 A：B/C/D 均未达到双条件。** 覆盖率最高的候选为 {best}，相对 A 提升 {delta:.2f} 个百分点。不能把“分数最好”直接写成“满足替换门槛”。绝对覆盖率较低，说明仅有五维属性与 MAL 用户相似偏好仍有明显距离；并非已经达到可上线质量。"]
    lines += ["", "替换规则严格为：覆盖率至少提升 **2 个百分点**，且 NDCG@5 不下降；两项必须同时满足。不会因为训练 AUC 较高、某个示例看起来更好，或仅命中率上升就建议替换。", "",
              "![四种算法测试集对比](comparison.png)", "", "### 成对 bootstrap 不确定性", "",
              "固定随机种子，对同一批目标重采样 3,000 次；下表是 95% 差值区间，不是总体效果保证。区间跨 0 时证据不稳定。没有为选出赢家重新调参或更换目标。", "",
              "| 方法相对 A | 覆盖率变化 95% CI（百分点） | NDCG 变化 95% CI |", "|---|---|---|"]
    for method, decision in summary["replacement_decisions"].items():
        cov = decision["paired_95pct_delta_ci"]["coverage_at_5"]
        ndcg = decision["paired_95pct_delta_ci"]["ndcg_at_5"]
        lines.append(f"| {method} | [{100*cov[0]:+.2f}, {100*cov[1]:+.2f}] | [{ndcg[0]:+.4f}, {ndcg[1]:+.4f}] |")
    lines += ["", "## 2. 数据、抽样及独立真值", "",
              "原 E 盘目录是 B 阶段 provisional 数据；采用后续 C 盘正式交付版本。原始正式包已通过 523 个文件哈希和 7 项结构检查。本文件夹只提取算法必需的五维元数据，不复制全部用户资料或 2,395 万行评分。提取规则与输入哈希见 `data/provenance.json`。", "",
              "先固定 120 个目标：20 个演示用知名作品，另从会员数四分位各随机抽取 25 个作品。按目标 ID 的固定哈希划分 validation/test；全部目标 ID 都从逻辑回归训练对的两个端点排除。四种方法参数预先固定，validation 只作诊断展示，最终结论只使用 test。", "",
              f"MAL 采集：{counts['eligible']} 个目标具备至少 3 条目录内推荐，{counts['insufficient_labels']} 个不足 3 条，{counts['request_failed']} 个请求/页面解析失败；固定目标总数 {len(audit)}。所有被排除目标均保留审计记录，没有根据算法成绩删除目标。", "",
              "优先读取 Jikan 的 MAL 推荐接口；失败时读取 MAL 原站 userrecs 页面。按真实推荐人数降序、编号升序取最多 5 条目录内作品，至少 3 条才能进入评测。原始 API 响应或网页 gzip、访问时间、URL 和 SHA-256 一同交付。页面解析只读取推荐卡片及其人数，不把侧栏作品链接当推荐。", "",
              "这是不完整的正例集合：未列入前 5 的作品仍可能相似，自动指标会低估其他合理答案。热门作品更容易取得推荐，自动评测不能替代人工标注。", "",
              "| 抽样组 | 有效 MAL 目标数 |", "|---|---:|"]
    for group in ["anchor", "popularity_q1", "popularity_q2", "popularity_q3", "popularity_q4"]:
        lines.append(f"| {group} | {counts_by_group[group]} |")
    lines += ["", "分组的 validation/test 指标完整保存在 `summary.json → strata`，不把知名作品的结果冒充均匀随机抽样效果。", "",
              "系列关系仅用于 sanity check：使用正式目录的直接 Prequel/Sequel 边，单独报告，不混入 MAL/人工真值、不参与训练标签、不作为替换依据。", "",
              "## 3. 四种算法与数据适配", "",
              "共同五维输入是 genre、theme、demographic、studio、episodes。genre/theme 去重后做集合 Jaccard；受众取排序后的首个非空值，工作室取源数组的首个名称。名称中的逗号保留；单值采用忽略大小写的精确匹配。已知为空与缺失均不生成匹配奖励。", "",
              f"数据中 {data['audit']['multi_demographic']} 部有多受众、{data['audit']['multi_studio']} 部有多工作室。实际词表为 {data['vocabulary_sizes']['genres']} 个 genre、{data['vocabulary_sizes']['themes']} 个 theme，不硬编码幻灯片示例词表大小。完整缺失统计见下表；这些投影规则对四种算法完全相同。", "",
              "| 字段 | 空/缺失数量 |", "|---|---:|"]
    for field in ["genres", "themes", "demographic", "studio", "episodes"]:
        lines.append(f"| {field} | {data['audit']['missing_or_empty_' + field]} |")
    lines += ["", "- **A**：`0.35×genre + 0.30×theme + 0.15×demographic + 0.10×studio + 0.10×length`。集数相似度沿用仓库 `1/(1+abs(e1-e2)/12)`；未知集数沿用 0.5。准确名称是五维加权相似度，后三维并非集合 Jaccard。",
              "- **B**：与 A 完全相同的五个数值特征，`sigmoid(w·x+b)` 作为排序分数。固定 L2 正则 C=1，用 L-BFGS 的梯度优化拟合逻辑回归。系数不强制非负或和为 1；该概率针对弱标签，不能解释为真实相似概率。",
              "- **C**：仅把五个维度拼成英文属性描述，不加入标题、简介、关系或评分。用原始 BGE small en v1.5 ONNX 浮点权重，CLS 池化、L2 归一化，余弦相似度排序。英文模型符合当前英文属性；未验证中文自然语言检索。",
              "- **D**：标签路径固定采用 A；语义余弦统一映射 `(cos+1)/2` 到 [0,1]，`0.6×A + 0.4×semantic`。不按每个查询重新做 min-max，不额外训练、不按结果调融合权重。",
              "", "所有方法按未四舍五入分数降序、MAL ID 升序稳定排序，排除目标自身。新用户无需历史；本次全部评测目标也未出现在 LR 训练对中。新增作品需具有同样元数据，C/D 对新增作品还需生成对应向量；交付向量绑定本次冻结目录。", "",
              "## 4. 训练与向量构建的实际记录", "",
              f"弱监督数据为 50,000 个共同高分正例和 50,000 个随机负例；用户评分只读取 `ratings_train.npz` 的原始 1–10 分。每个访问用户最多贡献一个正例，评分阈值 ≥8；随机负例排除已采样正例，但仍可能含假负例。按无向作品对的固定哈希分为 {training['training_pairs']:,} 个拟合样本和 {training['proxy_validation_pairs']:,} 个代理验证样本。训练样本交付只包含作品对，不含评分用户编号或用户名；MAL 原网页证据保留公开推荐者署名。", "",
              f"代理验证 AUC={training['proxy_validation_auc']:.4f}，LogLoss={training['proxy_validation_log_loss']:.4f}。这两个数字不是相似动漫排序指标。", "",
              f"学习系数（genre/theme/demographic/studio/length）：`{[round(value, 6) for value in training['coefficients']]}`；截距 `{training['intercept']:.6f}`。集数系数为负也原样保留，反映共同喜好弱标签与相似性目标可能不一致，没有在看到评测结果后手工改权重。", "",
              f"BGE 模型：`{embedding['model']}`，revision `{embedding['revision']}`；384 维，CPU {embedding['threads']} 线程，batch={embedding['batch_size']}，最大长度 {embedding['max_length']}；截断 {embedding['truncated_titles']} 部。全量编码耗时 **{embedding['encode_seconds']:.1f}s**。模型与 tokenizer 哈希、向量哈希见 `models/bge_metadata.json`。", "",
              f"本机环境：Python {summary['environment']['python']}，{summary['environment']['platform']}。依赖版本写入 `summary.json` 和 `requirements.lock.txt`。实验引擎加载约 {summary['timing']['load_seconds']:.3f}s；一次共同目标计算四种方法并排序的中位延迟约 {summary['timing']['all_four_methods_p50_ms']:.2f}ms、P95 约 {summary['timing']['all_four_methods_p95_ms']:.2f}ms。它们是本地离线耗时，不是系统接口或并发性能。", "",
              "## 5. 验收与复现", "",
              "运行命令见上级 `README.md`。`test_core.py` 覆盖空值、单值/集合区别、长度、归一化、稳定排序、自排除、指标定义、双条件门槛和人工标注导入；`test_artifacts.py` 核对真实训练隔离、原始推荐证据、指标复算与离线推荐。完整运行输出保存到 `tests.txt`。`DELIVERY_MANIFEST.json` 用于交付文件哈希校验。", "",
              "人工标注表见 `data/human_annotations_template.csv`。每个目标填写 3–5 个目录内作品 ID（分号分隔）、标注人、理由并设 status=verified；不要根据四种算法结果倒推标签。建议两位熟悉作品的同学独立标注后协商争议，再运行 `python -m kbbench.evaluate --human <完成表.csv>`，另写到 `results/human/`。", "",
              "## 6. 明确边界", "",
              "人工主验收尚未完成；MAL 推荐有人群与热度偏差，长尾无标签且未被当作负例；多工作室被压缩为单值；缺失属性可能造成相同标签文本；训练共同喜好并不等价于内容相似。保留这些限制是为了让结果可解释，而不是将低分改成预期结论。未执行系统联调、线上替换或系统性能测试。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    # Local-only browser report. All external text is rendered with textContent, never innerHTML.
    payload = json.dumps(dict(summary=summary, predictions=predictions, labels=labels), ensure_ascii=False).replace("<", "\\u003c")
    viewer = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>算法① 独立实验对比</title><style>
body{font:16px/1.65 system-ui,sans-serif;background:#f4f7fb;color:#1c2940;margin:0}main{max-width:1280px;margin:auto;padding:32px}h1{color:#173f83}select{padding:10px;max-width:100%;font:inherit}section{background:white;padding:24px;border-radius:12px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.card{border-top:4px solid #316ec7;padding:16px;background:#fff;border-radius:8px}li{margin-bottom:14px}.hit{color:#167748;font-weight:600}small{color:#62728c}img{max-width:100%}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}@media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:500px){.grid{grid-template-columns:1fr}}
</style><main><h1>算法① · 相似动漫实验</h1><p>14,202 部候选作品 · A/B/C/D 共享五维输入 · Top-5 排名。人工验收待补；本页展示 MAL 推荐评测。</p>
<section><h2>测试集结果</h2><div id="metrics"></div><p>图片 HR@5 指覆盖率（Recall@5）。替换要求：覆盖率至少 +2 个百分点，且 NDCG 不下降。</p><img src="comparison.png" alt="算法对比图"></section>
<section><h2>逐目标检查</h2><select id="target" aria-label="选择目标作品"></select><p id="truth"></p><small>绿色表示命中采集到的 MAL 真值；未命中不等于不相似。分数是各模型内部值，不能跨模型直接比较。</small></section><div class="grid" id="cards"></div>
<section><a href="REPORT.md">完整报告</a> · <a href="metrics.csv">指标表</a> · <a href="../data/human_annotations_template.csv">人工标注表</a></section></main>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent);const names={A:'A · 五维加权',B:'B · 逻辑回归',C:'C · BGE 语义',D:'D · 标签＋语义'};
const table=document.createElement('table');let header=document.createElement('tr');for(const text of ['方法','覆盖率@5','NDCG@5','命中率@5','目标数']){let e=document.createElement('th');e.textContent=text;header.append(e)}table.append(header);
for(const row of data.summary.summaries.filter(x=>x.source===data.summary.source&&x.split==='test')){let tr=document.createElement('tr');for(const text of [names[row.method],(100*row.coverage_at_5).toFixed(2)+'%',row.ndcg_at_5.toFixed(4),(100*row.hit_rate_at_5).toFixed(2)+'%',row.n_targets]){let td=document.createElement('td');td.textContent=text;tr.append(td)}table.append(tr)}document.getElementById('metrics').append(table);
const select=document.getElementById('target');for(const row of data.predictions.filter(x=>x.method==='A')){let o=document.createElement('option');o.value=row.target_id;o.textContent=row.target_id+' · '+row.target_title;select.append(o)}
function draw(){const aid=Number(select.value),truth=data.labels.find(x=>x.target_id===aid),ids=new Set(truth?truth.relevant_ids:[]);document.getElementById('truth').textContent=truth?'MAL 真值 ID：'+truth.relevant_ids.join('、')+'（'+truth.split+'）':'无足够 MAL 标签；仅有系列关系检查。';const cards=document.getElementById('cards');cards.replaceChildren();for(const row of data.predictions.filter(x=>x.target_id===aid)){const card=document.createElement('div');card.className='card';let h=document.createElement('h3');h.textContent=names[row.method];card.append(h);let ol=document.createElement('ol');for(const item of row.items){let li=document.createElement('li');li.className=ids.has(item.anime_id)?'hit':'';let title=document.createElement('div');title.textContent=item.title+' ['+item.anime_id+']';let score=document.createElement('small');score.textContent='score '+item.similarity.toFixed(5)+' · 特征 '+Object.values(item.dimensions).map(v=>v.toFixed(2)).join(' / ');li.append(title,score);ol.append(li)}card.append(ol);cards.append(card)}}select.addEventListener('change',draw);draw();
</script></html>'''
    if source == "human":
        viewer = viewer.replace("人工验收待补；本页展示 MAL 推荐评测。", "本页展示导入表中已验证的人工标注评测。")
        viewer = viewer.replace("MAL 真值", "人工真值").replace("采集到的 MAL 真值", "导入的人工真值")
    (output / "index.html").write_text(viewer.replace("PAYLOAD", payload), encoding="utf-8")
    print(f"Wrote {output / 'REPORT.md'}, charts and offline index.html", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    report(parser.parse_args().output)
