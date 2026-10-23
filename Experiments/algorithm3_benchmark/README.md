# ICF + GA Benchmark

This folder is a **self-contained, re-runnable** minimal project: it keeps only the
code and data necessary to reproduce every comparison result of the ICF + GA
experiment (the four rerank arms: greedy / window random / pool random / GA).

After setup it requires no external dependencies and no access to the original
experiment repository.

---

## 1. Quick Start

```bash
# 1) Run the main experiment (20,000 users, about 80~90 minutes) -> results/ranks.npz
E:/anaconda3/python.exe run_experiment.py

# 2) Generate the comparison data for both parts
E:/anaconda3/python.exe analyze_icf_part.py      # -> results/icf_part.json
E:/anaconda3/python.exe analyze_ga_part.py       # -> results/ga_part.json
```

**Smoke run** (not the full 20k users; confirms the environment works):
```bash
E:/anaconda3/python.exe run_experiment.py --prefix 300 --tag smoke
# Evaluates only the first 300 users of the real cohort, about 4 minutes;
# the output ranks_smoke.npz does not overwrite the official results
```

**Dependencies**: `numpy 1.26.4`, `scipy 1.16.0`, `torch 2.5.1+cu121` (NCF/SVD use CUDA),
`pandas 2.2.2`.

---

## 2. File Inventory (one file, one job)

### Configuration and data

| File | Purpose |
|---|---|
| `config.py` | **The single parameter center**: data paths, candidate pool grid, GA weights, engine hyper-parameters, random seeds |
| `data_loader.py` | Data loading: training ratings CSR, id↔index mappings, item features, premiere years, held-out validation set, eligible user cohort |
| `data/` | Datasets (8 files required by the experiment, 340 MiB total) |

### Recall engines (the three curves of Part 1)

| File | Purpose |
|---|---|
| `engine_icf.py` | **ICF** item-based collaborative filtering (item-item similarity). Deterministic engine; retraining reproduces results bit-for-bit |
| `engine_svd.py` | **SVD** matrix factorization (users/items share a 32-dim latent space, torch) |
| `engine_ncf.py` | **NCF** neural collaborative filtering (MLP + negative sampling, torch) |
| `engine_registry.py` | Unified loading of the three engines and the full-catalogue scoring interface |

### Rerank layer (the four arms of Part 2)

| File | Purpose |
|---|---|
| `rerank_greedy.py` | **Greedy baseline**: Top-10 by pure relevance (= the GA's internal baseline, and = no rerank) |
| `rerank_random.py` | **Random control arms**: window random / pool random share one function (only the passed pool differs) |
| `rerank_ga.py` | **Genetic algorithm**: four-component weighted fitness + random-key encoding + tournament/crossover/mutation/elitism |
| `list_metrics.py` | List-level metrics: Jaccard similarity, diversity, novelty, freshness, plus two feature preprocessing routines |

#### Why there are **two** random control arms, and why only one function

| Arm | How the 10 items are picked | Candidate source | Question it answers |
|---|---|---|---|
| Greedy | Top-10 by pure relevance | Top-20 by relevance | What happens with no rerank |
| **Window random** | Pick 10 at random within the window | **Identical to GA** (top 20) | **GA's reranking ability** — excludes the contribution of "randomness itself" |
| **Pool random** | Pick 10 at random, ignoring relevance | Whole candidate pool (N items) | The **upper bound** of exposure structure (hit rate ≈ K/N) |
| GA | Four-component weighted fitness | Top-20 by relevance | The protagonist |

The two arms use **exactly the same algorithm**; only the pool passed in by the
caller differs, so `rerank_random.py` keeps a single function:
`rerank_random(pool_idx, pool_score, rng)`.
Making "window/pool" a function parameter would only create a switch that always
takes one of two values, which is harder to read.

### Experiment and artifacts

| File | Purpose | Output |
|---|---|---|
| `candidate_pool.py` | Candidate pool construction (1 positive + N−1 uniform random negatives) + per-user deterministic random streams for GA and the random arms | — |
| `run_experiment.py` | **Main experiment**: 10 pool levels × 2 seeds × 3 engines × **4 rerank arms** | `results/ranks.npz`<br>`results/experiment.json` |
| `analyze_icf_part.py` | **Part 1** comparison data: HR@5 / HR@10 / lift multiples for 3 engines × 10 levels | `results/icf_part.json` |
| `analyze_ga_part.py` | **Part 2** comparison data: four-arm comparison, benefit LTE/LTR/GOS, cost HR, paired significance, swap ratio | `results/ga_part.json` |

---

## 3. Meaning of Every Metric

**Benefit metrics (what GA is supposed to improve)**

| Metric | Formula | One-line meaning |
|---|---|---|
| `nov@K` novelty | Mean novelty of the top-K items, `novelty(i)=1−log(1+pop_i)/log(1+max_pop)` | How obscure the recommendation list is overall. A 0~1 decimal with little intuitive meaning |
| `lt@10` long-tail share | Share of the top 10 with novelty ≥ 0.6156 (catalogue median) | Turns "obscure on average" into "how many items are actually obscure" |
| `LTE@10` long-tail exposure | Σ(number of long-tail items in each user's top 10) | How many **items** of the 200,000 site-wide recommendation slots are long-tail |
| `LTR@10` long-tail reach rate | Users with ≥ 1 long-tail item in their top 10 ÷ all users | **How many users are actually pushed a long-tail item** (the headline benefit metric) |
| `GOS@10` GA objective score | 0.3×diversity + 0.2×novelty + 0.1×freshness | Exactly the 60% of the GA fitness that GA is responsible for |

**Cost metrics (what GA must not destroy)**

| Metric | Formula | One-line meaning |
|---|---|---|
| `HR@5` / `HR@10` | Share of users whose hidden item lands in the top 5 / top 10 | Recommendation accuracy; corresponds to the 0.4-weight relevance term of the GA fitness |
| Random baseline | `K / N` | Hit rate of pure blind guessing |
| Lift multiple | `HR@K ÷ (K/N)` | How many times better than blind guessing; **the only metric comparable across pool levels** |

### Definition of novelty (the sole basis for long-tail classification)

```
novelty(i) = 1 − log(1 + pop_i) / log(1 + max_pop)
```

- **pop_i** = number of times item i was rated in the **training set**;
  **max_pop** = the catalogue maximum.
- `log(1+·)` is used instead of dividing by popularity directly to compress the
  head — otherwise the single most popular item would push every other item's
  novelty close to 1, destroying discrimination.
- Range **0 ~ 1**: 0 = the most popular item in the catalogue, 1 = never seen in
  the training set.
- **Long-tail threshold = catalogue novelty median = 0.6156** (7,110 of 14,202
  items, 50.1%, are long-tail).
- Reference: **items users actually consumed have an average novelty of only
  0.157** — users themselves consume head items, so recommending the long tail
  means deviating from historical preference, which bounds GA's benefit.
- Popularity is extremely concentrated: **the top 1% of items account for 24.9%
  of ratings, the top 10% for 75.8%**.

### GA fitness weights (algorithm ③ slide convention)

| Component | Weight | Guaranteed by |
|---|---|---|
| Relevance | **0.4** | Recall engine (ICF/SVD/NCF); GA does not optimize it, and HR@5/HR@10 monitor the cost |
| Diversity | **0.3** | GA |
| Novelty | **0.2** | GA |
| Freshness | **0.1** | GA |

`Fitness = 0.4×relevance + 0.3×diversity + 0.2×novelty + 0.1×freshness`, all four
min-max normalized to [0,1] within the pool.

**Why "benefit" in the report only counts the last three components?** Because
the 0.4 relevance weight is the recall engine's job: GA's rerank window has only
20 items, all at the top of relevance, so it cannot pull items from outside the
window in and thus cannot "improve relevance". The only thing GA should be
evaluated on is the **0.6** it can actually move — reported as **GOS@10**;
relevance changes are treated as **cost** and monitored separately with HR@5/HR@10.
**Recording benefit and cost separately avoids "using one weighted total to prove
GA is better, letting the choice of weights decide the conclusion".**

---

## 4. Protocol Essentials (three conventions to know before reading the report)

1. **Candidate pool N**: the candidate set = 1 hidden positive + (N−1) uniform
   random negatives. 10 levels; the full-catalogue level averages 14,097.9 items.
   **HR across levels must not be compared directly** — read the lift multiple instead.
2. **Greedy ≡ no rerank**: the recall list is already sorted by descending
   relevance, so both select exactly the same items. The "ICF baseline" in the
   report is this Top-10.
3. **Rerank window = 20**: GA reranks only within the top 20 by relevance and
   outputs the top 10. This explains why GA's benefit at the full-catalogue
   level is only of that magnitude — long-tail candidates never enter the window.
   **The window random arm uses the same window**, so its comparison with GA
   rules out any "different candidate source" confound.
4. **All arms output in "descending relevance" order**: first pick 10 items by
   each arm's own rule, then present them sorted by relevance — otherwise
   order-dependent metrics like HR@5 are not comparable across arms.

---

## 5. Reproducibility

- **ICF is a deterministic engine**: after a rerun its ranks must match
  historical results **bit-for-bit**; `run_experiment.py` verifies this
  automatically and exits with an error on any mismatch (meaning the environment changed).
- **GA random stream**: `default_rng([sampling seed, pool size, engine, user])`,
  independent per user. Keyed by the pool size itself (sentinel key `10^7` for
  the full catalogue) rather than the grid index, so results are unchanged when
  the grid is split across processes.
- **Random-arm random stream**: `default_rng([sampling seed, pool size, engine,
  user, arm])`, where `arm` = 1 window random / 2 pool random. **Independent of
  the GA stream**.
  ⚠️ **Never append `arm` to the `user_rng` seed list** — that changes the
  SeedSequence entropy, shifts all historical GA results wholesale, and makes
  every published number unreproducible.
  Measured: after adding the two new streams, all metrics across the other 9
  levels match historical results **bit-for-bit (max|Δ| = 0)**, except
  "candidate pool 2000" where the winner flips due to the small sample.
- **Sampling seeds**: [7042, 7043], consistent with the historical experiments.
- **Smoke/full correspondence**: `--prefix 300` takes **the first 300 users of
  the real 20,000-user cohort** (not a separately drawn small cohort), so
  results can be compared user by user.
- **Paired tests**: per-user differences under the same user, same candidate
  pool, same engine, same seed; Wilcoxon signed-rank for benefit metrics,
  McNemar for hit rates, all with 95% CI and Cohen's d_z.

---

## 6. Source of the Current Result Data

`results/ranks.npz` and `results/experiment.json` come from the **unified rerun
of 2026-10-02** (20,000 users, seeds [7042, 7043], all three engines retrained).
Running `run_experiment.py` in this folder once end-to-end reproduces equivalent
results; `analyze_*.py` read from these two files
programmatically, with zero manual transcription throughout.

Key verified numbers (bit-identical to the historical published report):

| Protocol | ICF baseline | ICF + GA |
|---|---|---|
| HR@5 (full catalogue) | 15.35% | 14.53% |
| HR@10 (full catalogue) | 20.93% | 19.51% |
| Long-tail exposure LTE@10 | 1,829 items | 3,161 items (+72.8%) |
| Long-tail reach LTR@10 | 2.33% | 3.45% (+222 users) |
| GA objective score GOS@10 | 0.2799 | 0.2923 (+4.4%) |
