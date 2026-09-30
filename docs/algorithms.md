# Algorithms

Formulas, hyperparameters (defaults from `configs/base.yaml`), fallbacks, and the differences between training
and serving. Every formula here matches the implementation's docstring; the module is named next to each one.
Results are in `docs/evaluation-results-*.md`.

Shared building blocks:

* **Catalogue snapshot** (`app/recommendation/features.py`): numpy arrays and sparse tag matrices built from the
  same catalogue and tag tables offline and online. Unknown values stay NaN or empty; nothing is imputed.
* **Constraint engine** (`app/recommendation/constraints.py`): all hard rules are vectorised masks. A missing value
  never satisfies a hard filter. Relaxations are suggestions only (see `docs/requirements-resolution.md`).
* **Explanations** (`app/recommendation/explanations.py`): reasons are computed from data. Each has a `code`,
  `params` and default English `text`; an LLM never writes them.

---

## 1. Similar titles (detail page)

Candidate pool: served, recommendable titles except the query. Output Top-5 (`similar.k`, up to `max_k = 10`).
The user may hide titles they have already rated or completed (`hide_watched`).

### A. Weighted attributes (default) — `similarity/weighted.py`

For each set dimension d in {genre, theme, demographic, studio}:

    J_d(i, j) = |A_d(i) ∩ A_d(j)| / |A_d(i) ∪ A_d(j)|      available only when both sets are non-empty
    L(i, j)   = exp(-|episodes_i - episodes_j| / τ)          τ = 12, available when both counts are known

    Sim_tag  = Σ_d w_d · avail_d · sim_d / Σ_d w_d · avail_d
    coverage = Σ_d w_d · avail_d / Σ_d w_d

Weights: genre 0.35, theme 0.30, demographic 0.15, studio 0.10, length 0.10. Two empty sets are *not* similar.
A pair with no comparable dimension is `insufficient_metadata` and is not ranked. Score and popularity are not
part of the formula. Ties are broken by item key, so results are deterministic. Reasons: shared tags, same
demographic or studio, similar length, the dimensions that could not be compared, and low coverage.

### B. Logistic regression — `similarity/logistic.py`

Features: the five `sim_d` values, the five availability flags and coverage (the same inputs as A). Positives come
only from similarity labels (human or a declared weak source). Negatives are 20 sampled unlabelled pool items per
query ("sampled_unlabelled", not proven dissimilar). Queries are split 60/20/20 by franchise component (from the
relation graph) so related titles never straddle splits. Standardisation and fitting use train only. `C` is
chosen from {0.01, 0.1, 1, 10} by validation NDCG@5. The output is the decision function (uncalibrated).
**Data gate:** at least 20 training and 5 validation query titles; below that the scorer is `limited_by_data`
and not served.

### C. BGE attribute embeddings — `similarity/embedding.py`

The five attributes are rendered with a fixed, versioned text template. Missing attributes are omitted, never
written as a shared "unknown". The text is encoded with `BAAI/bge-small-en-v1.5` (384-d, 512-token limit, English)
and compared by the cosine of L2-normalised vectors. Cosine is a similarity, not a probability.

### D. Tag × semantic hybrid

    Sim_hybrid = 0.6 · Sim_tag + 0.4 · clip((cos + 1) / 2, 0, 1)

When either side lacks an embedding, the weight moves to the tag part and the degradation is recorded.

A **replacement rule** decides whether B/C/D may replace A as the default: Hits@5 at least +2 points and NDCG@5
not lower than A, on the *validation* query group. Test numbers are only reported. `semantic_synopsis`
(synopsis embeddings) uses different information and is reported as a separate variant.

---

## 2. Collection recommendations — `app/recommendation/collection/`

**Profile** (`profiles.py`): per namespace, the share of collection titles carrying each tag, plus the
typical episode count. Titles the user rated low (<= 5, or more than 1.5 below their own mean) or marked not-interested are
skipped from the profile, and the skip is reported.

**Fixed rule layer:** served and recommendable, not already in the collection, not excluded, and passing the
user's filters with AND semantics: keywords (each must match a genre or the title), include tags, minimum
score, maximum episodes, types. An empty strict result returns `status: no_match` plus relaxation options.
Choosing an option re-runs the request with the rule marked "relaxed by you".

| Scorer | Formula / model | Role |
|---|---|---|
| `collection_profile_v1` (default) | `affinity_d(j) = Σ_{t∈tags_d(j)} profile_d(t) / max(1, |tags_d(j)|)`; `profile_match` = availability-weighted mean of the affinities and length fit; `score = α · profile_match + (1-α) · score_j/10`, α = 0.7 | served |
| `manual_slides_v0` | `score_j/10 + 0.15 · tag_hits` (the slides' formula, reproduced as written; its scale grows with tag counts) | baseline |
| GBDT | `HistGradientBoostingClassifier` (200 iterations, lr 0.05) over collection-preference features plus missingness masks; trained on leave-one-out cases (hidden item vs 4 sampled unobserved items per positive), grouped by owner | alternative, needs collection events |
| Two-tower | collection tower = pooled content vectors of the visible items; item tower = the title's own tags, length and score bucket; L2-normalised, `score = temperature · dot`; sampled-softmax loss | extension, deployed only with >= 10,000 collection interactions **and** a validation gain |

---

## 3. Personalised home — `app/recommendation/recall/`, `app/recommendation/rerank/`

### Cold start and the warm gate (`services/recommend.py:home`)

Let `n` = the number of the user's ratings on titles the CF model knows. Gate `G = 8`.

| Mode | Condition | Candidates and relevance |
|---|---|---|
| `cold` | n = 0 | content score from onboarding tags / saved collections; with no profile at all, a quality-controlled popular list (`popular_quality = 0.5 · minmax(log members) + 0.5 · minmax(score)`, score >= median) |
| `partial` | 0 < n < G | `rel = λ · cf + (1-λ) · content`, `λ = n / G`, over the union of both top lists |
| `warm` | n >= G | CF prediction, normalised `(r̂ - 1) / 9` |

The home **content score** is `(1 - g) · profile_match + g · popular_quality` (`home_content_scores`). On real data,
pure tag matching ranked the held-out title almost never in the top 10 (it surfaces obscure titles that happen
to share tags), so `g` keeps the cold-start list quality-controlled, as the brief's cold-start rule asks. `g` is
chosen on validation positives with histories of 1 / 3 / 5 ratings (`home.content_popularity_weight`, set by
`pipelines.apply_tuning`). Items ranked this way carry the reason "Ranked among popular, well-rated titles".

Already rated, completed and excluded titles are removed by the rule layer. Top `candidate_k = 20` candidates go
to the list optimiser; K = 10 are shown.

### A. ICF, exact adjusted cosine (default recall) — `recall/icf.py`

Training on the ranking train split only:

    μ_u = mean of u's training ratings;  a_ui = r_ui - μ_u   (an observed mask M keeps entries whose centred value is 0)
    s_raw(i,j) = Σ_{u∈U_ij} a_ui a_uj / sqrt(Σ_{U_ij} a_ui² · Σ_{U_ij} a_uj²)      U_ij = users who rated both
    kept when c_ij = |U_ij| >= 3;  s = s_raw · c_ij / (c_ij + 10);  top 150 positive neighbours per item

The denominator uses the co-rated users only, not full column norms. It is computed blockwise with sparse
products: `N = A_bᵀA`, `C = M_bᵀM`, `P = (A∘A)ᵀM`, `den = sqrt(P_ij · P_ji)`.

Prediction:

    r̂(u,i) = m_u + Σ_{j∈R_u∩N(i)} s(i,j)(r_uj - m_u) / (Σ |s(i,j)| + λ),  clipped to [1, 10]
    m_u    = (n_u · mean_u + k · μ_train) / (n_u + k)

Titles without neighbour support get no ICF prediction. Ratings below `m_u` contribute negatively.

* With **λ = 0 and k = 0** this is exactly the brief's formula (`icf_exact` in the reports).
* **k (user-mean prior weight)** shrinks the reference level of a *short* history towards the global train mean
  `μ_train` (stored in the ICF artifact). Without it, a new user who rated eight favourites 9/10 has every centred
  rating equal to 0: every prediction equals 9.0, and the list order is decided by tie-breaks. A 9 among 10s would
  also count as negative evidence. For long offline histories the effect is small.
* **λ (damping)** pulls predictions that rest on little neighbour support towards `m_u`.
* Both are chosen on **validation** positives of a disjoint 20 % user group, λ on full histories, then k on
  histories cut to 5 / 8 / 15 ratings and the full history. `python -m pipelines.apply_tuning --mode <mode>`
  copies the chosen values into `configs/<mode>.yaml`, and the config line names the evaluation run that produced
  them.

Reasons name the user's own supporting ratings, the similarity and the number of co-raters, never another
user's history.

### B. Biased MF — `recall/matrix_factorization.py`

    r̂(u,i) = μ + b_u + b_i + p_u · q_i
    loss = Σ (r_ui - r̂)² + reg · (|p_u|² + |q_i|² + b_u² + b_i²)

32 factors, reg 0.04, mini-batch SGD with lazy (row-wise) Adam, early stopping on a 2 % slice carved from train.
Application users are folded in by ridge regression with the item side fixed (`fold_in_reg = 1.0`). This needs
at least 3 known items; otherwise MF returns nothing and the service falls back to ICF.

### C. NCF, explicit variant — `recall/ncf.py`

GMF (element-wise product) and MLP (concatenated embeddings → 64-32-16) branches are fused into one logit, and
`r̂ = 1 + 9 · sigmoid(logit)`. Huber loss, dropout 0.2, weight decay 1e-5, early stopping. The model is ID-only:
an application user absent from training is `unsupported_new_user`, and the service falls back to ICF. No random
user vector is invented.

### List optimisation — `rerank/objective.py`, `rerank/ga.py`

Over the fixed candidate pool (the same pool and scores for every optimiser):

    Rel(L)   = mean normalised relevance
    Div(L)   = mean over valid pairs of (1 - Sim_tag)                    (no valid pair → unavailable)
    Nov(i)   = 1 - log(1 + train_count_i) / log(1 + max_train_count)   (all counts zero → unavailable)
    Fresh(L) = share of items with >= 1 genre/theme not in the user's visible history
                                                                        (empty history → unavailable)
    F(L) = Σ_t w_t · term_t / Σ_t w_t over available terms;  w = 0.4 / 0.3 / 0.2 / 0.1

| Optimiser | Definition |
|---|---|
| `relevance_greedy` | top-K by relevance (with monotone relevance this equals no-rerank; reported as an alias) |
| `mmr_lambda_0.7` | maximal marginal relevance, a separately named baseline |
| `ga_list` | chromosome = K distinct pool positions; population 64, up to 80 generations, tournament 3, elitism 2, crossover 0.8 (keep shared genes, fill from both parents), mutation 0.2 (replace one gene), repair to K unique genes, early stop after 20 stale generations, wall-clock budget 250 ms online; initial population seeded with greedy and MMR (λ 0.8 / 0.6 / 0.4) lists; the best-so-far individual is kept |
| `exhaustive_oracle` | every K-subset of a small pool (tests and optimality gap) |

The GA optimises this list objective inside the pool; it is not a catalogue-wide optimum, and a higher fitness is
not evidence of better accuracy. Item reasons include novelty, fresh tags, and the list-role delta (how much F
drops without the item; an explanatory estimate).

---

## 4. Chat-to-find — `app/chat/`, `app/retrieval/`

1. **Intent and constraints** (`intent.py`, deterministic, always available). Examples: "at most / up to / 最多 / 以内"
   → inclusive maximum; "under / fewer than" → exclusive; "no X / without X / 不要X" → exclusion;
   "preferably / ideally / 最好" → soft preference; "short / not too long" → soft length preference (never a
   silent "12 episodes"); "shorter than <title>" → soft, relative to the reference; a bare genre → hard include.
   Titles resolve to catalogue items via exact alias, FTS prefix or fuzzy match (`titles.py`); ambiguous matches
   are offered as alternatives. Contradictions (at least 50 and at most 3 episodes) trigger a clarification.
2. **Conversation state** (`conversation.py`): refinements ("at most 6 episodes") update the current state; a new,
   substantive request starts over. State is saved to the account only when the user asks.
3. **Retrieval** (`evidence.py`, `lexical.py`, `dense.py`, `fusion.py`). Hard constraints first; retrieval is
   restricted to eligible titles. Channels: FTS5 BM25 (top 50) and dense cosine over document vectors (top 50),
   fused by reciprocal rank fusion `Σ 1/(60 + rank)`. A title scores as its best document (ties: number of
   retrieved documents, then key), so one title cannot fill the list. If the dense channel fails, lexical only,
   with a warning.
4. **Corpus** (`documents.py`): attribute documents, synopsis chunks (150-350 tokens, 40 overlap), relation facts
   (packed into chunks of whole facts, at most 350 tokens), character lists, scenes (one per document), reviews
   (weak opinion source). Every document has a source type, a source reference and a locator.
5. **Answer** (`grounded_answer.py`). Without an LLM, a deterministic template over the verified facts and evidence
   ("local mode"). With an LLM, the model receives only delimited evidence and verified facts and returns JSON.
   Item keys must be candidates that passed the constraints, cited doc ids must have been in the context, and
   numbers (episodes, scores, episode numbers) are checked against the database and verified scenes. One repair
   round is allowed; otherwise the local answer is shown with a warning. Unsupported specifics become "unknown".
6. **LLM adapters** (`llm.py`): `openai_compatible` (HTTP) or `anthropic` (official SDK, JSON-schema structured
   output, refusal handling). Timeout 30 s, at most 2 calls per request. Cost is `null` (unknown) whenever
   usage or prices are missing.

Scene lookup and screenshots are data-conditional: without verified scene data, an episode-specific question
is answered with "not in the sources" and no episode number.

---

## 5. Viewing plan — `app/planning/`

**Durations** (`duration.py`): remaining episodes = known total - watched; full cost = remaining x per-episode
average; a movie is one indivisible unit (its runtime). Unknown episode counts or durations make a title
unplannable.

**Modes** (`optimizer.py`):

* complete: a unit = a title plus its unmet confirmed prequels (watched first); one bit per unit.
* partial (only when asked): gene = number of next episodes per title; titles needing an unwatched prequel are
  excluded.

**Objective** (weights: relevance 0.45, quality 0.2, diversity 0.2, novelty 0.05, utilisation 0.1):

    relevance / quality = minute-weighted means over planned titles
    diversity           = mean pairwise (1 - Sim_tag)   (unavailable for < 2 titles)
    utilisation         = min(1, planned / budget)       (small weight: no padding with weak titles)
    F = Σ w_t · term_t / Σ w_t over available terms

Budget, title cap, prerequisites and must-include titles are hard. Every chromosome is repaired before evaluation,
and the final plan is re-validated independently (`feasibility.py`). A must-include title that cannot fit is
reported, not forced. GA: population 60, 120 generations, 1.5 s budget, candidate pool 40 (minimum relevance
0.2). For small complete-mode pools (<= 14 units) the exact optimum is also computed (`exhaustive_complete`) and
the GA's gap to it is returned with the plan. User edits create a new revision
whose feasibility is recomputed. **Scheduling** (`scheduling.py`) packs episodes into daily slots: a movie is
never split, per-title breaks go before a title's first episode, and a schedule that does not fit is reported
with what is missing.

---

## 6. Data-conditional extensions

| Extension | Module | Behaviour |
|---|---|---|
| Relation graph | `app/graph/relations.py` | bounded, cycle-safe traversal of confirmed, sourced edges; `prequel` is the only viewing prerequisite; nothing inferred from "Season 2" names; a missing edge does not prove there is no prerequisite |
| Scene search | chat + `/api/v1/scenes/search` | retrieval over scene documents with spoiler levels and verification status |
| Screenshot identification | `app/multimodal/image.py` | frame gallery (CLIP with the `ml` extra; a deterministic colour histogram only for synthetic fixtures); no acceptance threshold until one is calibrated on validation data, so every result is an "unverified candidate"; cover art is never used as a substitute |
