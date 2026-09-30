"""Exactly five shared input dimensions, independent of users and database state."""
DIMENSIONS = ("genre", "theme", "demographic", "studio", "length")
WEIGHTS = (0.35, 0.30, 0.15, 0.10, 0.10)


def jaccard(left, right):
    a, b = set(left or []), set(right or [])
    return len(a & b) / len(a | b) if a and b else 0.0


def exact(left, right):
    return float(bool(left and right) and str(left).strip().casefold() == str(right).strip().casefold())


def length_similarity(left, right):
    if not left or not right or left <= 0 or right <= 0:
        return 0.5
    return 1.0 / (1.0 + abs(left - right) / 12.0)


def pair_features(left, right):
    return (jaccard(left["genres"], right["genres"]), jaccard(left["themes"], right["themes"]),
            exact(left["demographic"], right["demographic"]), exact(left["studio"], right["studio"]),
            length_similarity(left["episodes"], right["episodes"]))


def feature_matrix(target, candidates):
    import numpy as np
    return np.asarray([pair_features(target, candidate) for candidate in candidates], dtype=np.float64)


def semantic_text(item):
    # Keep title, synopsis and relations OUT: all methods receive the same five dimensions.
    return (f"Genres: {', '.join(item['genres']) or 'unknown'}. "
            f"Themes: {', '.join(item['themes']) or 'unknown'}. "
            f"Demographic: {item['demographic'] or 'unknown'}. "
            f"Studio: {item['studio'] or 'unknown'}. "
            f"Length: {str(item['episodes']) + ' episodes' if item['episodes'] else 'unknown'}.")


def cosine_to_unit(cosine):
    import numpy as np
    return np.clip((np.asarray(cosine) + 1.0) / 2.0, 0.0, 1.0)


def rank_top_k(ids, scores, target_id, k=5):
    import numpy as np
    ids, scores = np.asarray(ids), np.asarray(scores)
    if k < 1 or len(ids) != len(scores) or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid ranking inputs")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite similarity")
    eligible = np.flatnonzero(ids != target_id)
    order = np.lexsort((ids[eligible], -scores[eligible]))
    return eligible[order[:k]]
