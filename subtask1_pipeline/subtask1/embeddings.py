import hashlib
import pickle
from pathlib import Path

import numpy as np

import config

_model = None
_cache = None
_pool_ids = None
_pool_embeddings = None
_pool_embeddings_norm = None
# Language-partitioned index: populated lazily on first call to
# dense_retrieve_partitioned(). Maps lang_str -> list of int indices
# into _pool_ids / _pool_embeddings_norm.
_pool_lang_index: dict | None = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBED_MODEL)
        print(f"[embeddings] Loaded embedding model: {config.EMBED_MODEL}")
    return _model


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if config.EMBED_CACHE_PATH.exists():
        with open(config.EMBED_CACHE_PATH, "rb") as f:
            _cache = pickle.load(f)
    else:
        _cache = {}
    return _cache


def _save_cache():
    with open(config.EMBED_CACHE_PATH, "wb") as f:
        pickle.dump(_cache, f)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _apply_prefix(text: str, is_query: bool) -> str:
    """E5 models expect "query: "/"passage: " prefixes to perform as
    documented -- a no-op for any other EMBED_MODEL."""
    if not config.EMBED_NEEDS_E5_PREFIX:
        return text
    return f"{'query' if is_query else 'passage'}: {text}"


def encode(text: str, is_query: bool = True) -> list:
    """Returns a plain python list of floats (Neo4j-storable), cached.
    is_query=True for claims (the search query side), False for evidence
    passages -- only matters for E5-style asymmetric models."""
    cache = _load_cache()
    prefixed = _apply_prefix(text, is_query)
    h = _hash(prefixed)
    if h in cache:
        return cache[h]
    vec = _get_model().encode(prefixed, convert_to_numpy=True).tolist()
    cache[h] = vec
    # Persist single-item encodes immediately to avoid losing them
    _save_cache()
    return vec


def encode_batch(texts: list, batch_size: int = 64, save_every: int = 500,
                  is_query: bool = True) -> list:
    """Encode many texts efficiently, skipping any already cached, and
    periodically flushing the cache to disk so a crash mid-run doesn't
    lose progress on a 13k-item pool. is_query: see encode()."""
    cache = _load_cache()
    to_encode = []
    to_encode_idx = []
    prefixed_texts = [_apply_prefix(t, is_query) for t in texts]
    results = [None] * len(texts)

    for i, t in enumerate(prefixed_texts):
        h = _hash(t)
        if h in cache:
            results[i] = cache[h]
        else:
            to_encode.append(t)
            to_encode_idx.append(i)

    if to_encode:
        model = _get_model()
        from tqdm import tqdm
        for start in tqdm(range(0, len(to_encode), batch_size), desc="Encoding"):
            batch = to_encode[start:start + batch_size]
            vecs = model.encode(batch, convert_to_numpy=True)
            for j, vec in enumerate(vecs):
                idx = to_encode_idx[start + j]
                h = _hash(prefixed_texts[idx])
                v = vec.tolist()
                cache[h] = v
                results[idx] = v
            if (start // batch_size) % (save_every // batch_size + 1) == 0:
                _save_cache()
        _save_cache()

    return results


def cosine_sim_matrix(query_vec: list, matrix: np.ndarray) -> np.ndarray:
    """query_vec: 1D list/array. matrix: (N, D) array. Returns (N,) scores."""
    q = np.asarray(query_vec, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    m = np.asarray(matrix, dtype=np.float32)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)
    return m @ q


def prepare_pool_embeddings(pool_items: list, batch_size: int = 64, save_every_n_batches: int = 10) -> tuple[list, np.ndarray]:
    """Encode every pool item once, reuse a disk cache, and return pool IDs + embeddings.

    Also builds _pool_lang_index (lang -> [indices into _pool_ids]) so that
    dense_retrieve_partitioned() can restrict cosine search per-language.
    pool_items must include the 'Evidence' text so detect_lang() can label them.
    """
    global _pool_ids, _pool_embeddings, _pool_lang_index
    if _pool_ids is not None and _pool_embeddings is not None:
        return _pool_ids, _pool_embeddings

    cache_path = Path(config.POOL_EMBED_CACHE_PATH)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    ids_to_encode = []
    for item in pool_items:
        item_id = item.get("ID")
        if item_id is None:
            continue
        if item_id not in cache:
            ids_to_encode.append(item)

    if ids_to_encode:
        model = _get_model()
        total_to_encode = len(ids_to_encode)
        for batch_index, start in enumerate(range(0, total_to_encode, batch_size), start=1):
            batch_items = ids_to_encode[start:start + batch_size]
            batch_texts = [_apply_prefix(item.get("Evidence", ""), is_query=False) for item in batch_items]
            vecs = model.encode(batch_texts, convert_to_numpy=True, batch_size=batch_size)
            for item, vec in zip(batch_items, vecs):
                cache[item["ID"]] = vec.astype(np.float32).tolist()

            if batch_index % save_every_n_batches == 0 or start + batch_size >= total_to_encode:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(cache, f)
                encoded_so_far = min(start + batch_size, total_to_encode)
                print(f"[embeddings] Encoded {encoded_so_far}/{total_to_encode} pool items...")
        print(f"[embeddings] Finished encoding and saved cache to {cache_path}")
    else:
        print(f"[embeddings] Reused cached pool embeddings from {cache_path}")

    valid_ids = []
    valid_vecs = []
    # Keep a parallel list of per-item lang tags to build _pool_lang_index below
    pool_id_to_lang = {item["ID"]: item for item in pool_items if item.get("ID") is not None}
    for item in pool_items:
        item_id = item.get("ID")
        if item_id is not None and item_id in cache:
            valid_ids.append(item_id)
            valid_vecs.append(cache[item_id])

    _pool_ids = valid_ids
    _pool_embeddings = np.asarray(valid_vecs, dtype=np.float32) if valid_vecs else np.zeros((0, 0), dtype=np.float32)
    # Pre-normalize pool embeddings once for fast dot-product scoring
    global _pool_embeddings_norm
    if _pool_embeddings.size:
        norms = np.linalg.norm(_pool_embeddings, axis=1, keepdims=True) + 1e-8
        _pool_embeddings_norm = _pool_embeddings / norms
    else:
        _pool_embeddings_norm = np.zeros_like(_pool_embeddings)

    # Build language partition index: lang -> sorted list of integer positions
    # in _pool_ids / _pool_embeddings_norm. This lets dense_retrieve_partitioned
    # slice only the relevant rows without copying the full matrix.
    from subtask1 import data_loader as _dl
    lang_idx: dict = {}
    for pos, pid in enumerate(_pool_ids):
        item = pool_id_to_lang.get(pid)
        if item is None:
            continue
        lang = _dl.detect_lang(item.get("Evidence", ""))
        lang_idx.setdefault(lang, []).append(pos)
    _pool_lang_index = lang_idx
    print(f"[embeddings] Language partition index built: "
          f"{ {k: len(v) for k, v in lang_idx.items()} }")

    return _pool_ids, _pool_embeddings


def dense_retrieve(claim_text: str, top_k: int = 10) -> list:
    """Encode a claim and retrieve the top-k pool IDs by cosine similarity."""
    if _pool_embeddings is None or _pool_ids is None:
        raise RuntimeError("Call prepare_pool_embeddings(pool_items) before dense_retrieve().")
    if not _pool_ids:
        return []
    # Backwards-compatible: encode then delegate to vector-based retrieval
    claim_vec = encode(claim_text)
    return dense_retrieve_vec(claim_vec, top_k=top_k)


def dense_retrieve_vec(claim_vec: list, top_k: int = 10) -> list:
    """Retrieve by pre-computed claim vector (faster when many claims)."""
    global _pool_embeddings_norm, _pool_ids
    if _pool_embeddings_norm is None or _pool_ids is None:
        raise RuntimeError("Call prepare_pool_embeddings(pool_items) before dense_retrieve_vec().")
    if not _pool_ids:
        return []

    q = np.asarray(claim_vec, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    scores = _pool_embeddings_norm @ q
    ranked_idx = np.argsort(scores)[::-1]
    return [_pool_ids[i] for i in ranked_idx[:top_k]]


def dense_retrieve_partitioned(claim_text: str, claim_vec: list, top_k: int = 10) -> list:
    """Language-partitioned dense retrieval (Task 3 diagnostic).

    Restricts cosine similarity search to ONLY the pool items whose
    detect_lang() matches the claim's own detect_lang() result.
    This is the analogue of BM25Retriever's per-language partitioning,
    applied to the NumPy dense path.

    Requires:
      - prepare_pool_embeddings(pool_items) must have been called first
        (builds _pool_ids, _pool_embeddings_norm, and _pool_lang_index).
      - claim_vec: pre-computed embedding for claim_text (plain list/array).

    Design note: we pass claim_text *and* claim_vec separately because the
    caller typically has the vec pre-computed via encode_batch(); we only
    call detect_lang() here (cheap) and skip re-encoding.
    """
    global _pool_embeddings_norm, _pool_ids, _pool_lang_index
    if _pool_embeddings_norm is None or _pool_ids is None:
        raise RuntimeError("Call prepare_pool_embeddings(pool_items) before dense_retrieve_partitioned().")
    if _pool_lang_index is None:
        raise RuntimeError(
            "_pool_lang_index is None -- pool_items must be passed to "
            "prepare_pool_embeddings() so language detection can build the partition map."
        )
    if not _pool_ids:
        return []

    from subtask1 import data_loader as _dl
    lang = _dl.detect_lang(claim_text)
    indices = _pool_lang_index.get(lang, [])
    if not indices:
        return []

    # Slice only the matching-language rows from the pre-normalised matrix
    idx_arr = np.array(indices, dtype=np.int64)
    sub_matrix = _pool_embeddings_norm[idx_arr]  # (n_lang, D)

    q = np.asarray(claim_vec, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    scores = sub_matrix @ q  # (n_lang,)
    top_local = np.argsort(scores)[::-1][:top_k]
    return [_pool_ids[idx_arr[i]] for i in top_local]


if __name__ == "__main__":
    # Gate 1 smoke test -- verify the embedding model actually downloads
    # and runs on your machine. Could not be tested from this sandbox
    # (huggingface.co returned 403 on the egress proxy here).
    vecs = encode_batch(["Kejriwal was arrested", "কেজরিওয়াল গ্রেফতার হয়েছেন"])
    print("Embedding dim:", len(vecs[0]))
    sim = cosine_sim_matrix(vecs[0], np.array([vecs[1]]))
    print("Cross-lingual cosine similarity:", sim)
