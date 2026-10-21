"""Memory bank construction and nearest-neighbour scoring.

The memory bank is the compressed set of "known good" patch embeddings that
a production image is compared against. Two ideas do the heavy lifting:

* **Coreset subsampling** (Park et al., 2021). Storing every patch of every
  good image is wasteful -- the features are highly redundant. A greedy
  coreset keeps a small subset that preserves the coverage of the whole
  distribution, which is what lets the bank stay small enough for a factory
  edge box.

* **k-NN with local reweighting** (Bergmann et al., 2022). A patch is
  anomalous if it is far from the memory bank. But distance alone rewards
  *dense* regions of the bank, so PatchCore down-weights a neighbour whose
  own neighbourhood is far away -- i.e. a neighbour sitting in a sparse
  corner is less trustworthy as evidence of normality.

The NN backend is pluggable. FAISS is used when available (fast, GPU-capable,
memory-mapped indexes); otherwise we fall back to a chunked torch k-NN which
is accurate but uses more RAM. Both are exact -- no approximate search -- so
scores are reproducible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:  # optional
    import faiss  # type: ignore

    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    _HAS_FAISS = False


@dataclass
class NNBackend:
    """Abstract nearest-neighbour index over the memory bank."""

    dim: int

    def fit(self, features: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(distances, indices)`` for each query row."""
        raise NotImplementedError  # pragma: no cover

    def reconstruct(self, indices: np.ndarray) -> np.ndarray:
        """Fetch the feature vectors for the given row indices."""
        raise NotImplementedError  # pragma: no cover

    @property
    def size(self) -> int:
        raise NotImplementedError  # pragma: no cover

    def save(self, path: Path) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "NNBackend":  # pragma: no cover - interface
        raise NotImplementedError


class FaissNN(NNBackend):
    """FAISS-backed index (preferred in production)."""

    def __init__(self, dim: int, use_gpu: bool = False) -> None:
        if not _HAS_FAISS:
            raise RuntimeError("faiss is not installed")
        super().__init__(dim)
        self.use_gpu = use_gpu
        self.index: faiss.Index | None = None
        self._store: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> None:
        features = np.ascontiguousarray(features.astype(np.float32))
        self._store = features
        index = faiss.IndexFlatL2(self.dim)  # exact search
        if self.use_gpu and hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(features)
        self.index = index

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries.astype(np.float32))
        return self.index.search(queries, k)

    def reconstruct(self, indices: np.ndarray) -> np.ndarray:
        return self._store[indices]

    @property
    def size(self) -> int:
        return len(self._store)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "features.npy", self._store)
        faiss.write_index(self.index, str(path / "index.faiss"))

    @classmethod
    def load(cls, path: Path) -> "FaissNN":
        store = np.load(path / "features.npy")
        obj = cls(store.shape[1])
        obj._store = store
        obj.index = faiss.read_index(str(path / "index.faiss"))
        return obj


class TorchNN(NNBackend):
    """Pure-torch exact k-NN, chunked to bound peak memory.

    Uses the ``||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b`` expansion so the cost
    is a matmul rather than an O(N*M*D) broadcast. We still compute the true
    distance (not the squared one) so scores are comparable with FAISS.
    """

    def __init__(self, dim: int, chunk_size: int = 4096) -> None:
        super().__init__(dim)
        self.chunk_size = chunk_size
        self._store: np.ndarray | None = None
        self._store_t: torch.Tensor | None = None

    def fit(self, features: np.ndarray) -> None:
        self._store = np.ascontiguousarray(features.astype(np.float32))
        self._store_t = torch.from_numpy(self._store)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._store_t is not None, "index not fitted"
        q = torch.from_numpy(np.ascontiguousarray(queries.astype(np.float32)))
        k = min(k, self._store_t.shape[0])
        all_d: list[torch.Tensor] = []
        all_i: list[torch.Tensor] = []
        store_sq = (self._store_t * self._store_t).sum(dim=1)
        for start in range(0, q.shape[0], self.chunk_size):
            chunk = q[start : start + self.chunk_size]
            # ||a-b||^2 expansion
            d2 = (
                (chunk * chunk).sum(dim=1, keepdim=True)
                + store_sq.unsqueeze(0)
                - 2.0 * (chunk @ self._store_t.T)
            )
            d2 = d2.clamp_min_(0)
            dist, idx = torch.topk(d2, k, dim=1, largest=False)
            all_d.append(dist)
            all_i.append(idx)
        distances = torch.sqrt(torch.cat(all_d, dim=0)).numpy()
        indices = torch.cat(all_i, dim=0).numpy()
        return distances.astype(np.float32), indices.astype(np.int64)

    def reconstruct(self, indices: np.ndarray) -> np.ndarray:
        return self._store[indices]

    @property
    def size(self) -> int:
        return 0 if self._store is None else len(self._store)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "features.npy", self._store)

    @classmethod
    def load(cls, path: Path) -> "TorchNN":
        store = np.load(path / "features.npy")
        obj = cls(store.shape[1])
        obj.fit(store)
        return obj


def build_nn_backend(dim: int, prefer: str = "auto") -> NNBackend:
    """Create the best available NN backend.

    Args:
        dim: dimensionality of the feature vectors.
        prefer: ``"auto"``, ``"faiss"`` or ``"torch"``. ``"auto"`` uses FAISS
            when it is installed, since it is materially faster and supports
            larger banks.
    """
    if prefer == "faiss" or (prefer == "auto" and _HAS_FAISS):
        try:
            return FaissNN(dim)
        except RuntimeError:
            if prefer == "faiss":
                raise
    return TorchNN(dim)


def greedy_coreset(
    features: np.ndarray,
    percentage: float = 0.1,
    projection_dim: int = 128,
    device: str = "cpu",
    random_state: int = 0,
    max_candidates: int = 100_000,
) -> np.ndarray:
    """Greedy coreset selection (Park et al., 2021).

    Iteratively picks the point that is furthest from everything selected so
    far. The result is a small subset that covers the original distribution
    about as well as the full set for nearest-neighbour queries.

    The selection runs on a random-projection of the features, which is what
    the original method does: the projection preserves distances well enough
    for selection while making the selection much cheaper.

    Two implementation notes that matter for speed:

    * We maintain a running "distance to the nearest selected point" vector
      and update it with a single matrix-vector product per selection, rather
      than recomputing a full distance matrix. The naive O(N^2) memory
      version cannot be used here: a few hundred training images produce
      hundreds of thousands of patches, and the square of that is tens of
      gigabytes.
    * We cap the candidate pool. Past a certain size the extra candidates
      add little coverage and a lot of compute.

    Args:
        features: ``(N, D)`` array of patch embeddings.
        percentage: fraction of the bank to retain. 1.0 keeps everything.
        projection_dim: dimensionality of the random projection.
        device: torch device for the selection.
        random_state: seed for the projection, so runs are reproducible.
        max_candidates: cap on the pool the greedy selection runs over.

    Returns:
        Indices into ``features`` of the selected points.
    """
    n_total = len(features)
    if percentage >= 1.0:
        return np.arange(n_total)

    feats = torch.from_numpy(np.ascontiguousarray(features.astype(np.float32)))

    # Cap the candidate pool, keeping a random subset of it.
    candidate_idx = np.arange(n_total)
    if n_total > max_candidates:
        generator = np.random.default_rng(random_state)
        candidate_idx = np.sort(
            generator.choice(n_total, size=max_candidates, replace=False)
        )
    feats = feats[candidate_idx]

    # Random projection to a lower dimension.
    if feats.shape[1] > projection_dim:
        generator = torch.Generator(device="cpu").manual_seed(random_state)
        proj = torch.randn(
            feats.shape[1], projection_dim, generator=generator, dtype=torch.float32
        )
        feats = feats @ proj
    feats = feats.to(device)

    n = feats.shape[0]
    num_samples = max(1, int(n * percentage))

    # Distance from each candidate to the nearest already-selected point.
    min_dist = torch.full((n,), float("inf"), device=device)
    selected: list[int] = []
    selected_mask = torch.zeros(n, dtype=torch.bool, device=device)

    for _ in range(num_samples):
        idx = int(torch.argmax(min_dist).item())
        selected.append(idx)
        selected_mask[idx] = True

        # Update the running distance with a single matrix-vector product.
        anchor = feats[idx]
        dist = torch.sqrt(
            (feats * feats).sum(dim=1) + (anchor * anchor).sum()
            - 2.0 * (feats @ anchor)
        ).clamp_min_(0)
        min_dist = torch.minimum(min_dist, dist)
        min_dist[selected_mask] = 0.0

    return candidate_idx[np.array(selected, dtype=np.int64)]


@dataclass
class MemoryBank:
    """A fitted memory bank: compressed good-image features + an NN index."""

    features: np.ndarray
    index: NNBackend
    coreset_indices: np.ndarray
    n_input_features: int
    backbone: str
    config_hash: str
    created_at: str
    # Which source image each bank row came from. Used to exclude an image's
    # own patches when calibrating, so the threshold reflects a genuinely
    # unseen part.
    source_labels: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self.features)

    @property
    def compression_ratio(self) -> float:
        """Ratio of retained to original feature count."""
        return self.size / max(1, self.n_input_features)

    @classmethod
    def build(
        cls,
        features: np.ndarray,
        percentage: float = 0.1,
        backbone: str = "wide_resnet50_2",
        config_hash: str = "",
        nn_backend: str = "auto",
        random_state: int = 0,
        source_labels: np.ndarray | None = None,
    ) -> "MemoryBank":
        """Subsample features into a coreset and index them.

        Args:
            features: ``(N, D)`` array of patch embeddings.
            percentage: fraction of the bank to retain.
            backbone: which backbone produced the features.
            config_hash: hash of the embedding configuration.
            nn_backend: which NN backend to use.
            random_state: seed for the coreset selection.
            source_labels: optional ``(N,)`` array giving the source image
                index of each feature row. Carried through the coreset so
                that leave-one-out calibration can exclude an image's own
                patches.
        """
        from datetime import datetime, timezone

        indices = greedy_coreset(
            features,
            percentage=percentage,
            random_state=random_state,
        )
        reduced = features[indices]
        index = build_nn_backend(reduced.shape[1], prefer=nn_backend)
        index.fit(reduced)
        return cls(
            features=reduced,
            index=index,
            coreset_indices=indices,
            n_input_features=len(features),
            backbone=backbone,
            config_hash=config_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_labels=(
                np.asarray(source_labels)[indices]
                if source_labels is not None
                else None
            ),
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "coreset_indices.npy", self.coreset_indices)
        if self.source_labels is not None:
            np.save(path / "source_labels.npy", self.source_labels)
        self.index.save(path)

    @classmethod
    def load(cls, path: Path, meta: dict | None = None) -> "MemoryBank":
        index_path = path / "index.faiss"
        if index_path.exists():
            index = FaissNN.load(path)
        else:
            index = TorchNN.load(path)
        meta = meta or {}
        labels_path = path / "source_labels.npy"
        source_labels = np.load(labels_path) if labels_path.exists() else None
        return cls(
            features=index.reconstruct(np.arange(index.size)),
            index=index,
            coreset_indices=np.load(path / "coreset_indices.npy"),
            n_input_features=meta.get("n_input_features", index.size),
            backbone=meta.get("backbone", "unknown"),
            config_hash=meta.get("config_hash", ""),
            created_at=meta.get("created_at", ""),
            source_labels=source_labels,
        )


def knn_anomaly_scores(
    query_features: np.ndarray,
    index: NNBackend,
    n_nearest_neighbours: int = 1,
    reweight: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score query patches against the memory bank.

    Args:
        query_features: ``(Q, D)`` embeddings of the image under test.
        index: fitted NN index over good-image features.
        n_nearest_neighbours: how many neighbours to consider. The image
            score uses the paper's weighting, which needs k > 1.
        reweight: apply the paper's local-neighbourhood reweighting.

    Returns:
        ``(patch_scores, image_score, distances)`` where patch_scores is
        ``(Q,)`` and image_score is a scalar.
    """
    distances, indices = index.search(query_features, n_nearest_neighbours)

    # Patch score: distance to the closest good patch.
    patch_scores = distances[:, 0].astype(np.float64)

    if n_nearest_neighbours == 1 or not reweight:
        return patch_scores, float(patch_scores.max()), distances

    # Paper's weighted image score:
    #   1. take the most anomalous patch m_test
    #   2. find its nearest neighbour m* in the bank
    #   3. look at m*'s own neighbourhood in the bank
    #   4. weight the score by how typical m* is for its own neighbourhood
    max_idx = int(np.argmax(patch_scores))
    nn_index = indices[max_idx, 0]
    score = patch_scores[max_idx]

    nn_sample = index.reconstruct(np.array([nn_index]))
    support_distances, support_idx = index.search(nn_sample, min(n_nearest_neighbours, index.size))
    support_samples = index.reconstruct(support_idx[0])

    # Distance from the anomalous patch to each of its neighbour's neighbours.
    dist_to_support = np.linalg.norm(
        query_features[max_idx].reshape(1, -1) - support_samples, axis=1
    )
    # Softmax weights, inverted: a neighbour in a dense region is typical
    # (high weight), one in a sparse region is not.
    shifted = dist_to_support - dist_to_support.min()
    weights = 1.0 - np.exp(-shifted) / (np.exp(-shifted).sum() + 1e-12)
    weight = float(weights[0]) if len(weights) else 1.0

    return patch_scores, float(weight * score), distances


def timed(fn, *args, **kwargs):
    """Run ``fn`` and return ``(result, elapsed_seconds)``."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def knn_scores_excluding(
    query_features: np.ndarray,
    bank: MemoryBank,
    exclude_labels: np.ndarray,
    query_label: int,
) -> np.ndarray:
    """Score queries against the bank, ignoring rows from one source image.

    This is the leave-one-out trick. When calibrating, we want to know what
    a *new* good part scores -- not what a part the bank already contains
    scores. Scoring a training image against a bank that includes its own
    patches gives a distance of ~0, which makes the calibrated threshold
    far too low and causes the system to reject good parts.

    Rather than rebuilding the bank for each held-out image (which is
    expensive), we mask out the excluded rows during the search. We do this
    by computing the full k-NN and then re-checking any neighbour that
    belongs to the excluded image against the next-nearest allowed
    neighbour.

    Args:
        query_features: ``(Q, D)`` query embeddings.
        bank: the memory bank to search.
        exclude_labels: per-row source labels for the bank.
        query_label: the source label to exclude.

    Returns:
        ``(Q,)`` array of distances to the nearest allowed bank row.
    """
    if bank.source_labels is None:
        distances, _ = bank.index.search(query_features, 1)
        return distances[:, 0].astype(np.float64)

    labels = bank.source_labels

    # Ask for several extra neighbours so we have a fallback when the nearest
    # ones are excluded. A patch's closest neighbours are often from the
    # *same* image (similar positions across parts), so a small k would
    # frequently be entirely excluded.
    k = min(32, bank.size)
    distances, indices = bank.index.search(query_features, k)
    excluded_mask = labels[indices] == query_label  # (Q, k)

    # For each query, the first allowed neighbour within the k we fetched.
    has_allowed = ~excluded_mask.any(axis=1)
    out = np.full(len(query_features), np.inf, dtype=np.float64)
    if has_allowed.any():
        rows = np.where(has_allowed)[0]
        # Position of the first non-excluded neighbour in each row.
        first_allowed = np.argmax(~excluded_mask[rows], axis=1)
        out[rows] = distances[rows, first_allowed]

    # Queries whose k nearest neighbours were all excluded need a wider
    # search to find an allowed row.
    unresolved = np.where(~has_allowed)[0]
    if len(unresolved):
        all_dist, all_idx = bank.index.search(query_features[unresolved], bank.size)
        allowed_mask = labels[all_idx] != query_label
        any_allowed = allowed_mask.any(axis=1)
        for row_pos in np.where(any_allowed)[0]:
            first = int(np.argmax(allowed_mask[row_pos]))
            out[unresolved[row_pos]] = all_dist[row_pos, first]

    return out
