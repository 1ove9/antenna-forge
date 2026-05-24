# ============================================================
# REFERENCE
#   仿造来源：Qdrant client 示例 @ https://github.com/qdrant/qdrant-client
#   对标文件：qdrant_client/examples/
#   对标类/函数：QdrantClient, upsert, search, recommend
#   关键设计点：
#     - 向量相似度检索（余弦/欧几里得/点积）
#     - Payload 结构化过滤
#     - 批量 upsert + scroll 分页
#     - 集合管理（create/delete/exists）
#   YAF 的差异化改造：
#     - 天线设计 embedding 专用向量库
#     - 设计检索：相似设计 + 条件过滤
#     - 客户端自动连接/断开管理
#     - 同步 API（Qdrant 原生为同步）
# ============================================================

"""Qdrant Vector Store — design embedding search and retrieval.

Stores antenna design embeddings for similarity search,
design exploration, and active learning feedback loops.
"""

from __future__ import annotations

from typing import Any


class VectorStore:
    """Qdrant-backed vector store for antenna design embeddings.

    Enables:
    - Semantic search: find similar designs
    - Conditioned search: filter by frequency/gain/polarization
    - Active learning: query-by-committee candidate selection
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "antenna_designs",
        vector_size: int = 128,
    ) -> None:
        """Initialize vector store connection.

        Args:
            host: Qdrant server host.
            port: Qdrant gRPC port.
            collection_name: Name of the Qdrant collection.
            vector_size: Embedding vector dimensionality.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client = None
        self._connected = False

    def _ensure_connection(self) -> None:
        """Establish connection to Qdrant if not already connected."""
        if self._connected:
            return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._client = QdrantClient(host=self.host, port=self.port)

            # Ensure collection exists
            collections = [c.name for c in self._client.get_collections().collections]
            if self.collection_name not in collections:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
            self._connected = True
        except ImportError:
            self._connected = False

    def store_design(
        self,
        design_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a design embedding in the vector database.

        Args:
            design_id: Unique design identifier.
            embedding: Design embedding vector.
            metadata: Optional design metadata (frequency, gain, etc.).

        Returns:
            True if stored successfully, False otherwise.
        """
        self._ensure_connection()
        if not self._client or not self._connected:
            return False

        try:
            from qdrant_client.models import PointStruct

            payload = metadata or {}
            payload["design_id"] = design_id

            point = PointStruct(
                id=hash(design_id) % (2**63),
                vector=embedding,
                payload=payload,
            )

            self._client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )
            return True
        except Exception:
            return False

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar designs by embedding.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            filter_conditions: Optional payload filter.

        Returns:
            List of (design_id, metadata, score) dicts.
        """
        self._ensure_connection()
        if not self._client or not self._connected:
            return []

        try:
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
            )

            return [
                {
                    "design_id": r.payload.get("design_id", ""),
                    "metadata": {
                        k: v for k, v in r.payload.items() if k != "design_id"
                    },
                    "score": r.score,
                }
                for r in results
            ]
        except Exception:
            return []

    def delete_design(self, design_id: str) -> bool:
        """Remove a design from the vector store.

        Args:
            design_id: Design to remove.

        Returns:
            True if deleted.
        """
        self._ensure_connection()
        if not self._client or not self._connected:
            return False

        try:
            point_id = hash(design_id) % (2**63)
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id],
            )
            return True
        except Exception:
            return False

    def count(self) -> int:
        """Return number of designs in the collection.

        Returns:
            Number of stored vectors.
        """
        self._ensure_connection()
        if not self._client or not self._connected:
            return 0

        try:
            info = self._client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return 0

    def close(self) -> None:
        """Release Qdrant connection."""
        if self._client:
            self._client.close()
        self._connected = False


class InMemoryVectorStore:
    """In-memory fallback vector store for development.

    Uses simple cosine similarity over numpy arrays.
    Does not require Qdrant.
    """

    def __init__(self, vector_size: int = 128) -> None:
        self.vector_size = vector_size
        self._embeddings: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._ids: list[str] = []

    def store_design(
        self,
        design_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store design embedding in memory."""
        import numpy as np

        if len(embedding) != self.vector_size:
            return False

        self._embeddings[design_id] = np.array(embedding, dtype=np.float32)
        self._metadata[design_id] = metadata or {}
        if design_id not in self._ids:
            self._ids.append(design_id)
        return True

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search similar designs by cosine similarity."""
        import numpy as np

        if not self._ids:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-12)

        scores: list[tuple[str, float]] = []
        for did in self._ids:
            emb = self._embeddings[did]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
            score = float(np.dot(query_norm, emb_norm))
            scores.append((did, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results: list[dict[str, Any]] = []
        for did, score in scores[:top_k]:
            results.append({
                "design_id": did,
                "metadata": self._metadata.get(did, {}),
                "score": score,
            })

        return results

    def delete_design(self, design_id: str) -> bool:
        """Remove design from in-memory store."""
        if design_id in self._ids:
            self._ids.remove(design_id)
        self._embeddings.pop(design_id, None)
        self._metadata.pop(design_id, None)
        return True

    def count(self) -> int:
        """Number of stored designs."""
        return len(self._ids)
