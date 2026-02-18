"""Embedding service using OpenAI API."""

import os
from typing import List
import numpy as np
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class EmbeddingService:
    """OpenAI embedding service with batching."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        batch_size: int = 100,
    ):
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        result = await self.embed([text])
        return result[0] if result else []

    def normalize(self, embeddings: List[List[float]]) -> List[List[float]]:
        """L2 normalize embeddings for cosine similarity."""
        arr = np.array(embeddings)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        normalized = arr / (norms + 1e-8)
        return normalized.tolist()
