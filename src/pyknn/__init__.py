from .knn import EmbeddingIndex
from .embeds import Embedder, GloVeEmbeddings
from .index import IndexBackend, DictionaryIndexBackend
"""
PY KNN Search - A simple implementation of KNN.

This library uses the `EmbeddingIndex` with an implementation of `Embedder` to index and search for strings.

NOTE: As of now, this library is implemented using NumPy and will not perform computations on GPU.
"""