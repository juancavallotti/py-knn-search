from pyknn import EmbeddingIndex

def test_init_embedder(embedder):
    index = EmbeddingIndex.from_scratch(2, embedder)

    keys = ["a"]

    index = index.build_index(keys)

    assert embedder.called

def test_search(embedder):

    index = EmbeddingIndex.from_scratch(2, embedder)

    keys = ["a", "b"]

    index = index.build_index(keys)

    result = index.knn_search("a", k=2, search_words=True, use_synonyms=True)

    assert len(result) == 2