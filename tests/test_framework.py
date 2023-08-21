from pyknn import EmbeddingIndex
from pyknn.knn import cosine_distance
import numpy as np

def test_init_embedder(embedder):
    index = EmbeddingIndex.from_scratch(2, embedder)

    keys = ["a"]

    index = index.build_index(keys)

    assert embedder.called


def test_index_size(embedder):

    index = EmbeddingIndex.from_scratch(2, embedder)

    keys = ["a", "b", "c", "a b", "b c", "c d", "b c e "]

    assert len(index.planes) == 2

    index.build_index(keys)

    assert len(index.dump_index()) == len(keys)

def test_index_size_awe(awembedder):

    index = EmbeddingIndex.from_scratch(2, awembedder)

    keys = ["a", "b", "c", "a b", "b c", "c d", "b c e "]

    assert len(index.planes) == 2

    index.build_index(keys)

    assert len(index.dump_index()) == len(keys)

def test_index_clean_awe(awembedder):

    index = EmbeddingIndex.from_scratch(2, awembedder)

    keys = ["a", "b", "c", "a b", "b c", "c d", "b c e "]

    assert len(index.planes) == 2

    index.build_index(keys, embed_all_words=True)

    keys = ["a", "b"]

    index.build_index(keys, embed_all_words=True, clean_space=True)

    assert len(index.dump_index()) == len(keys)

def test_search(embedder):

    index = EmbeddingIndex.from_scratch(2, embedder)

    keys = ["a", "b"]

    index = index.build_index(keys)

    result = index.knn_search("a", k=2, search_words=True, use_synonyms=True)

    assert len(result) == 2

def test_simple_cosine():

    source = np.array([1, 0])
    target = np.array([1, 1]).reshape((1, 2))

    result = cosine_distance(source, target)

    assert all(np.round(result, 3) == [0.707])

def test_vector_cosine():
    
    source = np.array([0, 1]) #a unitary vector
    
    source = source.reshape((1, 2))

    #we set 2 dummy vectors, one on a 90 degree angle and another 45 degree, so we get easy cosines. 0 and 0.707
    testers = np.array([[1, 0], [1, 1], [0, 1]])

    result = cosine_distance(source, testers)

    expected = np.array([0, 0.707, 1]).reshape(3, 1)

    assert all(np.round(result, 3) == expected)


def test_pickle_file(awembedder):

    import tempfile
    import os

    index = EmbeddingIndex.from_scratch(2, awembedder)

    keys = ["a", "b", "c", "a b", "b c", "c d", "b c e "]

    assert len(index.planes) == 2

    index.build_index(keys, embed_all_words=True)

    filename = f"{tempfile.gettempdir()}/test.pickle"

    try: 
        index.to_pickle(filename=filename)
        index = EmbeddingIndex.from_pickle(filename=filename, embeds=awembedder)
    except:
        assert False, "Exception while trying to write pickle file"
    
    assert os.path.isfile(filename), "Pickle file should exist."
    #cleanup
    os.remove(filename)

def test_different_index_backend(awembedder):

    from pyknn import DictionaryIndexBackend

    be = DictionaryIndexBackend()

    index = EmbeddingIndex.from_scratch(2, awembedder, index_backend=be)

    keys = ["a", "b", "c", "a b", "b c", "c d", "b c e "]

    assert len(index.planes) == 2

    index.build_index(keys, embed_all_words=True)

    dump = be.dump()

    assert len(dump) == 1, "We only defined one space."

    assert len(dump['default']) > 0, "We indexed data in the space"


