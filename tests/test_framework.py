from pyknn import EmbeddingIndex
from pyknn.index import DictionaryIndexBackend
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

def test_index_multiple(awembedder):

    index = EmbeddingIndex.from_scratch(4, awembedder)

    index.build_index(["First element"])
    index.build_index(["Second element"], clean_space=False)
    
    dump = index.dump_index()
    assert len(dump) == 2

def test_search_empty(awembedder):

    ## NOTE: I have to look into this concurrency issue, to see if this is how python behaves or not.
    index = EmbeddingIndex.from_scratch(2, awembedder, index_backend=DictionaryIndexBackend(data={}))
    result = index.knn_search("some term")
    assert len(result) == 0, "Empty index produces no results"


def test_index_objects(awembedder):
    index = EmbeddingIndex.from_scratch(4, awembedder, index_backend=DictionaryIndexBackend(data={}))

    index.build_index([{'key':"First element tio", 'metadata': "something"}])
    index.build_index([{'key':"Second element tio", 'metadata': "something"}], clean_space=False)

    dump = index.dump_index()
    assert len(dump) == 2, "Index wasn't dumped correctly."

    #try to perform a search
    results = index.knn_search("element", search_words=True)
    assert len(results) > 0, "Search was not successful."


def test_embedder_args_call(embedder):
    index = EmbeddingIndex.from_scratch(2, embedder, index_backend=DictionaryIndexBackend(data={}))

    index.build_index([{'key':"element arg_call", 'metadata': "something"}], embedder_extra_args={'cache': True})

    #cause with this embedder we fall out of dict and the similarity is the edit distance.
    result = index.knn_search("element arg_cal", embedder_extra_args={'cache': False})

    assert len(result) == 1, "We only had indexed one element"

    #we should have called the embedder at least 3 times, one with cache and the rest without.

    assert len(embedder.calls) == 3, "Embedder wasnt called the right times" 

    assert embedder.calls[0][1]['cache'] == True
    assert embedder.calls[1][1]['cache'] == False
    assert embedder.calls[2][1]['cache'] == False
    


