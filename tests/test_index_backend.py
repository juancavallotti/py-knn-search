from pyknn import DictionaryIndexBackend

def test_basic_index_backend():

    value = "Hello world"

    key = "some key"

    subject = DictionaryIndexBackend()

    subject[key] = value

    assert subject[key] == value, "The dictionary should have kept its value."

def test_provided_index_backend():

    initial_data = {"hello world": "Success!"}

    subject = DictionaryIndexBackend(data=initial_data)

    assert subject['hello world'] == initial_data['hello world']