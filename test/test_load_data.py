from src.load import load_data

def test_data_valid():
    data = load_data()
    assert data["cases"].min() >= 0 
    assert data["week"].is_monotonic_increasing 
    assert not data["week"].duplicated().any() 