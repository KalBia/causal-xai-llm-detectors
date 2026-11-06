from src.interventions.deletion import Deletion


def test_deletion():
    assert Deletion().apply(["a", "b", "c"], [1]) == "a c"
