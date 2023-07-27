from roughpy import LieKey


def test_construct_int():
    key = LieKey(1, width=2, depth=4)
    assert str(key) == "1"


def test_construct_list_pair():
    key = LieKey([1, 2], width=2, depth=4)
    assert str(key) == "[1,2]"


def test_construct_nested_list():
    key = LieKey([1, [1, 2]], width=2, depth=4);
    assert str(key) == "[1,[1,2]]"


def test_construct_two_nested_lists():
    key = LieKey([[1, 2], [1, 3]], width=3, depth=4)

    assert str(key) == "[[1,2],[1,3]]"


def test_construct_lots_of_nesting_nesting():
    key = LieKey([[1, [1, 2]], [[1, 2], [1, [1, 3]]]], width=3, depth=4)

    assert str(key) == "[[1,[1,2]],[[1,2],[1,[1,3]]]]"
