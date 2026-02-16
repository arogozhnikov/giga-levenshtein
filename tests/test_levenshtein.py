import rust_levenshtein


class TestComputeLevenshtein1ToN:
    def test_basic(self):
        result = rust_levenshtein.compute_levenshtein_1_to_n(
            "kitten", ["sitting", "kitten", "mitten"]
        )
        assert result == [3, 0, 1]

    def test_empty_left(self):
        result = rust_levenshtein.compute_levenshtein_1_to_n("", ["abc", "", "hello"])
        assert result == [3, 0, 5]

    def test_empty_right_list(self):
        result = rust_levenshtein.compute_levenshtein_1_to_n("hello", [])
        assert result == []

    def test_single_element(self):
        result = rust_levenshtein.compute_levenshtein_1_to_n("abc", ["abc"])
        assert result == [0]

    def test_unicode(self):
        result = rust_levenshtein.compute_levenshtein_1_to_n(
            "café", ["cafe", "café", "caff"]
        )
        assert result[1] == 0  # identical
        assert result[0] > 0  # different (é vs e)


class TestComputeLevenshteinMToN:
    def test_basic(self):
        result = rust_levenshtein.compute_levenshtein_m_to_n(
            ["kitten", "sitting"],
            ["kitten", "sitting"],
        )
        assert result == [[0, 3], [3, 0]]

    def test_empty_left_list(self):
        result = rust_levenshtein.compute_levenshtein_m_to_n([], ["abc", "def"])
        assert result == []

    def test_empty_right_list(self):
        result = rust_levenshtein.compute_levenshtein_m_to_n(["abc", "def"], [])
        assert result == [[], []]

    def test_single_pair(self):
        result = rust_levenshtein.compute_levenshtein_m_to_n(["saturday"], ["sunday"])
        assert result == [[3]]

    def test_symmetric(self):
        left = ["abc", "def", "ghi"]
        right = ["xyz", "abc"]
        result = rust_levenshtein.compute_levenshtein_m_to_n(left, right)
        # result[i][j] should equal the distance from left[i] to right[j]
        assert len(result) == 3
        assert all(len(row) == 2 for row in result)
        assert result[0][1] == 0  # "abc" vs "abc"
