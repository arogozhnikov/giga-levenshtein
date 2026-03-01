import rust_levenshtein


class TestComputeLevenshtein1ToNBytes:
    def test_basic(self):
        result = rust_levenshtein.compute_levenshtein_1_to_n(
            b"kitten", [b"sitting", b"kitten", b"mitten"]
        )
        assert result == [3, 0, 1]

        # Bitty SIMD implementation requires exactly 256 elements (or multiples of 256)
        right = [b"sittin", b"kitten", b"mitten"] + [b"abcabc"] * 253
        result = rust_levenshtein.compute_levenshtein_1_to_n(b"kitten", right)
        assert result[:3] == [2, 0, 1], result
        assert len(result) == 256

    def test_empty_left(self):
        result = rust_levenshtein.compute_levenshtein_1_to_n(
            b"", [b"abc", b"", b"hello"]
        )
        assert result == [3, 0, 5]

    def test_empty_right_list(self):
        result = rust_levenshtein.compute_levenshtein_1_to_n(b"hello", [])
        assert result == []

    def test_single_element(self):
        result = rust_levenshtein.compute_levenshtein_1_to_n(b"abc", [b"abc"])
        assert result == [0]

    def test_unicode_as_bytes(self):
        # "café" in UTF-8 is b"caf\xc3\xa9" (5 bytes)
        # "cafe" in UTF-8 is b"cafe" (4 bytes)
        # "caff" in UTF-8 is b"caff" (4 bytes)
        result = rust_levenshtein.compute_levenshtein_1_to_n(
            "café".encode("utf-8"),
            ["cafe".encode("utf-8"), "café".encode("utf-8"), "caff".encode("utf-8")],
        )
        assert result[1] == 0  # identical
        assert result[0] > 0  # different (é vs e)


class TestComputeLevenshteinMToNBytes:
    def test_basic(self):
        result = rust_levenshtein.compute_levenshtein_m_to_n(
            [b"kitten", b"sitting"],
            [b"kitten", b"sitting"],
        )
        assert result == [[0, 3], [3, 0]]

    def test_empty_left_list(self):
        result = rust_levenshtein.compute_levenshtein_m_to_n([], [b"abc", b"def"])
        assert result == []

    def test_empty_right_list(self):
        result = rust_levenshtein.compute_levenshtein_m_to_n([b"abc", b"def"], [])
        assert result == [[], []]

    def test_single_pair(self):
        result = rust_levenshtein.compute_levenshtein_m_to_n([b"saturday"], [b"sunday"])
        assert result == [[3]]

    def test_symmetric(self):
        left = [b"abc", b"def", b"ghi"]
        right = [b"xyz", b"abc"]
        result = rust_levenshtein.compute_levenshtein_m_to_n(left, right)
        # result[i][j] should equal the distance from left[i] to right[j]
        assert len(result) == 3
        assert all(len(row) == 2 for row in result)
        assert result[0][1] == 0  # "abc" vs "abc"
