import rust_levenshtein


class TestComputeLevenshtein1ToNBytes:
    def test_basic(self):
        result = rust_levenshtein.compute_levenshtein_1_to_n_bytes(
            b"kitten", [b"sitting", b"kitten", b"mitten"]
        )
        assert result == [3, 0, 1]

    def test_empty_left(self):
        result = rust_levenshtein.compute_levenshtein_1_to_n_bytes(
            b"", [b"abc", b"", b"hello"]
        )
        assert result == [3, 0, 5]

    def test_empty_right_list(self):
        result = rust_levenshtein.compute_levenshtein_1_to_n_bytes(b"hello", [])
        assert result == []

    def test_single_element(self):
        result = rust_levenshtein.compute_levenshtein_1_to_n_bytes(b"abc", [b"abc"])
        assert result == [0]

    def test_unicode_as_bytes(self):
        # "café" in UTF-8 is b"caf\xc3\xa9" (5 bytes)
        # "cafe" in UTF-8 is b"cafe" (4 bytes)
        # "caff" in UTF-8 is b"caff" (4 bytes)
        result = rust_levenshtein.compute_levenshtein_1_to_n_bytes(
            "café".encode("utf-8"),
            ["cafe".encode("utf-8"), "café".encode("utf-8"), "caff".encode("utf-8")],
        )
        assert result[1] == 0  # identical
        assert result[0] > 0  # different (é vs e)


class TestComputeLevenshtein1ToNBytesSIMD:
    def test_basic(self):
        # Current SIMD implementation requires exactly 32 elements (or multiples of 32)
        right = [b"sittin", b"kitten", b"mitten"] + [b"abcabc"] * 29
        result = rust_levenshtein.compute_levenshtein_1_to_n_bytes_simd(
            b"kitten", right
        )
        assert result[:3] == [2, 0, 1], result
        assert len(result) == 32

    def test_unicode_as_bytes(self):
        right = [
            "cafe",
            "café",
            "caff",
        ] + [""] * 29

        def pad_encode(x: str):
            y = x.encode("utf-8")
            return y + b" " * (19 - len(y))

        right = [pad_encode(x) for x in right]

        result = rust_levenshtein.compute_levenshtein_1_to_n_bytes_simd(
            pad_encode("café"), right
        )
        assert result[1] == 0  # identical
        assert len(result) == 32


class TestComputeLevenshtein1ToNBytesBittySIMD:
    def test_basic(self):
        # Bitty SIMD implementation requires exactly 256 elements (or multiples of 256)
        right = [b"sittin", b"kitten", b"mitten"] + [b"abcabc"] * 253
        result = rust_levenshtein.compute_levenshtein_1_to_n_bytes_bitty_simd(
            b"kitten", right
        )
        assert result[:3] == [2, 0, 1], result
        assert len(result) == 256


class TestComputeLevenshteinMToNBytes:
    def test_basic(self):
        result = rust_levenshtein.compute_levenshtein_m_to_n_bytes(
            [b"kitten", b"sitting"],
            [b"kitten", b"sitting"],
        )
        assert result == [[0, 3], [3, 0]]

    def test_empty_left_list(self):
        result = rust_levenshtein.compute_levenshtein_m_to_n_bytes([], [b"abc", b"def"])
        assert result == []

    def test_empty_right_list(self):
        result = rust_levenshtein.compute_levenshtein_m_to_n_bytes([b"abc", b"def"], [])
        assert result == [[], []]

    def test_single_pair(self):
        result = rust_levenshtein.compute_levenshtein_m_to_n_bytes(
            [b"saturday"], [b"sunday"]
        )
        assert result == [[3]]

    def test_symmetric(self):
        left = [b"abc", b"def", b"ghi"]
        right = [b"xyz", b"abc"]
        result = rust_levenshtein.compute_levenshtein_m_to_n_bytes(left, right)
        # result[i][j] should equal the distance from left[i] to right[j]
        assert len(result) == 3
        assert all(len(row) == 2 for row in result)
        assert result[0][1] == 0  # "abc" vs "abc"
