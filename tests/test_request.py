import os
from io import StringIO
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory
import pytest
import receptiviti

receptiviti.readin_env()


@pytest.mark.skipif(os.getenv("RECEPTIVITI_KEY") is None, reason="no API key")
class TestRequest:
    def test_single_text(self):
        res = receptiviti.request("text to score", parallel=False, cache=False)
        assert res["summary.word_count"][0] == 3

    def test_multi_text(self):
        res = receptiviti.request(
            ["text to score", float("nan"), "another text", "another text"], cache=False
        )
        assert str(res["summary.word_count"][1]) == "nan"
        assert res["summary.word_count"].iloc[[0, 2, 3]].to_list() == [3, 2, 2]

    def test_framework_selection(self):
        res = receptiviti.request(
            ["text to score", "another text"], frameworks="summary", cache=False
        )
        assert res.shape == (2, 10)
        assert res["word_count"].to_list() == [3, 2]

    def test_framework_prefix(self):
        res = receptiviti.request(
            ["text to score", "another text"],
            frameworks="summary",
            framework_prefix=True,
            cache=False,
        )
        assert res["summary.word_count"].to_list() == [3, 2]

    def test_id_text(self):
        res = receptiviti.request(
            ["text to score", "another text"], ids=["a", "b"], return_text=True, cache=False
        )
        assert res["text"].to_list() == ["text to score", "another text"]
        assert res["id"].to_list() == ["a", "b"]

    def test_verbose(self):
        with redirect_stdout(StringIO()) as out:
            receptiviti.request(
                "text to score", frameworks=["summary", "sallee"], verbose=True, cache=False
            )
        messages = out.getvalue().split("\n")
        expected = ["prep"] * 3 + ["sele", "done", ""]
        assert len(messages) == len(expected) and all(
            line[:4] == expected[i] for i, line in enumerate(messages)
        )

    def test_cache_initialization(self):
        with TemporaryDirectory() as tempdir:
            receptiviti.request("a text to score", cache=tempdir, clear_cache=True)
            assert os.path.isdir(tempdir + "/bin=h")

    @pytest.mark.skipif(not os.path.isfile("../data.txt"), reason="no txt test file present")
    def test_from_directory(self):
        with TemporaryDirectory() as tempdir:
            res_single = receptiviti.request("../data.txt", cache=False)
            nth_text = 0
            with open("../data.txt", encoding="utf-8") as texts:
                for text in texts:
                    nth_text += 1
                    with open(f"{tempdir}/{nth_text}.txt", "w", encoding="utf-8") as txt:
                        txt.write(text)
            res_multi = receptiviti.request(tempdir, cache=False)
        assert res_single["summary.word_count"].sum() == res_multi["summary.word_count"].sum()

    @pytest.mark.skipif(not os.path.isfile("../data.csv"), reason="no csv test file present")
    def test_from_file(self):
        with TemporaryDirectory() as tempdir:
            res_parallel = receptiviti.request(
                "../data.csv",
                text_column="texts",
                id_column="id",
                bundle_size=20,
                cache=tempdir,
            )
            res_serial = receptiviti.request(
                "../data.csv",
                text_column="texts",
                id_column="id",
                bundle_size=20,
                parallel=False,
                cache=tempdir,
            )
        assert res_parallel["summary.word_count"].sum() == res_serial["summary.word_count"].sum()
