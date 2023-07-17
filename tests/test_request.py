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
        res = receptiviti.request("text to score", parallel=False)
        assert res["summary.word_count"][0] == 3

    def test_multi_text(self):
        res = receptiviti.request(["text to score", float("nan"), "another text", "another text"])
        assert str(res["summary.word_count"][1]) == "nan"
        assert res["summary.word_count"].iloc[[0, 2, 3]].to_list() == [3, 2, 2]

    def test_framework_selection(self):
        res = receptiviti.request(["text to score", "another text"], frameworks="summary")
        assert res.shape == (2, 10)
        assert res["word_count"].to_list() == [3, 2]

    def test_framework_prefix(self):
        res = receptiviti.request(
            ["text to score", "another text"], frameworks="summary", framework_prefix=True
        )
        assert res["summary.word_count"].to_list() == [3, 2]

    def test_id_text(self):
        res = receptiviti.request(
            ["text to score", "another text"], ids=["a", "b"], return_text=True
        )
        assert res["text"].to_list() == ["text to score", "another text"]
        assert res["id"].to_list() == ["a", "b"]

    def test_verbose(self):
        with redirect_stdout(StringIO()) as out:
            receptiviti.request("text to score", frameworks=["summary", "sallee"], verbose=True)
        messages = out.getvalue().split("\n")
        expected = ["prep"] * 3 + ["sele", "done", ""]
        assert len(messages) == len(expected) and all(
            line[:4] == expected[i] for i, line in enumerate(messages)
        )

    @pytest.mark.skipif(not os.path.isfile("../data.txt"), reason="no txt test file present")
    def test_from_directory(self):
        res_single = receptiviti.request("../data.txt")
        nth_text = 0
        with TemporaryDirectory() as tempdir:
            with open("../data.txt", encoding="utf-8") as texts:
                for text in texts:
                    nth_text += 1
                    with open(f"{tempdir}/{nth_text}.txt", "w", encoding="utf-8") as txt:
                        txt.write(text)
            res_multi = receptiviti.request(tempdir)
        assert res_single["summary.word_count"].sum() == res_multi["summary.word_count"].sum()

    @pytest.mark.skipif(not os.path.isfile("../data.csv"), reason="no csv test file present")
    def test_from_file(self):
        res_parallel = receptiviti.request(
            "../data.csv", text_column="texts", id_column="id", bundle_size=20
        )
        res_serial = receptiviti.request(
            "../data.csv", text_column="texts", id_column="id", bundle_size=20, parallel=False
        )
        assert res_parallel["summary.word_count"].sum() == res_serial["summary.word_count"].sum()
