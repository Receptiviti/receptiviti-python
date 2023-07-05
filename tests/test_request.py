import os
from io import StringIO
from contextlib import redirect_stdout
import pytest
import receptiviti

receptiviti.readin_env()


@pytest.mark.skipif(os.getenv("RECEPTIVITI_KEY") is None, reason="no API key")
class TestRequest:
    def test_single_text(self):
        res = receptiviti.request("text to score", cores=1)
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

    @pytest.mark.skipif(not os.path.isfile("../data.csv"), reason="no test file present")
    def test_from_file(self):
        res_parallel = receptiviti.request(
            "../data.csv", text_column="texts", id_column="id", bundle_size=20
        )
        res_serial = receptiviti.request(
            "../data.csv", text_column="texts", id_column="id", bundle_size=20, cores=1
        )
        assert res_parallel["summary.word_count"].sum() == res_serial["summary.word_count"].sum()
