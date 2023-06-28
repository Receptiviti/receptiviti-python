import pytest
import os
import receptiviti

receptiviti.readin_env()


@pytest.mark.skipif(os.getenv("RECEPTIVITI_KEY") is None, reason="no API key")
class TestRequest:
    def test_single_text(self):
        res = receptiviti.request("text to score", cores=1)
        assert res["summary.word_count"][0] == 3

    def test_multi_text(self):
        res = receptiviti.request(["text to score", float("nan"), "another text", "another text"])
        assert res["summary.word_count"].to_list() == [3, 2]

    def test_framework_selection(self):
        res = receptiviti.request(["text to score", "another text"], frameworks="summary")
        assert res.shape == (2, 10)
        assert res["word_count"].to_list() == [3, 2]

    def test_framework_prefix_works(self):
        res = receptiviti.request(
            ["text to score", "another text"], frameworks="summary", framework_prefix=True
        )
        assert res["summary.word_count"].to_list() == [3, 2]

    @pytest.mark.skipif(not os.path.isfile("../data.csv"), reason="no test file present")
    def test_from_file(self):
        res = receptiviti.request("../data.csv", text_column="texts", bundle_size=100)
        assert len(res) == 489
