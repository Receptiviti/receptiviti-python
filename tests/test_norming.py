import os

import pandas
import pytest

import receptiviti

receptiviti.readin_env()


@pytest.mark.skipif(os.getenv("RECEPTIVITI_KEY") is None, reason="no API key")
class TestStatus:
    def test_listing(self):
        res = receptiviti.norming()
        assert isinstance(res, pandas.DataFrame)
        assert "custom/test" in res["name"].values

        res = receptiviti.norming(name_only=True)
        assert isinstance(res, list)
        assert "custom/test" in res

    def test_single_status(self):
        res = receptiviti.norming("test")
        assert isinstance(res, pandas.Series)
        assert "custom/test" == res["name"]

    def test_updating(self):
        norming_context = "test"
        receptiviti.norming(norming_context, delete=True)
        receptiviti.norming(norming_context, options={"min_word_count": 1})
        with pytest.raises(RuntimeError, match="is not on record"):
            receptiviti.request("a text to score", version="v2", custom_context=norming_context)
        receptiviti.norming(norming_context, "new text to add")
        final_status = receptiviti.norming(norming_context)
        assert isinstance(final_status, pandas.Series)
        assert final_status["status"] == "completed"
        base_request = receptiviti.request("a new text to add", version="v2")
        self_normed_request = receptiviti.request("a new text to add", version="v2", custom_context=norming_context)
        assert base_request is not None and self_normed_request is not None
        assert not base_request.equals(self_normed_request)
