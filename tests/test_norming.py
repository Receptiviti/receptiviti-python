import os

import pytest

import receptiviti

receptiviti.readin_env()


@pytest.mark.skipif(os.getenv("RECEPTIVITI_KEY") is None, reason="no API key")
class TestStatus:
    def test_listing(self):
        res = receptiviti.norming()
        assert "test" in res["name"].values

    def test_single_status(self):
        res = receptiviti.norming("test")
        assert "test" == res["name"]

    def test_updating(self):
        norming_context = "short_text8"
        with pytest.warns(UserWarning, match="option invalid_option was not set"):
            initial_status = receptiviti.norming(norming_context, options={"word_count_filter": 1, "invalid_option": 1})
        if initial_status["status"] != "completed":
            with pytest.raises(RuntimeError, match="has not been completed"):
                receptiviti.request("a text to score", version="v2", api_args={"custom_context": norming_context})
            receptiviti.norming(norming_context, "new text to add")
        final_status = receptiviti.norming(norming_context)
        assert final_status["status"] == "completed"
        base_request = receptiviti.request("a new text to add", version="v2")
        self_normed_request = receptiviti.request(
            "a new text to add", version="v2", api_args={"custom_context": norming_context}
        )
        assert not base_request.equals(self_normed_request)
