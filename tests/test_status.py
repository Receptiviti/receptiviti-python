import os
from io import StringIO
from contextlib import redirect_stdout
import pytest
import receptiviti

receptiviti.readin_env()


class TestStatus:
    def test_unrecognized_key(self):
        res = receptiviti.status(key="123", secret="123")
        assert res.json()["code"] == 1411

    def test_unrecognized_key_message(self):
        with redirect_stdout(StringIO()) as out:
            receptiviti.status(key="123", secret="123")
        message = out.getvalue()
        assert message.split("\n")[0] == "Status: ERROR"

    def test_invalid_url(self):
        with pytest.raises(TypeError):
            receptiviti.status("localhost")

    @pytest.mark.skipif(condition=os.getenv("RECEPTIVITI_KEY") is None, reason="no API key")
    def test_key_works(self):
        res = receptiviti.status()
        assert res.status_code == 200
