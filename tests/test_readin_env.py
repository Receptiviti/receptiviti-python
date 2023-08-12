import os
import tempfile

import receptiviti


class TestEnv:
    def test_explicit_works(self):
        receptiviti.readin_env()
        assert os.getenv("RECEPTIVITI_TEST") is None
        with tempfile.NamedTemporaryFile("w", delete=False) as file:
            file.write("RECEPTIVITI_TEST='123'\n")
            path = file.name
        receptiviti.readin_env(path)
        os.unlink(path)
        assert os.getenv("RECEPTIVITI_TEST") == "123"
