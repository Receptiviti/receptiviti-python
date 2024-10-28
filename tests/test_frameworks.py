import os

import pytest

import receptiviti

receptiviti.readin_env()


@pytest.mark.skipif(condition=os.getenv("RECEPTIVITI_KEY") is None, reason="no API key")
def test_listing():
    res = receptiviti.frameworks()
    assert res and isinstance(res, list)
