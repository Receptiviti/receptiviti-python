import os
from contextlib import redirect_stdout
from io import StringIO
from tempfile import TemporaryDirectory

import pandas
import pytest

import receptiviti

receptiviti.readin_env()


@pytest.mark.skipif(os.getenv("RECEPTIVITI_KEY") is None, reason="no API key")
class TestRequest:
    def test_unreachable(self):
        url = "http://localhost:0/not_served"
        assert receptiviti.status(url) is None
        with pytest.raises(RuntimeError, match="URL is not reachable"):
            receptiviti.request("a text", url=url)

    def test_single_text(self):
        res = receptiviti.request("text to score")
        assert res["summary.word_count"][0] == 3

    def test_invalid_text(self):
        with pytest.raises(RuntimeError, match="one of your texts is over the bundle size limit"):
            receptiviti.request(" " * int(1e7))
        with pytest.raises(RuntimeError, match="no valid texts to process"):
            receptiviti.request("")

    def test_multi_text(self):
        res = receptiviti.request(["text to score", float("nan"), "another text", "another text"])
        assert str(res["summary.word_count"][1]) == "nan"
        assert res["summary.word_count"].iloc[pandas.Index([0, 2, 3])].to_list() == [3, 2, 2]

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
        expected = ["prep"] * 3 + ["requ", "done", "prep", "sele", "done", ""]
        assert len(messages) == len(expected) and all(
            line[:4] == expected[i] for i, line in enumerate(messages)
        )

    def test_cache_initialization(self):
        with TemporaryDirectory() as tempdir:
            receptiviti.request("a text to score", cache=tempdir, clear_cache=True)
            assert os.path.isdir(tempdir + "/bin=h")

    def test_id_assignment(self):
        text = ["text to score", "another text"]
        with TemporaryDirectory() as tempdir:
            txt_file = f"{tempdir}/text.txt"
            with open(txt_file, "w", encoding="utf-8") as txt:
                txt.write("\n".join(text))
            csv_file = f"{tempdir}/text.csv"
            pandas.DataFrame({"text": text}).to_csv(csv_file)
            assert receptiviti.request(txt_file)["id"].to_list() == [
                txt_file + "1",
                txt_file + "2",
            ]
            assert receptiviti.request(csv_file, text_column="text")["id"].to_list() == [
                csv_file + "1",
                csv_file + "2",
            ]

    @pytest.mark.skipif(not os.path.isfile("../data.txt"), reason="no txt test file present")
    def test_from_directory(self):
        with TemporaryDirectory() as tempdir:
            cache = tempdir + "/cache"
            res_single = receptiviti.request("../data.txt", cache=cache)
            nth_text = 0
            txt_files = []
            csv_files = []
            with open("../data.txt", encoding="utf-8") as texts:
                for text in texts:
                    nth_text += 1
                    txt_file = f"{tempdir}/{nth_text}.txt"
                    txt_files.append(txt_file)
                    with open(txt_file, "w", encoding="utf-16") as txt:
                        txt.write(text)
                    csv_file = f"{tempdir}/{nth_text}.csv"
                    csv_files.append(csv_file)
                    pandas.DataFrame({"text": [text]}).to_csv(csv_file, encoding="cp1252")
            res_misencode = receptiviti.request(
                tempdir, encoding="utf-8", return_text=True, cache=cache
            )
            res_multi = receptiviti.request(tempdir, return_text=True, cache=cache)
            res_multi_txt = receptiviti.request(txt_files, cache=cache)
            res_multi_csv = receptiviti.request(csv_files, text_column="text", cache=cache)
            res_multi_txt_collapse = receptiviti.request(
                txt_files, collapse_lines=True, cache=cache
            )
            res_multi_csv_collapse = receptiviti.request(
                csv_files, text_column="text", collapse_lines=True, cache=cache
            )
        assert not all((a == b for a, b in zip(res_multi["text"], res_misencode["text"])))
        assert res_single["summary.word_count"].sum() == res_multi["summary.word_count"].sum()
        assert res_multi["summary.word_count"].sum() == res_multi_txt["summary.word_count"].sum()
        assert res_multi["summary.word_count"].sum() == res_multi_csv["summary.word_count"].sum()
        assert (
            res_multi["summary.word_count"].sum()
            == res_multi_txt_collapse["summary.word_count"].sum()
        )
        assert (
            res_multi["summary.word_count"].sum()
            == res_multi_csv_collapse["summary.word_count"].sum()
        )

    @pytest.mark.skipif(not os.path.isfile("../data.csv"), reason="no csv test file present")
    def test_from_file(self):
        with TemporaryDirectory() as tempdir:
            res_parallel = receptiviti.request(
                "../data.csv",
                text_column="texts",
                id_column="id",
                bundle_size=20,
                cores=2,
                cache=tempdir,
                in_memory=False,
            )
            res_serial = receptiviti.request(
                "../data.csv",
                text_column="texts",
                id_column="id",
                bundle_size=20,
                cache=tempdir,
            )
        assert res_parallel["summary.word_count"].sum() == res_serial["summary.word_count"].sum()

    @pytest.mark.skipif(
        receptiviti.status(os.getenv("RECEPTIVITI_URL_TEST", "")) is None,
        reason="test API is not reachable",
    )
    def test_endpoint_version(self):
        with TemporaryDirectory() as tempdir:
            receptiviti.request(
                "text to process",
                url=os.getenv("RECEPTIVITI_URL_TEST"),
                key=os.getenv("RECEPTIVITI_KEY_TEST"),
                secret=os.getenv("RECEPTIVITI_SECRET_TEST"),
                cache=tempdir,
            )
            with redirect_stdout(StringIO()) as out:
                receptiviti.request(
                    "text to process",
                    url=os.getenv("RECEPTIVITI_URL_TEST") + "v2/taxonomies",
                    key=os.getenv("RECEPTIVITI_KEY_TEST"),
                    secret=os.getenv("RECEPTIVITI_SECRET_TEST"),
                    cache=tempdir,
                    verbose=True,
                )
        messages = out.getvalue().split("\n")
        expected = ["prep"] * 3 + ["requ", "done", "clea", "addi", "prep", "done", ""]
        assert len(messages) == len(expected) and all(
            line[:4] == expected[i] for i, line in enumerate(messages)
        )
