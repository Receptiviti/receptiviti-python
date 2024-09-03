"""Make requests to the API."""

import hashlib
import json
import math
import os
import pickle
import re
import shutil
import sys
import urllib.parse
from glob import glob
from multiprocessing import Process, Queue, current_process
from tempfile import TemporaryDirectory, gettempdir
from time import perf_counter, sleep, time
from typing import List, Union

import numpy
import pandas
import pyarrow
import pyarrow.compute
import pyarrow.dataset
import requests
from chardet.universaldetector import UniversalDetector
from tqdm import tqdm

from receptiviti.readin_env import readin_env
from receptiviti.status import status

CACHE = gettempdir() + "/receptiviti_cache/"
REQUEST_CACHE = gettempdir() + "/receptiviti_request_cache/"


def request(
    text: Union[str, List[str], pandas.DataFrame, None] = None,
    output: Union[str, None] = None,
    ids: Union[str, List[str], List[int], None] = None,
    text_column: Union[str, None] = None,
    id_column: Union[str, None] = None,
    files: Union[List[str], None] = None,
    directory: Union[str, None] = None,
    file_type: str = "txt",
    encoding: Union[str, None] = None,
    return_text=False,
    api_args: Union[dict, None] = None,
    frameworks: Union[str, List[str], None] = None,
    framework_prefix: Union[bool, None] = None,
    bundle_size=1000,
    bundle_byte_limit=75e5,
    collapse_lines=False,
    retry_limit=50,
    clear_cache=False,
    request_cache=True,
    cores=1,
    in_memory: Union[bool, None] = None,
    verbose=False,
    progress_bar: Union[str, bool] = os.getenv("RECEPTIVITI_PB", "True"),
    overwrite=False,
    make_request=True,
    text_as_paths=False,
    dotenv: Union[bool, str] = True,
    cache: Union[str, bool] = os.getenv("RECEPTIVITI_CACHE", ""),
    cache_overwrite=False,
    cache_format=os.getenv("RECEPTIVITI_CACHE_FORMAT", ""),
    key=os.getenv("RECEPTIVITI_KEY", ""),
    secret=os.getenv("RECEPTIVITI_SECRET", ""),
    url=os.getenv("RECEPTIVITI_URL", ""),
    version=os.getenv("RECEPTIVITI_VERSION", ""),
    endpoint=os.getenv("RECEPTIVITI_ENDPOINT", ""),
) -> pandas.DataFrame | None:
    """
    Send texts to be scored by the API.

    Args:
        text (str | list[str] | pandas.DataFrame): Text to be processed, as a string or vector of
            strings containing the text itself, or the path to a file from which to read in text.
            If a DataFrame, `text_column` is used to extract such a vector. A string may also
            represent a directory in which to search for files. To best ensure paths are not
            treated as texts, either set `text_as_path` to `True`, or use `directory` to enter
            a directory path, or `files` to enter a vector of file paths.
        output (str): Path to a file to write results to.
        ids (str | list[str | int]): Vector of IDs for each `text`, or a column name in `text`
            containing IDs.
        text_column (str): Column name in `text` containing text.
        id_column (str): Column name in `text` containing IDs.
        files (list[str]): Vector of file paths, as alternate entry to `text`.
        directory (str): A directory path to search for files in, as alternate entry to `text`.
        file_type (str): Extension of the file(s) to be read in from a directory (`txt` or `csv`).
        encoding (str | None): Encoding of file(s) to be read in; one of the
            [standard encodings](https://docs.python.org/3/library/codecs.html#standard-encodings).
            If this is `None` (default), encoding will be predicted for each file, but this can
            potentially fail, resulting in mis-encoded characters. For best (and fastest) results,
            specify encoding.
        return_text (bool): If `True`, will include a `text` column in the output with the
            original text.
        api_args (dict): Additional arguments to include in the request.
        frameworks (str | list): One or more names of frameworks to return.
        framework_prefix (bool): If `False`, will drop framework prefix from column names.
            If one framework is selected, will default to `False`.
        bundle_size (int): Maximum number of texts per bundle.
        bundle_byte_limit (float): Maximum byte size of each bundle.
        collapse_lines (bool): If `True`, will treat files as containing single texts, and
            collapse multiple lines.
        retry_limit (int): Number of times to retry a failed request.
        clear_cache (bool): If `True`, will delete the `cache` before processing.
        request_cache (bool): If `False`, will not temporarily save raw requests for reuse
            within a day.
        cores (int): Number of CPU cores to use when processing multiple bundles.
        in_memory (bool | None): If `False`, will write bundles to disc, to be loaded when
            processed. Defaults to `True` when processing in parallel.
        verbose (bool): If `True`, will print status messages and preserve the progress bar.
        progress_bar (str | bool): If `False`, will not display a progress bar.
        overwrite (bool): If `True`, will overwrite an existing `output` file.
        text_as_paths (bool): If `True`, will explicitly mark `text` as a list of file paths.
            Otherwise, this will be detected.
        dotenv (bool | str): Path to a .env file to read environment variables from. By default,
            will for a file in the current directory or `~/Documents`.
            Passed to `readin_env` as `path`.
        cache (bool | str): Path to a cache directory, or `True` to use the default directory.
        cache_overwrite (bool): If `True`, will not check the cache for previously cached texts,
            but will store results in the cache (unlike `cache = False`).
        cache_format (str): File format of the cache, of available Arrow formats.
        key (str): Your API key.
        secret (str): Your API secret.
        url (str): The URL of the API; defaults to `https://api.receptiviti.com`.
        version (str): Version of the API; defaults to `v1`.
        endpoint (str): Endpoint of the API; defaults to `framework`.

    Returns:
        Scores associated with each input text.

    Cache:
        If `cache` is specified, results for unique texts are saved in an Arrow database
        in the cache location (`os.getenv("RECEPTIVITI_CACHE")`), and are retrieved with
        subsequent requests. This ensures that the exact same texts are not re-sent to the API.
        This does, however, add some processing time and disc space usage.

        If `cache` if `True`, a default directory (`receptiviti_cache`) will be
        looked for in the system's temporary directory (`tempfile.gettempdir()`).

        The primary cache is checked when each bundle is processed, and existing results are
        loaded at that time. When processing many bundles in parallel, and many results have
        been cached, this can cause the system to freeze and potentially crash.
        To avoid this, limit the number of cores, or disable parallel processing.

        The `cache_format` arguments (or the `RECEPTIVITI_CACHE_FORMAT` environment variable) can be
        used to adjust the format of the cache.

        You can use the cache independently with
        `pyarrow.dataset.dataset(os.getenv("RECEPTIVITI_CACHE"))`.

        You can also set the `clear_cache` argument to `True` to clear the cache before it is used
        again, which may be useful if the cache has gotten big, or you know new results will be
        returned.

        Even if a cached result exists, it will be reprocessed if it does not have all of the
        variables of new results, but this depends on there being at least 1 uncached result. If,
        for instance, you add a framework to your account and want to reprocess a previously
        processed set of texts, you would need to first clear the cache.

        Either way, duplicated texts within the same call will only be sent once.

        The `request_cache` argument controls a more temporary cache of each bundle request. This
        is cleared after a day. You might want to set this to `False` if a new framework becomes
        available on your account and you want to process a set of text you re-processed recently.

        Another temporary cache is made when `in_memory` is `False`, which is the default when
        processing in parallel (when there is more than 1 bundle and `cores` is over 1). This is a
        temporary directory that contains a file for each unique bundle, which is read in as needed
        by the parallel workers.

    Parallelization:
        `text`s are split into bundles based on the `bundle_size` argument. Each bundle represents
        a single request to the API, which is why they are limited to 1000 texts and a total size
        of 10 MB. When there is more than one bundle and `cores` is greater than 1, bundles are
        processed by multiple cores.

        If you have texts spread across multiple files, they can be most efficiently processed in
        parallel if each file contains a single text (potentially collapsed from multiple lines).
        If files contain multiple texts (i.e., `collapse_lines=False`), then texts need to be
        read in before bundling in order to ensure bundles are under the length limit.

        If you are calling this function from a script, parallelization will involve rerunning
        that script in each process, so anything you don't want rerun should be protected by
        a check that `__name__` equals `"__main__"`
        (placed within an `if __name__ == "__main__":` clause).
    """
    if cores > 1 and current_process().name != "MainProcess":
        return None
    if output is not None and os.path.isfile(output) and not overwrite:
        msg = "`output` file already exists; use `overwrite=True` to overwrite it"
        raise RuntimeError(msg)
    start_time = perf_counter()

    if request_cache:
        if verbose:
            print(f"preparing request cache ({perf_counter() - start_time:.4f})")
        _manage_request_cache()

    # resolve credentials and check status
    if dotenv:
        readin_env("." if isinstance(dotenv, bool) else dotenv)
    if not url:
        url = os.getenv("RECEPTIVITI_URL", "https://api.receptiviti.com")
    url_parts = re.search("/([Vv]\\d+)/?([^/]+)?", url)
    if url_parts:
        from_url = url_parts.groups()
        if not version and from_url[0] is not None:
            version = from_url[0]
        if not endpoint and from_url[1] is not None:
            endpoint = from_url[1]
    url = ("https://" if re.match("http", url, re.I) is None else "") + re.sub("/+[Vv]\\d+(?:/.*)?$|/+$", "", url)
    if not key:
        key = os.getenv("RECEPTIVITI_KEY", "")
    if not secret:
        secret = os.getenv("RECEPTIVITI_SECRET", "")
    if not version:
        version = os.getenv("RECEPTIVITI_VERSION", "v1")
    version = version.lower()
    if not version or not re.search("^v\\d+$", version):
        msg = f"invalid version: {version}"
        raise RuntimeError(msg)
    if not endpoint:
        endpoint_default = "framework" if version == "v1" else "analyze"
        endpoint = os.getenv("RECEPTIVITI_ENDPOINT", endpoint_default)
    endpoint = re.sub("^.*/", "", endpoint).lower()
    if not endpoint or re.search("[^a-z]", endpoint):
        msg = f"invalid endpoint: {endpoint}"
        raise RuntimeError(msg)
    if version != "v1" and api_args:
        if "context" in api_args and "custom_context" in api_args:
            msg = "only one of `context` or `custom_context` may be specified"
            raise RuntimeError(msg)
    api_status = status(url, key, secret, dotenv, verbose=False)
    if not api_status or api_status.status_code != 200:
        msg = (
            f"API status failed: {api_status.status_code}: {api_status.reason}"
            if api_status
            else "URL is not reachable"
        )
        raise RuntimeError(msg)

    # resolve text and ids
    text_as_dir = False
    if text is None:
        if directory is not None:
            text = directory
            text_as_dir = True
        elif files is not None:
            text_as_paths = True
            text = files
        else:
            msg = "enter text as the first argument, or use the `files` or `directory` arguments"
            raise RuntimeError(msg)
    if isinstance(text, str) and (text_as_dir or text_as_paths or len(text) < 260):
        if not text_as_dir and os.path.isfile(text):
            if verbose:
                print(f"reading in texts from a file ({perf_counter() - start_time:.4f})")
            text = _readin([text], text_column, id_column, collapse_lines, encoding)
            if isinstance(text, pandas.DataFrame):
                id_column = "ids"
                text_column = "text"
            text_as_paths = False
        elif os.path.isdir(text):
            text = glob(f"{text}/*{file_type}")
            text_as_paths = True
        elif os.path.isdir(os.path.dirname(text)):
            msg = f"`text` appears to point to a directory, but it does not exist: {text}"
            raise RuntimeError(msg)
    if isinstance(text, pandas.DataFrame):
        if id_column is not None:
            if id_column in text:
                ids = text[id_column].to_list()
            else:
                msg = f"`id_column` ({id_column}) is not in `text`"
                raise IndexError(msg)
        if text_column is not None:
            if text_column in text:
                text = text[text_column].to_list()
            else:
                msg = f"`text_column` ({text_column}) is not in `text`"
                raise IndexError(msg)
        else:
            msg = "`text` is a DataFrame, but no `text_column` is specified"
            raise RuntimeError(msg)
    if isinstance(text, str):
        text = [text]
    text_is_path = all(isinstance(t, str) and (text_as_paths or len(t) < 260) and os.path.isfile(t) for t in text)
    if text_as_paths and not text_is_path:
        msg = "`text` treated as a list of files, but not all of the entries exist"
        raise RuntimeError(msg)
    if text_is_path and not collapse_lines:
        ids = text
        text = _readin(text, text_column, id_column, collapse_lines, encoding)
        if isinstance(text, pandas.DataFrame):
            if id_column is None:
                ids = text["ids"].to_list()
            elif id_column in text:
                ids = text[id_column].to_list()
            if text_column is None:
                text_column = "text"
            text = text[text_column].to_list()
        text_is_path = False
    if ids is None and text_is_path:
        ids = text

    id_specified = ids is not None
    if ids is None:
        ids = numpy.arange(1, len(text) + 1).tolist()
    elif len(ids) != len(text):
        msg = "`ids` is not the same length as `text`"
        raise RuntimeError(msg)
    original_ids = set(ids)
    if len(ids) != len(original_ids):
        msg = "`ids` contains duplicates"
        raise RuntimeError(msg)

    # prepare bundles
    if verbose:
        print(f"preparing text ({perf_counter() - start_time:.4f})")
    data = pandas.DataFrame({"text": text, "id": ids})
    n_original = len(data)
    data_subset = data[~(data.duplicated(subset=["text"]) | (data["text"] == "") | data["text"].isna())]
    n_texts = len(data_subset)
    if not n_texts:
        msg = "no valid texts to process"
        raise RuntimeError(msg)
    bundle_size = max(1, bundle_size)
    n_bundles = math.ceil(n_texts / min(1000, bundle_size))
    groups = data_subset.groupby(
        numpy.sort(numpy.tile(numpy.arange(n_bundles) + 1, bundle_size))[:n_texts],
        group_keys=False,
    )
    bundles = []
    for _, group in groups:
        if sys.getsizeof(group) > bundle_byte_limit:
            start = current = end = 0
            for txt in group["text"]:
                size = os.stat(txt).st_size if text_is_path else sys.getsizeof(txt)
                if size > bundle_byte_limit:
                    msg = f"one of your texts is over the bundle size limit ({bundle_byte_limit / 1e6} MB)"
                    raise RuntimeError(msg)
                if (current + size) > bundle_byte_limit:
                    bundles.append(group[start:end])
                    start = end = end + 1
                    current = size
                else:
                    end += 1
                    current += size
            bundles.append(group[start:])
        else:
            bundles.append(group)
    n_bundles = len(bundles)
    if verbose:
        print(
            f"prepared {n_texts} unique text{'s' if n_texts > 1 else ''} in "
            f"{n_bundles} {'bundles' if n_bundles > 1 else 'bundle'}",
            f"({perf_counter() - start_time:.4f})",
        )

    # process bundles
    if isinstance(cache, str):
        if cache:
            if clear_cache and os.path.exists(cache):
                shutil.rmtree(cache, True)
            os.makedirs(cache, exist_ok=True)
            if not cache_format:
                cache_format = os.getenv("RECEPTIVITI_CACHE_FORMAT", "parquet")
        else:
            cache = False
    opts = {
        "url": (f"{url}/{version}/{endpoint}/bulk" if version == "v1" else f"{url}/{version}/{endpoint}").lower(),
        "version": version,
        "auth": requests.auth.HTTPBasicAuth(key, secret),
        "retries": retry_limit,
        "add": {} if api_args is None else api_args,
        "request_cache": request_cache,
        "cache": "" if cache_overwrite or isinstance(cache, bool) and not cache else cache,
        "cache_format": cache_format,
        "make_request": make_request,
        "text_is_path": text_is_path,
        "text_column": text_column,
        "id_column": id_column,
        "collapse_lines": collapse_lines,
        "encoding": encoding,
    }
    if version != "v1" and api_args:
        opts["url"] += "?" + urllib.parse.urlencode(api_args)
    opts["add_hash"] = hashlib.md5(
        json.dumps(
            {**opts["add"], "url": opts["url"], "key": key, "secret": secret},
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if isinstance(progress_bar, str):
        progress_bar = progress_bar == "True"
    use_pb = (verbose and progress_bar) or progress_bar
    parallel = n_bundles > 1 and cores > 1
    if in_memory is None:
        in_memory = not parallel
    with TemporaryDirectory() as scratch_cache:
        if not in_memory:
            if verbose:
                print(f"writing to scratch cache ({perf_counter() - start_time:.4f})")

            def write_to_scratch(i: int, bundle: pandas.DataFrame):
                temp = f"{scratch_cache}/{i}.json"
                with open(temp, "wb") as scratch:
                    pickle.dump(bundle, scratch)
                return temp

            bundles = [write_to_scratch(i, b) for i, b in enumerate(bundles)]
        if parallel:
            if verbose:
                print(f"requesting in parallel ({perf_counter() - start_time:.4f})")
            waiter: "Queue[pandas.DataFrame]" = Queue()
            queue: "Queue[tuple[int, pandas.DataFrame]]" = Queue()
            manager = Process(
                target=_queue_manager,
                args=(queue, waiter, n_texts, n_bundles, use_pb, verbose),
            )
            manager.start()
            nb = math.ceil(n_bundles / min(n_bundles, cores))
            cores = math.ceil(n_bundles / nb)
            procs = [
                Process(
                    target=_process,
                    args=(bundles[(i * nb) : min(n_bundles, (i + 1) * nb)], opts, queue),
                )
                for i in range(cores)
            ]
            for cl in procs:
                cl.start()
            res = waiter.get()
        else:
            if verbose:
                print(f"requesting serially ({perf_counter() - start_time:.4f})")
            pb = tqdm(total=n_texts, leave=verbose) if use_pb else None
            res = _process(bundles, opts, pb=pb)
            if pb is not None:
                pb.close()
    if verbose:
        print(f"done requesting ({perf_counter() - start_time:.4f})")

    # finalize
    if not res.shape[0]:
        msg = "no results"
        raise RuntimeError(msg)
    if isinstance(cache, str):
        _update_cache(res, cache, cache_format, verbose, start_time, [e[0] for e in opts["add"]])
    if verbose:
        print(f"preparing output ({perf_counter() - start_time:.4f})")
    data.set_index("id", inplace=True)
    res.set_index("id", inplace=True)
    if len(res) != n_original:
        res = res.join(data["text"])
        data_absent = data.loc[list(set(data.index).difference(res.index))]
        data_absent = data_absent.loc[data_absent["text"].isin(res["text"])]
        if data.size:
            res = res.reset_index()
            res.set_index("text", inplace=True)
            data_dupes = res.loc[data_absent["text"]]
            data_dupes["id"] = data_absent.index.to_list()
            res = pandas.concat([res, data_dupes])
            res.reset_index(inplace=True, drop=True)
            res.set_index("id", inplace=True)
    res = res.join(data["text"], how="right")
    if not return_text:
        res.drop("text", axis=1, inplace=True)
    res = res.reset_index()

    if output is not None:
        if verbose:
            print(f"writing results to file: {output} ({perf_counter() - start_time:.4f})")
        res.to_csv(output, index=False)

    drops = ["custom", "bin"]
    if not id_specified:
        drops.append("id")
    res.drop(
        list({*drops}.intersection(res.columns)),
        axis="columns",
        inplace=True,
    )
    if frameworks is not None:
        if verbose:
            print(f"selecting frameworks ({perf_counter() - start_time:.4f})")
        if isinstance(frameworks, str):
            frameworks = [frameworks]
        if len(frameworks) == 1 and framework_prefix is None:
            framework_prefix = False
        select = []
        if id_specified:
            select.append("id")
        if return_text:
            select.append("text")
        select.append("text_hash")
        res = res.filter(regex=f"^(?:{'|'.join(select + frameworks)})(?:$|\\.)")
    if isinstance(framework_prefix, bool) and not framework_prefix:
        prefix_pattern = re.compile("^[^.]+\\.")
        res.columns = pandas.Index([prefix_pattern.sub("", col) for col in res.columns])

    if verbose:
        print(f"done ({perf_counter() - start_time:.4f})")

    return res


def _queue_manager(
    queue: "Queue[tuple[int, Union[pandas.DataFrame, None]]]",
    waiter: "Queue[pandas.DataFrame]",
    n_texts: int,
    n_bundles: int,
    use_pb=True,
    verbose=False,
):
    if use_pb:
        pb = tqdm(total=n_texts, leave=verbose)
    res: List[pandas.DataFrame] = []
    for size, chunk in iter(queue.get, None):
        if isinstance(chunk, pandas.DataFrame):
            if use_pb:
                pb.update(size)
            res.append(chunk)
            if len(res) >= n_bundles:
                break
        else:
            break
    waiter.put(pandas.concat(res, ignore_index=True, sort=False))


def _process(
    bundles: list,
    opts: dict,
    queue: Union["Queue[tuple[int, Union[pandas.DataFrame, None]]]", None] = None,
    pb: Union[tqdm, None] = None,
) -> pandas.DataFrame:
    reses: List[pandas.DataFrame] = []
    for bundle in bundles:
        if isinstance(bundle, str):
            with open(bundle, "rb") as scratch:
                bundle = pickle.load(scratch)
        body = []
        bundle.insert(0, "text_hash", "")
        if opts["text_is_path"]:
            bundle["text"] = _readin(
                bundle["text"],
                opts["text_column"],
                opts["id_column"],
                opts["collapse_lines"],
                opts["encoding"],
            )
        for i, text in enumerate(bundle["text"]):
            text_hash = hashlib.md5((opts["add_hash"] + text).encode()).hexdigest()
            bundle.iat[i, 0] = text_hash
            if opts["version"] == "v1":
                body.append({"content": text, "request_id": text_hash, **opts["add"]})
            else:
                body.append({"text": text, "request_id": text_hash})
        cached = None
        if opts["cache"] and os.path.isdir(opts["cache"] + "/bin=h"):
            db = pyarrow.dataset.dataset(
                opts["cache"],
                partitioning=pyarrow.dataset.partitioning(
                    pyarrow.schema([pyarrow.field("bin", pyarrow.string())]), flavor="hive"
                ),
                format=opts["cache_format"],
            )
            if "text_hash" in db.schema.names:
                su = db.filter(pyarrow.compute.field("text_hash").isin(bundle["text_hash"]))
                if su.count_rows() > 0:
                    cached = su.to_table().to_pandas(split_blocks=True, self_destruct=True)
        res = "failed to retrieve results"
        if cached is None or len(cached) < len(bundle):
            if cached is None or not len(cached):
                res = _prepare_results(body, opts)
            else:
                fresh = ~pyarrow.compute.is_in(
                    bundle["text_hash"].to_list(), pyarrow.array(cached["text_hash"])
                ).to_pandas(split_blocks=True, self_destruct=True)
                res = _prepare_results([body[i] for i, ck in enumerate(fresh) if ck], opts)
            if not isinstance(res, str):
                if cached is not None:
                    if len(res) != len(cached) or not all(cached.columns.isin(res.columns)):
                        cached = _prepare_results([body[i] for i, ck in enumerate(fresh) if not ck], opts)
                    res = pandas.concat([res, cached])
        else:
            res = cached
        if not isinstance(res, str):
            res = res.merge(bundle.loc[:, ["text_hash", "id"]], on="text_hash")
            reses.append(res)
        if queue is not None:
            queue.put((0, None) if isinstance(res, str) else (len(res), res))
        elif pb is not None:
            pb.update(len(bundle))
        if isinstance(res, str):
            raise RuntimeError(res)
    return reses[0] if len(reses) == 1 else pandas.concat(reses, ignore_index=True, sort=False)


def _prepare_results(body: list, opts: dict):
    json_body = json.dumps(body, separators=(",", ":"))
    bundle_hash = REQUEST_CACHE + hashlib.md5(json_body.encode()).hexdigest() + ".json" if opts["request_cache"] else ""
    raw_res = _request(
        json_body,
        opts["url"],
        opts["auth"],
        opts["retries"],
        bundle_hash,
        opts["make_request"],
    )
    if isinstance(raw_res, str):
        return raw_res
    res = pandas.json_normalize(raw_res)
    res.rename(columns={"request_id": "text_hash"}, inplace=True)
    if "text_hash" not in res:
        res.insert(0, "text_hash", [text["request_id"] for text in body])
    res.drop(
        list({"response_id", "language", "version", "error"}.intersection(res.columns)),
        axis="columns",
        inplace=True,
    )
    res.insert(res.shape[1], "bin", ["h" + h[0] for h in res["text_hash"]])
    return res


def _request(
    body: str,
    url: str,
    auth: requests.auth.HTTPBasicAuth,
    retries: int,
    cache="",
    execute=True,
) -> Union[dict, str]:
    if not os.path.isfile(cache):
        if not execute:
            return "`make_request` is `False`, but there are texts with no cached results"
        res = requests.post(url, body, auth=auth, timeout=9999)
        if cache and res.status_code == 200:
            with open(cache, "w", encoding="utf-8") as response:
                json.dump(res.json(), response)
    else:
        with open(cache, encoding="utf-8") as response:
            data = json.load(response)
            return data["results"] if "results" in data else data
    if res.status_code == 200:
        data = res.json()
        data = dict(data[0] if isinstance(data, list) else data)
        return data["results"] if "results" in data else data
    if os.path.isfile(cache):
        os.remove(cache)
    if retries > 0:
        cd = re.search(
            "[0-9]+(?:\\.[0-9]+)?",
            (res.json()["message"] if res.headers["Content-Type"] == "application/json" else res.text),
        )
        sleep(1 if cd is None else float(cd[0]) / 1e3)
        return _request(body, url, auth, retries - 1, cache)
    return f"request failed, and have no retries: {res.status_code}: {res.reason}"


def _update_cache(
    res: pandas.DataFrame,
    cache: str,
    cache_format: str,
    verbose: bool,
    start_time: float,
    add_names: list,
) -> None:
    part: pyarrow.dataset.Partitioning = pyarrow.dataset.partitioning(
        pyarrow.schema([pyarrow.field("bin", pyarrow.string())]), flavor="hive"
    )
    exclude = {"id", *add_names}

    def initialize_cache() -> None:
        initial: pandas.DataFrame = res.iloc[[0]].drop(
            exclude.intersection(res.columns),
            axis="columns",
        )
        initial["text_hash"] = ""
        initial["bin"] = "h"
        int_cols = initial.columns[
            (
                ~initial.columns.isin(["summary.word_count", "summary.sentence_count"])
                & (initial.dtypes != object).to_list()
            )
        ]
        initial[[int_cols]] = 0.1
        pyarrow.dataset.write_dataset(
            pyarrow.Table.from_pandas(initial),
            cache,
            basename_template="0-{i}." + cache_format,
            format=cache_format,
            partitioning=part,
            existing_data_behavior="overwrite_or_ignore",
        )

    if not os.path.isdir(cache + "/bin=h"):
        if verbose:
            print(f"initializing cache ({perf_counter() - start_time:.4f})")
        initialize_cache()
    db = pyarrow.dataset.dataset(cache, partitioning=part, format=cache_format)
    if any(name not in exclude and name not in db.schema.names for name in res.columns.to_list()):
        if verbose:
            print(
                "clearing cache since it contains columns not in new results",
                f"({perf_counter() - start_time:.4f})",
            )
        shutil.rmtree(cache, True)
        initialize_cache()
        db = pyarrow.dataset.dataset(cache, partitioning=part, format=cache_format)
    fresh = res[~res.duplicated(subset=["text_hash"])]
    su = db.filter(pyarrow.compute.field("text_hash").isin(fresh["text_hash"]))
    if su.count_rows() > 0:
        cached = ~pyarrow.compute.is_in(
            fresh["text_hash"].to_list(),
            su.scanner(columns=["text_hash"]).to_table()["text_hash"],
        ).to_pandas(split_blocks=True, self_destruct=True)
        if any(cached):
            fresh = fresh[cached.to_list()]
        else:
            return
    n_new = len(fresh)
    if n_new:
        if verbose:
            print(
                f"adding {n_new} result{'' if n_new == 1 else 's'}",
                f"to cache ({perf_counter() - start_time:.4f})",
            )
        pyarrow.dataset.write_dataset(
            pyarrow.Table.from_pandas(
                fresh.drop(
                    list(exclude.intersection(fresh.columns)),
                    axis="columns",
                )
            ),
            cache,
            basename_template=str(math.ceil(time())) + "-{i}." + cache_format,
            format=cache_format,
            partitioning=part,
            existing_data_behavior="overwrite_or_ignore",
        )


def _manage_request_cache():
    os.makedirs(REQUEST_CACHE, exist_ok=True)
    try:
        refreshed = time()
        log_file = REQUEST_CACHE + "log.txt"
        if os.path.exists(log_file):
            with open(log_file, encoding="utf-8") as log:
                logged = log.readline()
                if isinstance(logged, list):
                    logged = logged[0]
                refreshed = float(logged)
        else:
            with open(log_file, "w", encoding="utf-8") as log:
                log.write(str(time()))
        if time() - refreshed > 86400:
            for cached_request in glob(REQUEST_CACHE + "*.json"):
                os.remove(cached_request)
    except Exception as exc:
        msg = "failed to manage request cache"
        raise RuntimeWarning(msg) from exc


def _readin(
    paths: List[str],
    text_column: Union[str, None],
    id_column: Union[str, None],
    collapse_lines: bool,
    encoding: Union[str, None],
) -> Union[List[str], pandas.DataFrame]:
    text = []
    ids = []
    sel = []
    if text_column is not None:
        sel.append(text_column)
    if id_column is not None:
        sel.append(id_column)
    enc = encoding
    predict_encoding = enc is None
    if predict_encoding:
        detect = UniversalDetector()

        def handle_encoding(file: str):
            detect.reset()
            with open(file, "rb") as text:
                while True:
                    chunk = text.read(1024)
                    if not chunk:
                        break
                    detect.feed(chunk)
                    if detect.done:
                        break
            detected = detect.close()["encoding"]
            if detected is None:
                msg = "failed to detect encoding; please specify with the `encoding` argument"
                raise RuntimeError(msg)
            return detected

    if os.path.splitext(paths[0])[1] == ".txt" and not sel:
        if collapse_lines:
            for file in paths:
                if predict_encoding:
                    enc = handle_encoding(file)
                with open(file, encoding=enc, errors="ignore") as texts:
                    text.append(" ".join([line.rstrip() for line in texts]))
        else:
            for file in paths:
                if predict_encoding:
                    enc = handle_encoding(file)
                with open(file, encoding=enc, errors="ignore") as texts:
                    lines = [line.rstrip() for line in texts]
                    text += lines
                    ids += [file] if len(lines) == 1 else [file + str(i + 1) for i in range(len(lines))]
            return pandas.DataFrame({"text": text, "ids": ids})
    elif collapse_lines:
        for file in paths:
            if predict_encoding:
                enc = handle_encoding(file)
            temp = pandas.read_csv(file, encoding=enc, usecols=sel)
            text.append(" ".join(temp[text_column]))
    else:
        for file in paths:
            if predict_encoding:
                enc = handle_encoding(file)
            temp = pandas.read_csv(file, encoding=enc, usecols=sel)
            if text_column not in temp:
                msg = f"`text_column` ({text_column}) was not found in all files"
                raise IndexError(msg)
            text += temp[text_column].to_list()
            ids += (
                temp[id_column].to_list()
                if id_column is not None
                else [file] if len(temp) == 1 else [file + str(i + 1) for i in range(len(temp))]
            )
        return pandas.DataFrame({"text": text, "ids": ids})
    return text
