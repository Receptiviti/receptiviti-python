"""Make requests to the API."""

import hashlib
import json
import math
import os
import pickle
import re
import sys
import urllib.parse
import warnings
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
import pyarrow.feather
import pyarrow.parquet
import requests
from chardet.universaldetector import UniversalDetector
from tqdm import tqdm

from receptiviti.status import _resolve_request_def, status

CACHE = gettempdir() + "/receptiviti_cache/"
REQUEST_CACHE = gettempdir() + "/receptiviti_request_cache/"


def _manage_request(
    text: Union[str, List[str], pandas.DataFrame, None] = None,
    ids: Union[str, List[str], List[int], None] = None,
    text_column: Union[str, None] = None,
    id_column: Union[str, None] = None,
    files: Union[List[str], None] = None,
    directory: Union[str, None] = None,
    file_type="txt",
    encoding: Union[str, None] = None,
    context="written",
    api_args: Union[dict, None] = None,
    bundle_size=1000,
    bundle_byte_limit=75e5,
    collapse_lines=False,
    retry_limit=50,
    request_cache=True,
    cores=1,
    collect_results=True,
    in_memory: Union[bool, None] = None,
    verbose=False,
    progress_bar: Union[str, bool] = os.getenv("RECEPTIVITI_PB", "True"),
    make_request=True,
    text_as_paths=False,
    dotenv: Union[bool, str] = True,
    cache=os.getenv("RECEPTIVITI_CACHE", ""),
    cache_overwrite=False,
    cache_format=os.getenv("RECEPTIVITI_CACHE_FORMAT", "parquet"),
    key=os.getenv("RECEPTIVITI_KEY", ""),
    secret=os.getenv("RECEPTIVITI_SECRET", ""),
    url=os.getenv("RECEPTIVITI_URL", ""),
    version=os.getenv("RECEPTIVITI_VERSION", ""),
    endpoint=os.getenv("RECEPTIVITI_ENDPOINT", ""),
    to_norming=False,
) -> tuple[pandas.DataFrame, Union[pandas.DataFrame, None], bool]:
    if cores > 1 and current_process().name != "MainProcess":
        return (pandas.DataFrame(), None, False)
    start_time = perf_counter()

    if request_cache:
        if verbose:
            print(f"preparing request cache ({perf_counter() - start_time:.4f})")
        _manage_request_cache()

    # resolve credentials and check status
    full_url, url, key, secret = _resolve_request_def(url, key, secret, dotenv)
    url_parts = re.search("/([Vv]\\d+)/?([^/]+)?", full_url)
    if url_parts:
        from_url = url_parts.groups()
        if not version and from_url[0] is not None:
            version = from_url[0]
        if not endpoint and from_url[1] is not None:
            endpoint = from_url[1]
    if to_norming:
        version = "v2"
        endpoint = "norming"
        request_cache = False
    else:
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
    api_status = status(url, key, secret, dotenv, verbose=False)
    if api_status is None or api_status.status_code != 200:
        msg = (
            "URL is not reachable"
            if api_status is None
            else f"API status failed: {api_status.status_code}: {api_status.reason}"
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
    opts = {
        "url": (
            full_url
            if to_norming
            else (
                f"{url}/{version}/{endpoint}/bulk" if version == "v1" else f"{url}/{version}/{endpoint}/{context}"
            ).lower()
        ),
        "version": version,
        "auth": requests.auth.HTTPBasicAuth(key, secret),
        "retries": retry_limit,
        "add": {} if api_args is None else api_args,
        "request_cache": request_cache,
        "cache": cache,
        "cache_overwrite": cache_overwrite,
        "cache_format": cache_format,
        "to_norming": to_norming,
        "make_request": make_request,
        "text_is_path": text_is_path,
        "text_column": text_column,
        "id_column": id_column,
        "collapse_lines": collapse_lines,
        "encoding": encoding,
        "collect_results": collect_results,
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
            waiter: "Queue[List[Union[pandas.DataFrame, None]]]" = Queue()
            queue: "Queue[tuple[int, Union[pandas.DataFrame, None]]]" = Queue()
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

    return (data, pandas.concat(res, ignore_index=True, sort=False) if opts["collect_results"] else None, id_specified)


def _queue_manager(
    queue: "Queue[tuple[int, Union[pandas.DataFrame, None]]]",
    waiter: "Queue[List[Union[pandas.DataFrame, None]]]",
    n_texts: int,
    n_bundles: int,
    use_pb=True,
    verbose=False,
):
    if use_pb:
        pb = tqdm(total=n_texts, leave=verbose)
    res: List[Union[pandas.DataFrame, None]] = []
    for size, chunk in iter(queue.get, None):
        if size:
            if use_pb:
                pb.update(size)
            res.append(chunk)
            if len(res) >= n_bundles:
                break
        else:
            break
    waiter.put(res)


def _process(
    bundles: List[pandas.DataFrame],
    opts: dict,
    queue: Union["Queue[tuple[int, Union[pandas.DataFrame, None]]]", None] = None,
    pb: Union[tqdm, None] = None,
) -> List[Union[pandas.DataFrame, None]]:
    reses: List[Union[pandas.DataFrame, None]] = []
    for bundle in bundles:
        if isinstance(bundle, str):
            with open(bundle, "rb") as scratch:
                bundle = pickle.load(scratch)
        body = []
        bundle.insert(0, "text_hash", "")
        if opts["text_is_path"]:
            bundle["text"] = _readin(
                bundle["text"].to_list(),
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
        ncached = 0
        cached: Union[pandas.DataFrame, None] = None
        cached_cols: List[str] = []
        if not opts["cache_overwrite"] and opts["cache"] and os.listdir(opts["cache"]):
            db = pyarrow.dataset.dataset(
                opts["cache"],
                partitioning=pyarrow.dataset.partitioning(
                    pyarrow.schema([pyarrow.field("bin", pyarrow.string())]), flavor="hive"
                ),
                format=opts["cache_format"],
            )
            cached_cols = db.schema.names
            if "text_hash" in cached_cols:
                su = db.filter(pyarrow.compute.field("text_hash").isin(bundle["text_hash"]))
                ncached = su.count_rows()
                if ncached > 0:
                    cached = (
                        su.to_table().to_pandas(split_blocks=True, self_destruct=True)
                        if opts["collect_results"]
                        else su.scanner(["text_hash"]).to_table().to_pandas(split_blocks=True, self_destruct=True)
                    )
        res: Union[str, pandas.DataFrame] = "failed to retrieve results"
        json_body = json.dumps(body, separators=(",", ":"))
        bundle_hash = hashlib.md5(json_body.encode()).hexdigest()
        if cached is None or ncached < len(bundle):
            if cached is None:
                res = _prepare_results(json_body, bundle_hash, opts)
            else:
                fresh = ~bundle["text_hash"].isin(cached["text_hash"])
                json_body = json.dumps([body[i] for i, ck in enumerate(fresh) if ck], separators=(",", ":"))
                res = _prepare_results(json_body, hashlib.md5(json_body.encode()).hexdigest(), opts)
            if not isinstance(res, str):
                if ncached:
                    if res.ndim != len(cached_cols) or not pandas.Series(cached_cols).isin(res.columns).all():
                        json_body = json.dumps([body[i] for i, ck in enumerate(fresh) if ck], separators=(",", ":"))
                        cached = _prepare_results(json_body, hashlib.md5(json_body.encode()).hexdigest(), opts)
                    if cached is not None and opts["collect_results"]:
                        res = pandas.concat([res, cached])
                if opts["cache"]:
                    writer = _get_writer(opts["cache_format"])
                    schema = pyarrow.schema(
                        (
                            col,
                            (
                                pyarrow.string()
                                if res[col].dtype == "O"
                                else (
                                    pyarrow.int32()
                                    if col in ["summary.word_count", "summary.sentence_count"]
                                    else pyarrow.float32()
                                )
                            ),
                        )
                        for col in res.columns
                        if col not in ["id", "bin", *(opts["add"].keys() if opts["add"] else [])]
                    )
                    for id_bin, d in res.groupby("bin"):
                        bin_dir = f"{opts['cache']}/bin={id_bin}"
                        os.makedirs(bin_dir, exist_ok=True)
                        writer(
                            pyarrow.Table.from_pandas(d, schema, preserve_index=False),
                            f"{bin_dir}/fragment-{bundle_hash}-0.{opts['cache_format']}",
                        )
        else:
            res = cached
        nres = len(res)
        if not opts["collect_results"]:
            reses.append(None)
        elif not isinstance(res, str):
            if "text_hash" in res:
                res = res.merge(bundle[["text_hash", "id"]], on="text_hash")
            reses.append(res)
        if queue is not None:
            queue.put((0, None) if isinstance(res, str) else (nres + ncached, res))
        elif pb is not None:
            pb.update(len(bundle))
        if isinstance(res, str):
            raise RuntimeError(res)
    return reses


def _prepare_results(body: str, bundle_hash: str, opts: dict):
    raw_res = _request(
        body,
        opts["url"],
        opts["auth"],
        opts["retries"],
        REQUEST_CACHE + bundle_hash + ".json" if opts["request_cache"] else "",
        opts["to_norming"],
        opts["make_request"],
    )
    if isinstance(raw_res, str):
        return raw_res
    res = pandas.json_normalize(raw_res)
    if "request_id" in res:
        res.rename(columns={"request_id": "text_hash"}, inplace=True)
        res.drop(
            list({"response_id", "language", "version", "error"}.intersection(res.columns)),
            axis="columns",
            inplace=True,
        )
        res.insert(res.ndim, "bin", ["h" + h[0] for h in res["text_hash"]])
    return res


def _request(
    body: str,
    url: str,
    auth: requests.auth.HTTPBasicAuth,
    retries: int,
    cache="",
    to_norming=False,
    execute=True,
) -> Union[dict, str]:
    if not os.path.isfile(cache):
        if not execute:
            return "`make_request` is `False`, but there are texts with no cached results"
        if to_norming:
            res = requests.patch(url, body, auth=auth, timeout=9999)
        else:
            res = requests.post(url, body, auth=auth, timeout=9999)
        if cache and res.status_code == 200:
            with open(cache, "w", encoding="utf-8") as response:
                json.dump(res.json(), response)
    else:
        with open(cache, encoding="utf-8") as response:
            data = json.load(response)
            return data["results"] if "results" in data else data
    data = res.json()
    if res.status_code == 200:
        data = dict(data[0] if isinstance(data, list) else data)
        return data["results"] if "results" in data else data
    if os.path.isfile(cache):
        os.remove(cache)
    if retries > 0 and "code" in data and data["code"] == 1420:
        cd = re.search(
            "[0-9]+(?:\\.[0-9]+)?",
            (res.json()["message"] if res.headers["Content-Type"] == "application/json" else res.text),
        )
        sleep(1 if cd is None else float(cd[0]) / 1e3)
        return _request(body, url, auth, retries - 1, cache, to_norming)
    return f"request failed, and have no retries: {res.status_code}: {data['error'] if 'error' in data else res.reason}"


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
        warnings.warn(UserWarning(f"failed to manage request cache: {exc}"), stacklevel=2)


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


def _get_writer(write_format: str):
    if write_format == "parquet":
        return pyarrow.parquet.write_table
    if write_format == "feather":
        return pyarrow.feather.write_feather
