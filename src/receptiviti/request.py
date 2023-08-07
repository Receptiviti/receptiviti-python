"""Make requests to the API."""

from typing import Union, List
import os
import shutil
from glob import glob
import pickle
from tempfile import gettempdir
from time import perf_counter, sleep, time
from multiprocessing import Process, Queue, cpu_count
from tqdm import tqdm
from tempfile import TemporaryDirectory
import pyarrow
from pyarrow import dataset
from pyarrow import compute
import sys
import re
import hashlib
import requests
import json
import math
import numpy
import pandas
from .status import status
from .readin_env import readin_env

CACHE = gettempdir() + "/receptiviti_cache/"
REQUEST_CACHE = gettempdir() + "/receptiviti_request_cache/"


def request(
    text: Union[str, list, pandas.DataFrame],
    output: Union[str, None] = None,
    ids: Union[str, List[Union[str, int]], None] = None,
    text_column: Union[str, None] = None,
    id_column: Union[str, None] = None,
    file_type: str = "txt",
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
    cores=cpu_count() - 2,
    in_memory: Union[bool, None] = None,
    verbose=False,
    progress_bar=True,
    overwrite=False,
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
) -> pandas.DataFrame:
    """
    Send texts to be scored by the API.

    Args:
      text (str | list | pandas.DataFrame): Text to be processed.
      output (str): Path to a file to write results to.
      ids (str | list): Vector of IDs for each `text`, or a column name in `text` containing IDs.
      text_column (str): Column name in `text` containing text.
      id_column (str): Column name in `text` containing IDs.
      file_type (str): Extension of the file(s) to be read in from a directory (`txt` or `csv`).
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
      request_cache (bool): If `False`, will not temporarily save raw requests for reuse within a day.
      cores (int): Number of CPU cores to use when processing multiple bundles.
      in_memory (bool | None): If `False`, will write bundles to disc, to be loaded when processed.
        Defaults to `True` when processing in parallel.
      verbose (bool): If `True`, will print status messages and preserve the progress bar.
      progress_bar (bool): If `False`, will not display a progress bar.
      overwrite (bool): If `True`, will overwrite an existing `output` file.
      text_as_paths (bool): If `True`, will explicitly mark `text` as a list of file paths.
        Otherwise, this will be detected.
      dotenv (bool | str): Path to a .env file to read environment variables from. By default,
        will for a file in the current directory or `~/Documents`. Passed to `readin_env` as `path`.
      cache (bool | str): Path to a cache directory, `True` or `""` to use the default directory, or `False`
        to not use a cache.
      cache_overwrite (bool): If `True`, will not check the cache for previously cached texts, but will
        store results in the cache (unlike `cache = False`).
      cache_format (str): File format of the cache, of available Arrow formats.
      key (str): Your API key.
      secret (str): Your API secret.
      url (str): The URL of the API; defaults to `https://api.receptiviti.com`.
      version (str): Version of the API; defaults to `v1`.
      endpoint (str): Endpoint of the API; defaults to `framework`.

    Returns:
      Scores associated with each input text.
    """
    if output is not None and os.path.isfile(output) and not overwrite:
        raise RuntimeError("`output` file already exists; use `overwrite=True` to overwrite it")
    start_time = perf_counter()

    if request_cache:
        if verbose:
            print(f"preparing request cache ({perf_counter() - start_time:.4f})")
        _manage_request_cache()

    # resolve credentials and check status
    if dotenv:
        readin_env("." if isinstance(dotenv, bool) else dotenv)
    if url == "":
        url = os.getenv("RECEPTIVITI_URL", "https://api.receptiviti.com")
    url_parts = re.search("/([Vv]\\d)/?([^/]+)?", url)
    if url_parts:
        from_url = url_parts.groups()
        if version == "" and from_url[0] is not None:
            version = from_url[0]
        if endpoint == "" and from_url[1] is not None:
            endpoint = from_url[1]
    url = ("https://" if re.match("http", url, re.I) is None else "") + re.sub(
        "/[Vv]\\d(?:/.*)?$|/+$", "", url
    )
    if key == "":
        key = os.getenv("RECEPTIVITI_KEY", "")
    if secret == "":
        secret = os.getenv("RECEPTIVITI_SECRET", "")
    if version == "":
        version = os.getenv("RECEPTIVITI_VERSION", "v1")
    if endpoint == "":
        endpoint = os.getenv(
            "RECEPTIVITI_ENDPOINT",
            "framework" if version.lower() == "v1" else "taxonomies",
        )
    api_status = status(url, key, secret, dotenv, verbose=False)
    if api_status.status_code != 200:
        raise RuntimeError(f"API status failed: {api_status.status_code}: {api_status.reason}")

    # resolve text and ids
    def readin(
        paths: list[str],
        text_cols=text_column,
        id_cols=id_column,
        collapse=collapse_lines,
    ) -> list:
        sel = []
        if text_cols is not None:
            sel.append(text_cols)
        if id_cols is not None:
            sel.append(id_cols)
        text = []
        if os.path.splitext(paths[0])[1] == ".txt" and not sel:
            for file in paths:
                with open(file, encoding="utf-8") as texts:
                    lines = [line.rstrip() for line in texts]
                    if collapse:
                        text.append(" ".join(lines))
                    else:
                        text += lines
        else:
            text = pandas.concat([pandas.read_csv(file, usecols=sel) for file in paths])
        return text

    if isinstance(text, str):
        if os.path.isfile(text):
            if verbose:
                print(f"reading in texts from a file ({perf_counter() - start_time:.4f})")
            text = readin([text])
            text_as_paths = False
        elif os.path.isdir(text):
            text = glob(f"{text}/*{file_type}")
            text_as_paths = True
    if isinstance(text, pandas.DataFrame):
        if id_column is not None:
            if id_column in text:
                ids = text[id_column].to_list()
            else:
                raise IndexError(f"`id_column` ({id_column}) is not in `text`")
        if text_column is not None:
            if text_column in text:
                text = text[text_column].to_list()
            else:
                raise IndexError(f"`text_column` ({text_column}) is not in `text`")
        else:
            raise RuntimeError("`text` is a DataFrame, but no `text_column` is specified")
    if isinstance(text, str):
        text = [text]
    text_is_path = all((len(t) < 500 and os.path.isfile(t) for t in text))
    if text_as_paths and not text_is_path:
        raise RuntimeError("`text` treated as a list of files, but not all of the entries exist")
    if text_is_path and not collapse_lines:
        text = readin(text)
        text_is_path = False

    id_specified = ids is not None
    if not id_specified:
        ids = numpy.arange(1, len(text) + 1)
    elif len(ids) != len(text):
        raise RuntimeError("`ids` is not the same length as `text`")
    original_ids = set(ids)
    if len(ids) != len(original_ids):
        raise RuntimeError("`ids` contains duplicates")

    # prepare bundles
    if verbose:
        print(f"preparing text ({perf_counter() - start_time:.4f})")
    data = pandas.DataFrame({"text": text, "id": ids})
    n_original = len(data)
    groups = data[~(data.duplicated(subset=["text"]) | (data["text"] == "") | data["text"].isna())]
    n_texts = len(groups)
    if not n_texts:
        raise RuntimeError("no valid texts to process")
    bundle_size = max(1, bundle_size)
    n_bundles = math.ceil(n_texts / min(1000, bundle_size))
    groups = groups.groupby(
        numpy.sort(numpy.tile(numpy.arange(n_bundles) + 1, bundle_size))[:n_texts],
        group_keys=False,
    )
    bundles = []
    getsize = sys.getsizeof if not text_is_path else lambda f: os.stat(f).st_size
    for _, group in groups:
        if getsize(group) > bundle_byte_limit:
            start = current = end = 0
            for txt in group["text"]:
                size = getsize(txt)
                if size > bundle_byte_limit:
                    raise RuntimeError(
                        "one of your texts is over the bundle size"
                        + f" limit ({bundle_byte_limit / 1e6} MB)"
                    )
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
        if cache == "":
            cache = CACHE
        if clear_cache and os.path.exists(cache):
            shutil.rmtree(cache, True)
        os.makedirs(cache, exist_ok=True)
    opts = {
        "url": f"{url}/{version}/{endpoint}/bulk".lower(),
        "auth": requests.auth.HTTPBasicAuth(key, secret),
        "retries": retry_limit,
        "add": {} if api_args is None else api_args,
        "are_paths": text_is_path,
        "request_cache": request_cache,
        "cache": "" if cache_overwrite or isinstance(cache, bool) and not cache else cache,
        "cache_format": cache_format,
        "make_request": make_request,
    }
    opts["add_hash"] = hashlib.md5(
        json.dumps(
            {**opts["add"], "url": opts["url"], "key": key, "secret": secret},
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    use_pb = (verbose and progress_bar) or progress_bar
    parallel = n_bundles > 1 and cores > 1
    if in_memory is None:
        in_memory = not parallel
    with TemporaryDirectory() as scratch_cache:
        if not in_memory:
            if verbose:
                print(f"writing to scratch cache ({perf_counter() - start_time:.4f})")

            def write_to_scratch(i: int, bundle: pandas.DataFrame):
                temp = f"{scratch_cache}/{i}.pickle"
                with open(temp, "wb") as scratch:
                    pickle.dump(bundle, scratch)
                return temp

            bundles = [write_to_scratch(i, b) for i, b in enumerate(bundles)]
        if parallel:
            if verbose:
                print(f"requesting in parallel ({perf_counter() - start_time:.4f})")
            waiter = Queue()
            queue = Queue()
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
            for cl in procs:
                cl.join()
            res = waiter.get()
        else:
            if verbose:
                print(f"requesting serially ({perf_counter() - start_time:.4f})")
            if use_pb:
                pb = tqdm(total=n_texts, leave=verbose)
            res = _process(bundles, opts, pb=pb)
            if use_pb:
                pb.close()
    if verbose:
        print(f"done requesting ({perf_counter() - start_time:.4f})")

    # finalize
    if not res.shape[0]:
        raise RuntimeError("no results")
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
    res = res.join(data["text"], how="outer")
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
        {*drops}.intersection(res.columns),
        axis="columns",
        inplace=True,
    )
    if frameworks is not None:
        if verbose:
            print(f"selecting frameworks ({perf_counter() - start_time:.4f})")
        if isinstance(frameworks, str) or len(frameworks) == 1:
            if framework_prefix is None:
                framework_prefix = False
            frameworks = [frameworks]
        select = []
        if id_specified:
            select.append("id")
        if return_text:
            select.append("text")
        select.append("text_hash")
        res = res.filter(regex=f"^(?:{'|'.join(select + frameworks)})(?:$|\\.)")
    if isinstance(framework_prefix, bool) and not framework_prefix:
        prefix_pattern = re.compile("^[^.]+\\.")
        res.columns = [prefix_pattern.sub("", col) for col in res.columns]

    if verbose:
        print(f"done ({perf_counter() - start_time:.4f})")

    return res


def _queue_manager(
    queue: Queue,
    waiter: Queue,
    n_texts: int,
    n_bundles: int,
    use_pb=True,
    verbose=False,
):
    if use_pb:
        pb = tqdm(total=n_texts, leave=verbose)
    res = []
    for size, chunk in iter(queue.get, None):
        if size:
            if use_pb:
                pb.update(size)
            res.append(chunk)
            if len(res) >= n_bundles:
                break
        else:
            break
    waiter.put(pandas.concat(res, ignore_index=True, sort=False))
    return


def _process(
    bundles: list,
    opts: dict,
    queue: Union[Queue, None] = None,
    pb: Union[tqdm, None] = None,
) -> pandas.DataFrame:
    reses = []
    for bundle in bundles:
        if isinstance(bundle, str):
            with open(bundle, "rb") as scratch:
                bundle = pickle.load(scratch)
        body = []
        bundle.insert(0, "text_hash", "")
        for i, text in enumerate(bundle["text"]):
            text_hash = hashlib.md5((opts["add_hash"] + text).encode()).hexdigest()
            bundle.iat[i, 0] = text_hash
            body.append({"content": text, "request_id": text_hash, **opts["add"]})
        cached = None
        if opts["cache"] != "" and os.path.isdir(opts["cache"] + "/bin=h"):
            db = dataset.dataset(
                opts["cache"],
                partitioning=dataset.partitioning(
                    pyarrow.schema([("bin", pyarrow.string())]), flavor="hive"
                ),
                format=opts["cache_format"],
            )
            if "text_hash" in db.schema.names:
                su = db.filter(compute.field("text_hash").isin(bundle["text_hash"]))
                if su.count_rows() > 0:
                    cached = su.to_table().to_pandas(split_blocks=True, self_destruct=True)
        res = "failed to retrieve results"
        if cached is None or len(cached) < len(bundle):
            if cached is None or not len(cached):
                res = _prepare_results(body, opts)
            else:
                fresh = ~compute.is_in(
                    bundle["text_hash"].to_list(), pyarrow.array(cached["text_hash"])
                ).to_pandas(split_blocks=True, self_destruct=True)
                res = _prepare_results([body[i] for i, ck in enumerate(fresh) if ck], opts)
            if not isinstance(res, str):
                if cached is not None:
                    if len(res) != len(cached) or not all(cached.columns.isin(res.columns)):
                        cached = _prepare_results(
                            [body[i] for i, ck in enumerate(fresh) if not ck], opts
                        )
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


def _prepare_results(body: list, opts: list):
    json_body = json.dumps(body, separators=(",", ":"))
    bundle_hash = (
        REQUEST_CACHE + hashlib.md5(json_body.encode()).hexdigest() + ".pickle"
        if opts["request_cache"]
        else ""
    )
    res = _request(
        json_body,
        opts["url"],
        opts["auth"],
        opts["retries"],
        bundle_hash,
        opts["make_request"],
    )
    if not isinstance(res, str):
        res = pandas.json_normalize(res)
        res.rename(columns={"request_id": "text_hash"}, inplace=True)
        if "text_hash" not in res:
            res.insert(0, "text_hash", [text["request_id"] for text in body])
        res.drop(
            {"response_id", "language", "version", "error"}.intersection(res.columns),
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
        if cache != "":
            with open(cache, "wb") as response:
                pickle.dump(res, response)
    else:
        with open(cache, "rb") as response:
            res = pickle.load(response)
    if res.status_code == 200:
        res = res.json()
        if "results" in res:
            res = res["results"]
        return res
    if os.path.isfile(cache):
        os.remove(cache)
    if retries > 0:
        cd = re.search(
            "[0-9]+(?:\\.[0-9]+)?",
            res.json()["message"]
            if res.headers["Content-Type"] == "application/json"
            else res.text,
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
):
    part = dataset.partitioning(pyarrow.schema([("bin", pyarrow.string())]), flavor="hive")
    exclude = {"id", *add_names}

    def initialize_cache():
        initial = res.iloc[[0]].drop(
            exclude.intersection(res.columns),
            axis="columns",
        )
        initial["text_hash"] = ""
        initial["bin"] = "h"
        initial.loc[
            :,
            ~initial.columns.isin(["summary.word_count", "summary.sentence_count"])
            & (initial.dtypes != object).to_list(),
        ] = 0.1
        dataset.write_dataset(
            pyarrow.Table.from_pandas(initial),
            cache,
            partitioning=part,
            format=cache_format,
            existing_data_behavior="overwrite_or_ignore",
        )

    if not os.path.isdir(cache + "/bin=h"):
        if verbose:
            print(f"initializing cache ({perf_counter() - start_time:.4f})")
        initialize_cache()
    db = dataset.dataset(cache, partitioning=part, format=cache_format)
    if any((name not in exclude and name not in db.schema.names for name in res.columns.to_list())):
        if verbose:
            print(
                f"clearing cache since it contains columns not in new results ({perf_counter() - start_time:.4f})"
            )
        shutil.rmtree(cache, True)
        initialize_cache()
        db = dataset.dataset(cache, partitioning=part, format=cache_format)
    fresh = res[~res.duplicated(subset=["text_hash"])]
    su = db.filter(compute.field("text_hash").isin(fresh["text_hash"]))
    if su.count_rows() > 0:
        cached = ~compute.is_in(
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
        dataset.write_dataset(
            pyarrow.Table.from_pandas(
                fresh.drop(
                    exclude.intersection(fresh.columns),
                    axis="columns",
                )
            ),
            cache,
            partitioning=part,
            format=cache_format,
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
            for cached_request in glob(REQUEST_CACHE + "*.pickle"):
                os.remove(cached_request)
    except Exception as exc:
        raise RuntimeWarning("failed to manage request cache") from exc
