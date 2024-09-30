"""Make requests to the API."""

import math
import os
import re
import shutil
from multiprocessing import current_process
from tempfile import gettempdir
from time import perf_counter, time
from typing import List, Union

import pandas
import pyarrow
import pyarrow.compute
import pyarrow.dataset

from receptiviti.manage_request import _manage_request
from receptiviti.norming import norming

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

    # check norming context status
    if api_args and "custom_context" in api_args:
        if "context" in api_args:
            msg = "only one of `context` or `custom_context` may be specified"
            raise RuntimeError(msg)
        if verbose:
            print(f"retrieving custom norming list ({perf_counter() - start_time:.4f})")
        norming_status: pandas.DataFrame = norming(url=url, key=key, secret=secret, verbose=False)
        if len(norming_status) == 0 or api_args["custom_context"] not in norming_status["name"].values:
            msg = f"custom norming context {api_args['custom_context']} is not on record"
            raise RuntimeError(msg)
        norming_status = norming_status[norming_status["name"] == api_args["custom_context"]]
        if norming_status.iloc[0]["status"] != "completed":
            msg = f"custom norming context {api_args['custom_context']} has not been completed"
            raise RuntimeError(msg)

    if isinstance(cache, str):
        if cache:
            if clear_cache and os.path.exists(cache):
                shutil.rmtree(cache, True)
            os.makedirs(cache, exist_ok=True)
            if not cache_format:
                cache_format = os.getenv("RECEPTIVITI_CACHE_FORMAT", "parquet")
        else:
            cache = False

    data, res, id_specified = _manage_request(
        text=text,
        ids=ids,
        text_column=text_column,
        id_column=id_column,
        files=files,
        directory=directory,
        file_type=file_type,
        encoding=encoding,
        api_args=api_args,
        bundle_size=bundle_size,
        bundle_byte_limit=bundle_byte_limit,
        collapse_lines=collapse_lines,
        retry_limit=retry_limit,
        request_cache=request_cache,
        cores=cores,
        in_memory=in_memory,
        verbose=verbose,
        progress_bar=progress_bar,
        make_request=make_request,
        text_as_paths=text_as_paths,
        dotenv=dotenv,
        cache=cache,
        cache_overwrite=cache_overwrite,
        cache_format=cache_format,
        key=key,
        secret=secret,
        url=url,
        version=version,
        endpoint=endpoint,
    )

    # finalize
    if res is None or not res.shape[0]:
        msg = "no results"
        raise RuntimeError(msg)
    if isinstance(cache, str):
        _update_cache(res, cache, cache_format, verbose, start_time, [e[0] for e in api_args] if api_args else [])
    if verbose:
        print(f"preparing output ({perf_counter() - start_time:.4f})")
    data.set_index("id", inplace=True)
    res.set_index("id", inplace=True)
    if len(res) != len(data):
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
            pandas.Index(exclude.intersection(res.columns)),
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
    if any(name not in exclude and name not in db.schema.names for name in res.columns.values):
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
