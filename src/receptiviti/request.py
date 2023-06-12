import os
from time import perf_counter, sleep
import sys
import re
import hashlib
import requests
import numpy
import pandas
from multiprocessing import Pool, cpu_count
from .status import status
from .readin_env import readin_env


def process(bundle: pandas.DataFrame, ops: dict) -> pandas.DataFrame | None:
    body = [
        {"content": text, "request_id": hashlib.md5(text.encode()).hexdigest(), **ops["add"]}
        for text in bundle["text"]
    ]
    res = requests.post(ops["url"], auth=ops["auth"], json=body, timeout=9999)
    content = None
    if res.status_code == 200:
        content = pandas.DataFrame.from_dict(pandas.json_normalize(res.json()["results"]))
    elif ops["retries"] > 0:
        sleep(1)
        ops["retries"] -= 1
        print(res.text)
        process(bundle, ops)
    return content


def request(
    text: str | list | pandas.DataFrame,
    output: str | None = None,
    id: str | list | None = None,
    text_column: str | None = None,
    id_column: str | None = None,
    api_args: dict = {},
    bundle_size=1000,
    bundle_byte_limit=75e5,
    retry_limit=50,
    cores=cpu_count() - 2,
    verbose=True,
    overwrite=False,
    dotenv: bool | str = True,
    key=os.getenv("RECEPTIVITI_KEY", ""),
    secret=os.getenv("RECEPTIVITI_SECRET", ""),
    url=os.getenv("RECEPTIVITI_URL", ""),
) -> pandas.DataFrame:
    """
    Send texts to be scored by the API.

    Args:
      text (str | list | pandas.DataFrame): Text to be processed.
      output (str | None): Path to a file to write results to.
      id (str | list): Vector of IDs for each `text`, or a column name in `text` containing IDs.
      text_column (str | None): Column name in `text` containing text.
      id_column (str | None): Column name in `text` containing ids.
      api_args (dict): Additional arguments to include in the request.
      bundle_size (int): Maximum number of texts per bundle.
      bundle_byte_limit (float): Maximum byte size of each bundle.
      retry_limit (int): Number of times to retry a failed request.
      cores (int): Number of CPU cores to use.
      verbose (bool): If `False`, will not print status messages.
      overwrite (bool): If `True`, will overwrite an existing `output` file.
      dotenv (bool | str): Path to a .env file to read environment variables from. By default,
        will for a file in the current directory or `~/Documents`. Passed to `readin_env` as `path`.
      key (str): Your API key.
      secret (str): Your API secret.
      url (str): The URL of the API.

    Returns:
      pandas.DataFrame: results
    """
    if output is not None and os.path.isfile(output) and not overwrite:
        raise RuntimeError("`output` file already exists; use `overwrite=True` to overwrite it")
    start_time = perf_counter()

    # resolve credentials and check status
    if dotenv:
        readin_env("." if isinstance(dotenv, bool) else dotenv)
    if url == "":
        url = os.getenv("RECEPTIVITI_URL", "https://api.receptiviti.com")
    url = ("https://" if re.match("http", url, re.I) is None else "") + re.sub(
        "/[Vv]\\d(?:/.*)?$|/+$", "", url
    )
    if key == "":
        key = os.getenv("RECEPTIVITI_KEY", "")
    if secret == "":
        secret = os.getenv("RECEPTIVITI_SECRET", "")
    api_status = status(url, key, secret, False)
    if api_status.status_code != 200:
        raise RuntimeError(f"API status failed: {api_status.status_code}")

    # resolve text and id
    if isinstance(text, str) and os.path.isfile(text):
        if verbose:
            print(f"reading in texts from a file ({perf_counter() - start_time:.4f})")
        text = pandas.read_csv(text)
    if isinstance(text, pandas.DataFrame):
        if id_column is not None:
            if id_column in text:
                id = text[id_column].to_list()
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
    n_texts = len(text)
    if id is None:
        id = numpy.arange(1, n_texts + 1)
    elif len(id) != n_texts:
        raise RuntimeError("`id` is not the same length as `text`")

    # prepare bundles
    if verbose:
        print(f"preparing text ({perf_counter() - start_time:.4f})")
    data = pandas.DataFrame({"text": text, "id": id})
    data = data[(~data.duplicated(subset=["text"])) | (data["text"] == "") | (data["text"].isna())]
    if not len(data):
        raise RuntimeError("no valid texts to process")
    n_bundles = n_texts / min(1000, max(1, bundle_size))
    groups = data.groupby(
        numpy.tile(numpy.arange(n_bundles + 1), n_texts)[:n_texts], group_keys=False
    )
    bundles = []
    for _, group in groups:
        if sys.getsizeof(group) > bundle_byte_limit:
            start = current = end = 0
            for txt in group["text"]:
                size = sys.getsizeof(txt)
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
    if verbose:
        print(
            f"prepared text in {len(bundles)} {'bundles' if len(bundles) > 1 else 'bundle'}",
            f"({perf_counter() - start_time:.4f})",
        )

    # process bundles
    args = {
        "url": url + "/v1/framework/bulk",
        "auth": (key, secret),
        "retries": retry_limit,
        "add": api_args,
    }
    if cores > 1:
        with Pool(cores) as p:
            res = p.starmap_async(process, [(b, args) for b in bundles]).get()
    else:
        res = [process(b, args) for b in bundles]
    res = pandas.concat(res, ignore_index=True, sort=False)

    # finalize
    if output is not None:
        res.to_csv(output, index=False)
    if verbose:
        print(f"done ({perf_counter() - start_time:.4f})")

    return res
