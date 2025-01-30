"""Interact with the norming endpoint."""

import json
import os
import re
import warnings
from typing import Dict, List, Union

import pandas
import requests

from receptiviti.manage_request import _manage_request, _resolve_request_def


def norming(
    name: Union[str, None] = None,
    text: Union[str, List[str], pandas.DataFrame, None] = None,
    options: Union[dict, None] = None,
    delete=False,
    name_only=False,
    dotenv: Union[bool, str] = True,
    key=os.getenv("RECEPTIVITI_KEY", ""),
    secret=os.getenv("RECEPTIVITI_SECRET", ""),
    url=os.getenv("RECEPTIVITI_URL", ""),
    verbose=True,
    **kwargs,
) -> Union[None, List[str], pandas.DataFrame, pandas.Series, Dict[str, Union[pandas.Series, pandas.DataFrame, None]]]:
    """
    View or Establish Custom Norming Contexts.

    Custom norming contexts can be used to process later texts by specifying the
    `custom_context` API argument in the `receptiviti.request` function (e.g.,
    `receptiviti.request("text to score", version="v2", custom_context="norm_name")`,
    where `norm_name` is the name you set here).

    Args:
        name (str): Name of a new norming context, to be established from the provided 'text'.
            Not providing a name will list the previously created contexts.
        text (str): Text to be processed and used as the custom norming context.
            Not providing text will return the status of the named norming context.
        options (dict): Options to set for the norming context (e.g.,
            `{"min_word_count": 350, "max_punctuation": .25}`).
        delete (bool): If `True`, will request removal of the `name` context.
        name_only (bool): If `True`, will return a list of context names only, including those of
            build-in contexts.
        dotenv (bool | str): Path to a .env file to read environment variables from. By default,
            will for a file in the current directory or `~/Documents`.
            Passed to `readin_env` as `path`.
        key (str): Your API key.
        secret (str): Your API secret.
        url (str): The URL of the API; defaults to `https://api.receptiviti.com`.
        verbose (bool): If `False`, will not show status messages.
        **kwargs (Any): Additional arguments to specify how texts are read in and processed;
            see [receptiviti.request][receptiviti.request].

    Returns:
        Nothing if `delete` is `True`.
            If `name_only` is `True`, a `list` containing context names (built-in and custom).
            Otherwise, either a `pandas.DataFrame` containing all existing custom context statuses
            (if no `name` is specified), a `pandas.Series` containing the the status of
            `name` (if `text` is not specified), a dictionary:

            - `initial_status`: Initial status of the context.
            - `first_pass`: Response after texts are sent the first time, or
              `None` if the initial status is `pass_two`.
            - `second_pass`: Response after texts are sent the second time.

    Examples:
        ```python
        # list all available contexts:
        receptiviti.norming()

        # create or get the status of a single context:
        receptiviti.norming("new_context")
        ```

        Send texts to establish the context, just like
        the [receptiviti.request][receptiviti.request] function.
        ```python
        ## such as directly:
        receptiviti.norming("new_context", ["text to send", "another text"])

        ## or from a file:
        receptiviti.norming("new_context", "./path/to/file.csv", text_column = "text")

        ## delete the new context:
        receptiviti.norming("new_context", delete=True)
        ```
    """
    _, url, key, secret = _resolve_request_def(url, key, secret, dotenv)
    auth = requests.auth.HTTPBasicAuth(key, secret)
    if name_only:
        if verbose:
            print("requesting list of existing custom norming contests")
        req = requests.get(url + "/v2/norming/", auth=auth, timeout=9999)
        if req.status_code != 200:
            msg = f"failed to make norming list request: {req.status_code} {req.reason}"
            raise RuntimeError(msg)
        norms = req.json()
        if norms and verbose:
            custom_prefix = re.compile("^custom/")
            print("available norming context(s): " + ", ".join([custom_prefix.sub("", name) for name in norms]))
        return norms

    url += "/v2/norming/custom/"
    if name and re.search("[^a-z0-9_.-]", name):
        msg = "`name` can only include lowercase letters, numbers, hyphens, underscores, or periods"
        raise RuntimeError(msg)

    # list current context
    if verbose:
        print("requesting list of existing custom norming contests")
    req = requests.get(url, auth=auth, timeout=9999)
    if req.status_code != 200:
        msg = f"failed to make custom norming list request: {req.status_code} {req.reason}"
        raise RuntimeError(msg)
    norms = pandas.json_normalize(req.json())
    if not name:
        if len(norms):
            if verbose:
                custom_prefix = re.compile("^custom/")
                print(
                    "custom norming context(s) found: "
                    + ", ".join([custom_prefix.sub("", name) for name in norms["name"]])
                )
        elif verbose:
            print("no custom norming contexts found")
        return norms
    context_id = "custom/" + name
    if len(norms) and context_id in norms["name"].values:
        if delete:
            res = requests.delete(url + name, auth=auth, timeout=9999)
            content = res.json() if res.text[:1] == "[" else {"message": res.text}
            if res.status_code != 200:
                msg = f"Request Error ({res.status_code!s})" + (
                    (" (" + str(content["code"]) + ")" if "code" in content else "") + ": " + content["message"]
                )
                raise RuntimeError(msg)
            return None
        status = norms[norms["name"] == context_id].iloc[0]
        if options:
            warnings.warn(UserWarning(f"context {name} already exists, so options do not apply"), stacklevel=2)
    elif delete:
        print(f"context {name} does not exist")
        return None
    else:
        if verbose:
            print(f"requesting creation of context {name}")
        req = requests.post(url, json.dumps({"name": name, **(options if options else {})}), auth=auth, timeout=9999)
        if req.status_code != 200:
            msg = f"failed to make norming creation request: {req.json().get('error', 'reason unknown')}"
            raise RuntimeError(msg)
        status = pandas.json_normalize(req.json()).iloc[0]
        if options:
            for param, value in options.items():
                if param in status and value != status[param]:
                    warnings.warn(UserWarning(f"set option {param} does not match the requested value"), stacklevel=2)
    if verbose:
        print(f"status of {name}:")
        print(status)
    if not text:
        return status
    status_step = status["status"]
    if status_step == "completed":
        warnings.warn(UserWarning("status is `completes`, so cannot send text"), stacklevel=2)
        return {"initial_status": status, "first_pass": None, "second_pass": None}
    if status_step == "pass_two":
        first_pass = None
    else:
        if verbose:
            print(f"sending first-pass sample for {name}")
        _, first_pass, _ = _manage_request(
            text=text,
            **kwargs,
            dotenv=dotenv,
            key=key,
            secret=secret,
            url=f"{url}{name}/one",
            to_norming=True,
        )
    second_pass = None
    if first_pass is not None and (first_pass["analyzed_samples"] == 0).all():
        warnings.warn(
            UserWarning("no texts were successfully analyzed in the first pass, so second pass was skipped"),
            stacklevel=2,
        )
    else:
        if verbose:
            print(f"sending second-pass samples for {name}")
        _, second_pass, _ = _manage_request(
            text=text,
            **kwargs,
            dotenv=dotenv,
            key=key,
            secret=secret,
            url=f"{url}{name}/two",
            to_norming=True,
        )
    if second_pass is None or (second_pass["analyzed_samples"] == 0).all():
        warnings.warn(UserWarning("no texts were successfully analyzed in the second pass"), stacklevel=2)
    return {"initial_stats": status, "first_pass": first_pass, "second_pass": second_pass}
