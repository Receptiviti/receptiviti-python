"""Check the status of the API."""

import os
import re
from typing import Union

import requests

from receptiviti.readin_env import readin_env


def status(
    url: str = os.getenv("RECEPTIVITI_URL", ""),
    key: str = os.getenv("RECEPTIVITI_KEY", ""),
    secret: str = os.getenv("RECEPTIVITI_SECRET", ""),
    dotenv: Union[bool, str] = False,
    verbose=True,
) -> Union[requests.Response, None]:
    """
    Check the API's status.

    Ping the Receptiviti API to see if it's available, and if your credentials are valid.

    Args:
      url (str): The URL of the API.
      key (str): Your API key.
      secret (str): Your API secret.
      dotenv (bool | str): Path to a .env file to read environment variables from, or `True`
        to look for a file in the current directory.
      verbose (bool): If `False`, will not print status messages.

    Returns:
      Response from the API server.

    Examples:
        >>> receptiviti.status()
    """
    _, url, key, secret = _resolve_request_def(url, key, secret, dotenv)
    try:
        res = requests.get(url.lower() + "/v1/ping", auth=(key, secret), timeout=9999)
    except requests.exceptions.RequestException:
        if verbose:
            print("Status: ERROR\nMessage: URL is unreachable")
        return None
    content = res.json() if res.text[:1] == "{" else {"message": res.text}
    if verbose:
        print("Status: " + ("OK" if res.status_code == 200 else "ERROR"))
        print(
            "Message: "
            + (
                str(res.status_code)
                + (" (" + str(content["code"]) + ")" if "code" in content else "")
                + ": "
                + content["pong" if "pong" in content else "message"]
            )
        )
    return res


def _resolve_request_def(url: str, key: str, secret: str, dotenv: Union[bool, str]):
    if dotenv:
        readin_env("." if isinstance(dotenv, bool) else dotenv)
    if not url:
        url = os.getenv("RECEPTIVITI_URL", "https://api.receptiviti.com")
    full_url = url
    url = ("https://" if re.match("http", url, re.I) is None else "") + re.sub("/+[Vv]\\d+(?:/.*)?$|/+$", "", url)
    if re.match("https?://[^.]+[.:][^.]", url, re.I) is None:
        raise TypeError("`url` does not appear to be valid: " + url)
    if not key:
        key = os.getenv("RECEPTIVITI_KEY", "")
        if not key:
            msg = "specify your key, or set it to the RECEPTIVITI_KEY environment variable"
            raise RuntimeError(msg)
    if not secret:
        secret = os.getenv("RECEPTIVITI_SECRET", "")
        if not secret:
            msg = "specify your secret, or set it to the RECEPTIVITI_SECRET environment variable"
            raise RuntimeError(secret)
    return (full_url, url, key, secret)
