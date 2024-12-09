"""Check the status of the API."""

import os
from typing import List, Union

import requests

from receptiviti.status import _resolve_request_def


def frameworks(
    url: str = os.getenv("RECEPTIVITI_URL", ""),
    key: str = os.getenv("RECEPTIVITI_KEY", ""),
    secret: str = os.getenv("RECEPTIVITI_SECRET", ""),
    dotenv: Union[bool, str] = False,
) -> List[str]:
    """
    List Available Frameworks.

    Retrieve a list of all frameworks available to your account.

    Args:
      url (str): The URL of the API.
      key (str): Your API key.
      secret (str): Your API secret.
      dotenv (bool | str): Path to a .env file to read environment variables from, or `True`
        to look for a file in the current directory.

    Returns:
      List of framework names.

    Examples:
        ```python
        receptiviti.frameworks()
        ```
    """
    _, url, key, secret = _resolve_request_def(url, key, secret, dotenv)
    res = requests.get(url.lower() + "/v2/frameworks", auth=(key, secret), timeout=9999)
    content = res.json() if res.text[:1] == "[" else {"message": res.text}
    if res.status_code != 200:
        msg = f"Request Error ({res.status_code!s})" + (
            (" (" + str(content["code"]) + ")" if "code" in content else "") + ": " + content["message"]
        )
        raise RuntimeError(msg)
    return content if isinstance(content, list) else []
