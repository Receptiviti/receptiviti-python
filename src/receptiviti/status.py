import os
import re
import requests
from .readin_env import readin_env


def status(
    url: str = os.getenv("RECEPTIVITI_URL", ""),
    key: str = os.getenv("RECEPTIVITI_KEY", ""),
    secret: str = os.getenv("RECEPTIVITI_SECRET", ""),
    dotenv: bool | str = False,
    verbose=True,
) -> requests.Response:
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
      requests.Response: Response from the API server.

    Examples:
        >>> receptiviti.status()
    """
    if dotenv is not None and dotenv:
        readin_env("." if isinstance(dotenv, bool) else dotenv)
    if url == "":
        url = os.getenv("RECEPTIVITI_URL", "https://api.receptiviti.com")
    if key == "":
        key = os.getenv("RECEPTIVITI_KEY", "")
    if secret == "":
        secret = os.getenv("RECEPTIVITI_SECRET", "")
    url = ("https://" if re.match("http", url, re.I) is None else "") + re.sub(
        "/[Vv]\\d(?:/.*)?$|/+$", "", url
    )
    if re.match("https?://[^.]+[.:][^.]", url, re.I) is None:
        raise TypeError("`url` does not appear to be valid: " + url)
    res = requests.get(url + "/v1/ping", auth=(key, secret), timeout=9999)
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
