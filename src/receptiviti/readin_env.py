"""Read in a environment variables."""

import os
import re


def readin_env(path=".", name=".env", overwrite=False) -> None:
    """
    Set environment variables from a .env file.

    Args:
      path (str): Path to a .env file, or to a directory containing such a file.
        By default, this will fall back on `~/Documents`.
      name (str): Name of the file, when `path` points to a directory.
        By default, this will fall back on `.Renviron`.
      overwrite (bool): If `True`, overwrites existing environment variables with
        the same name as those in the .env file.

    Returns:
      If a file is found, it will add contents to `os.environ`.
    """
    path = os.path.expanduser(path)
    envpath = path if os.path.isfile(path) else path + "/" + name
    if os.path.isfile(envpath):
        ql = re.compile("^['\"]|['\"\\s]+$")
        with open(envpath, encoding="utf-8") as file:
            for line in file:
                entry = line.split("=", 1)
                if len(entry) == 2 and (overwrite or os.getenv(entry[0]) is None):
                    os.environ[entry[0]] = ql.sub("", entry[1])
    elif name != ".Renviron":
        readin_env(path, ".Renviron", overwrite)
    elif path != os.path.expanduser("~/Documents"):
        readin_env("~/Documents", name, overwrite)
