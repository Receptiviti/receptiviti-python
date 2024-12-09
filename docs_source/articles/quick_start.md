---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: 'articles'
  language: python
  name: 'articles'
---

```{code-cell}
:tags: [hide_cell]

import os
os.environ["RECEPTIVITI_PB"]="False"
import receptiviti
receptiviti.readin_env()
os.environ["RECEPTIVITI_URL"]=os.environ["RECEPTIVITI_URL_TEST"]
os.environ["RECEPTIVITI_KEY"]=os.environ["RECEPTIVITI_KEY_TEST"]
os.environ["RECEPTIVITI_SECRET"]=os.environ["RECEPTIVITI_SECRET_TEST"]
```

## Install and Load

First, download and install Python from <a href="https://python.org/downloads" rel="noreferrer" target="_blank">python.org</a>.

Then, install the package:

```sh
pip install git+https://github.com/receptiviti/receptiviti-python.git
```

Each time you start a Python session, load the package:

```{code-cell}
import receptiviti
```

## Set Up API Credentials

You can find your API key and secret on your <a href="https://dashboard.receptiviti.com" rel="noreferrer" target="_blank">dashboard</a>.

You can set these credentials up in Python permanently or temporarily:

### Permanent

Open or create a `~/.env` file, Then add these environment variables with your key and secret:

```sh
RECEPTIVITI_KEY=""
RECEPTIVITI_SECRET=""
```

These can be read in with the `receptiviti.readin_env()` function, which is automatically called if credentials are not otherwise provided (and the `dotenv` argument is `True`).

### Temporary

Add your key and secret, and run at the start of each session:

```py
import os
os.environ["RECEPTIVITI_KEY"]="32lettersandnumbers"
os.environ["RECEPTIVITI_SECRET"]="56LettersAndNumbers"
```

### Confirm Credentials

Check that the API is reachable, and your credentials are recognized:

```{code-cell}
receptiviti.status()
```

If your credentials are not recognized, you'll get a response like this:

```{code-cell}
receptiviti.status(key=123, secret=123)
```

## Enter Your Text

### Loaded Text

If your texts are already in Python, you can enter them directly.

These can be in a single character:

```{code-cell}
results = receptiviti.request("texts to score")
```

Or a character vector:

```{code-cell}
results = receptiviti.request(["text one", "text two"])
```

Or from a `DataFrame`:

```{code-cell}
import pandas
data = pandas.DataFrame({"text": ["text a", "text b"]})

# directly
results = receptiviti.request(data["text"])

# by column name
results = receptiviti.request(data, text_column="text")
```

### Text in files

You can enter paths to files containing separate texts in each line:

```{code-cell}
# single
results = receptiviti.request("files/file.txt")

# multiple
results = receptiviti.request(
  files = ["files/file1.txt", "files/file2.txt"]
)
```

Or to a comma delimited file with a column containing text.
Here, the `text_column` argument specifies which column contains text:

```{code-cell}
# single
results = receptiviti.request("files/file.csv", text_column="text")

# multiple
results = receptiviti.request(
  files = ["files/file1.csv", "files/file2.csv"],
  text_column="text"
)
```

Or you can point to a directory containing text files:

```{code-cell}
results = receptiviti.request(directory = "files")
```

By default `.txt` files will be looked for, but you can specify
`.csv` files with the `file_type` argument:

```{code-cell}
results = receptiviti.request(
  directory = "files",
  text_column="text", file_type="csv"
)
```

## Use Results

### Returned Results

Results are returned as a `DataFrame`, with a row for each
text, and columns for each framework variable:

```{code-cell}
results = receptiviti.request("texts to score")
results.iloc[:, :3]
```

Here, the first column (`text_hash`) is the MD5 hash of the text,
which identifies unique texts, and is stored in the main cache.

The entered text can also be included with the `return_text` argument:

```{code-cell}
results = receptiviti.request("texts to score", return_text=True)
results[["text_hash", "text"]]
```

You can also select frameworks before they are all returned:

```{code-cell}
results = receptiviti.request("texts to score", frameworks="liwc")
results.iloc[:, :5]
```

By default, a single framework will have column names without the framework name,
but you can retain these with `framework_prefix=True`:

```{code-cell}
results = receptiviti.request(
  "texts to score",
  frameworks="liwc", framework_prefix=True
)
results.iloc[:, :4]
```

### Aligning Results

Results are returned in a way that aligns with the text you enter originally,
including any duplicates or invalid entries.

This means you can add the results object to original data:

```{code-cell}
data = pandas.DataFrame({
  "id": [1, 2, 3, 4],
  "text": ["text a", float("nan"), "", "text a"]
})
results = receptiviti.request(data["text"])

# combine data and results
data.join(results).iloc[:, :5]
```

You can also provide a vector of unique IDs to be returned with results so they can be merged with
other data:

```{code-cell}
results = receptiviti.request(["text a", "text b"], ids=["a", "b"])
results.iloc[:, :4]
```

```{code-cell}
# merge with a new dataset
data = pandas.DataFrame({
  "id": ["a1", "b1", "a2", "b2"],
  "type": ["a", "b", "a", "b"]
})
data.join(results.set_index("id"), "type").iloc[:, :5]
```

### Saved Results

Results can also be saved to a `.csv` file:

```{code-cell}
receptiviti.request("texts to score", "~/Documents/results.csv", overwrite=True)
results = pandas.read_csv("~/Documents/results.csv")
results.iloc[:, :4]
```

## Preserving Results

The `receptiviti.request` function tries to avoid sending texts to the API as much as possible:

- As part of the preparation process, it excludes duplicates and invalid texts.
- If enabled, it checks the primary cache to see if any texts have already been scored.
  - The primary cache is an Arrow database located by the `cache` augment.
  - Its format is determined by `cache_format`.
  - You can skip checking it initially while still writing results to it with `cache_overwrite=True`.
  - It can be cleared with `clear_cache=True`.
- It will check for any responses to previous, identical requests.
  - Responses are stored in the `receptiviti_request_cache` directory of your system's temporary directory (`tempfile.gettempdir()`).
  - You can avoid using this cache with `request_cache=False`.
  - This cache is cleared after a day.

If you want to make sure no texts are sent to the API, you can use `make_request=False`.
This will use the primary and request cache, but will fail if any texts are not found there.

If a call fails before results can be written to the cache or returned, all received responses will
still be in the request cache, but those will be deleted after a day.

## Handling Big Data

The `receptiviti.request` function will handle splitting texts into bundles, so the limit on how many texts
you can process at once will come down to your system's amount of random access memory (RAM).
Several thousands of texts should be fine, but getting into millions of texts, you may not be able
to have all of the results loaded at once. To get around this, you can fully process subsets
of your texts.

A benefit of processing more texts at once is that requests can be parallelized, but this
is more RAM intensive, and the primary cache is updated less frequently (as it is updated
only at the end of a complete run).

You could also parallelize your own batches, but be sure to set `cores` to `1` (to disable
the function's parallelization) and do not enable the primary cache (to avoid attempting to
read from the cache while it is being written to by another instance).

Not using the primary cache is also more efficient, but you may want to ensure you are not
sending duplicate texts between calls. The function handles duplicate texts within calls (only
ever sending unique texts), but depends on the cache to avoid sending duplicates between calls.
