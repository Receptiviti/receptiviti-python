A Python package to process text with the [Receptiviti](https://www.receptiviti.com) API.

An R package is also available at [Receptiviti/receptiviti-r](https://receptiviti.github.io/receptiviti-r/).

## Installation

If needed, download Python from [python.org](https://www.python.org/downloads), then install the package with pip:

Release ([version 0.1.2](https://pypi.org/project/receptiviti/0.1.2))

```sh
pip install receptiviti
```

Development

```sh
pip install git+https://github.com/receptiviti/receptiviti-python.git
```

And load the package in a Python console:

```py
import receptiviti
```

## Examples

```py
# score a single text
single = receptiviti.request("a text to score")

# score multiple texts, and write results to a file
multi = receptiviti.request(["first text to score", "second text"], "filename.csv")

# score texts in separate files
## defaults to look for .txt files
file_results = receptiviti.request(directory = "./path/to/txt_folder")

## could be .csv
file_results = receptiviti.request(
  directory = "./path/to/csv_folder",
  text_column = "text", file_type = "csv"
)

# score texts in a single file
results = receptiviti.request("./path/to/file.csv", text_column = "text")
```

## API Access

To access the API, you will need to load your key and secret, as found on your [dashboard](https://dashboard.receptiviti.com).

You can enter these as arguments in each function call, but by default they will be looked for in these environment variables:

```
RECEPTIVITI_KEY="32lettersandnumbers"
RECEPTIVITI_SECRET="56LettersAndNumbers"
```

You can store these in a `.env` file (in the current directory or `~/Documents`) permanently, or set them temporarily:

```py
import os
os.environ["RECEPTIVITI_KEY"]="32lettersandnumbers"
os.environ["RECEPTIVITI_SECRET"]="56LettersAndNumbers"
```

## Request Process

The `request` function handles texts and results in several steps:

1. Prepare bundles (split `text` into <= `bundle_size` and <= `bundle_byte_limit` bundles).
   1. If `text` points to a directory or list of files, these will be read in later.
   2. If `in_memory` is `False`, bundles are written to a temporary location, and read back in when the request is made.
2. Get scores for texts within each bundle.
   1. If texts are paths, or `in_memory` is `False`, will load texts.
   2. If `cache` is set, will skip any texts with cached scores.
   3. If `request_cache` is `True`, will check for a cached request.
   4. If any texts need scoring and `make_request` is `True`, will send unscored texts to the API.
3. If a request was made and `request_cache` is set, will cache the response.
4. If `cache` is set, will write bundle scores to the cache.
5. After requests are made, if `cache` is set, will defragment the cache (combine bundle results within partitions).
6. If `collect_results` is `True`, will prepare results:
   1. Will realign results with `text` (and `id` if provided).
   2. If `output` is specified, will write realigned results to it.
   3. Will drop additional columns (such as `custom` and `id` if not provided).
   4. If `framework` is specified, will use it to select columns of the results.
   5. Returns results.
