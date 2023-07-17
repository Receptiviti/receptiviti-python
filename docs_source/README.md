# Home

A Python package to process text with the [Receptiviti](https://www.receptiviti.com) API.

## Installation

If needed, download Python from [python.org](https://www.python.org/downloads), then install the package with pip:

```sh
pip install git+https://github.com/miserman/receptiviti-py.git
```

And load the package in a python console:

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
file_results <- receptiviti.request("./path/to/txt_folder")

## could be .csv
file_results <- receptiviti.request(
  "./path/to/csv_folder",
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

You can store these in a `.env` (in the current directory or `~/Documents`) file permanently:

```py
# use the `dotenv` argument
receptiviti.status(dotenv=True)

# or load variables in beforehand
receptiviti.readin_env()
```

Or set them temporarily:

```py
import os
os.environ["RECEPTIVITI_KEY"]="32lettersandnumbers"
os.environ["RECEPTIVITI_SECRET"]="56LettersAndNumbers"
```
