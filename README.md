A Python package to process text with the [Receptiviti](https://www.receptiviti.com) API.

An R package is also available at [Receptiviti/receptiviti-r](https://receptiviti.github.io/receptiviti-r/).

## Installation

If needed, download Python from [python.org](https://www.python.org/downloads), then install the package with pip:

Release ([version 0.1.1](https://pypi.org/project/receptiviti/0.1.1))

```sh
pip install receptiviti
```

Development (version 0.1.2)

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
