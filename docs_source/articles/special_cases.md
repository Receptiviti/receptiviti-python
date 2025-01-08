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
  display_name: '-'
  language: python
  name: '-'
---

```{code-cell}
:tags: [hide_cell]

import os
os.environ["RECEPTIVITI_PB"]="False"

import receptiviti

receptiviti.norming("custom_example", delete=True)
from shutil import rmtree
rmtree("../test_text_results", True)
```

# Special Cases

## Norming Contexts

Some measures are <a href="https://docs.receptiviti.com/the-receptiviti-api/normed-vs-dictionary-counted-measures" rel="noreferrer" target="_blank">normed</a> against a sample of text.
These samples may be more or less appropriate to your texts.

### Built-In

The default context is meant for general written text, and there is another built-in context for general spoken text:

```{code-cell}
import receptiviti
import pandas

text = "Text to be normed differently."
written = receptiviti.request(text, version="v2")
spoken = receptiviti.request(text, version="v2", context="spoken")

# select a few categories that differ between contexts
differing = written.columns[(written != spoken).iloc[0]][0:10]

# note that the text hashes are sensitive to the set context
pandas.concat([written, spoken], ignore_index=True)[differing].rename(
    {0: "written", 1: "spoken"}
).T
```

### Custom

You can also norm against your own sample, which involves first establishing a context,
then scoring against it.

Use the `receptiviti.norming` function to establish a custom context:

```{code-cell}
context_text = ["Text with normed in it.", "Text with differently in it."]

# set lower word count filter for this toy sample
context_status = receptiviti.norming(
    name="custom_example",
    text=context_text,
    options={"word_count_filter": 1},
    verbose=False,
)

# the `second_pass` result shows what was analyzed
context_status["second_pass"]
```

Then use the `custom_context` argument to specify that norming context when scoring:

```{code-cell}
custom = receptiviti.request(text, version="v2", custom_context="custom_example")

custom[differing].rename({0: "custom"}).T
```

## High Volume

The Receptiviti API has <a href="https://docs.receptiviti.com/api-reference/framework-bulk" rel="noreferrer" target="_blank">limits</a>
on bundle requests, so the `receptiviti.request()` function splits texts into acceptable bundles, to be spread across multiple requests.

This means the only remaining limitation on the number of texts that can be processed comes
from the memory of the system sending requests.

The basic way to work around this limitation is to fully process smaller chunks of text.

There are a few ways to avoid loading all texts and results.

### Cache as Output

Setting the `collect_results` argument to `False` avoids retaining all batch results in memory as they are receive, but means
results are not returned, so the they have to be collected in the cache.

If texts are also too big to load into memory, they can be loaded from files at request time.
By default, when multiple files pointed to as `text`, the actual texts are only loaded when they are being sent for scoring,
which means only `bundle_size` \* `cores` texts are loaded at a time.

We can start by writing some small text examples to files:

```{code-cell}
from os import makedirs

base_dir = "../../../"
text_dir = base_dir + "test_texts"
makedirs(text_dir, exist_ok=True)

for i in range(10):
    with open(f"{text_dir}/example_{i}.txt", "w", encoding="utf-8") as file:
        file.write(f"An example text {i}.")
```

And then minimally load these and their results by saving results to a Parquet dataset.

Disabling the `request_cache` will also avoid storing a copy of raw results.

```{code-cell}
import receptiviti

db_dir = base_dir + "test_results"
makedirs(db_dir, exist_ok=True)

receptiviti.request(
  directory=text_dir, collect_results=False, cache=db_dir, request_cache=False
)
```

Results are now available in the cache directory, which you can load in using the request function again:

```{code-cell}
# adding make_request=False just ensures requests are not made if not found
results = receptiviti.request(directory=text_dir, cache=db_dir, make_request=False)
results.iloc[:, 0:3]
```

### Manual Chunking

A more flexible approach would be to process smaller chunks of text normally, and handle loading and storing results yourself.

In this case, it may be best to disable parallelization, and explicitly disable the primary cache
(in case it's specified in an environment variable).

```{code-cell}
res_dir = base_dir + "text_results_manual"
makedirs(res_dir, exist_ok=True)

# using the same files as before
files = [f"{text_dir}/{file}" for file in os.listdir(text_dir)]

# process 5 files at a time
for i in range(0, len(files), 5):
  file_subset = files[i : i + 5]
  results = receptiviti.request(
    files=file_subset, ids=file_subset, cores=1, cache=False, request_cache=False
  )
  results.to_csv(f"{res_dir}/files_{i}-{i + 5}.csv.xz", index=False)
```

Now results will be stored in smaller files:

```{code-cell}
from pandas import read_csv

read_csv(f"{res_dir}/files_0-5.csv.xz").iloc[:, 0:3]
```
