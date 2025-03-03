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

This example uses the `Receptiviti` API to analyze commencement speeches.

## Data

We'll start by collecting and processing the speeches.

### Collection

The speeches used to be provided more directly, but the service hosting them has since shut down.

They are still available in a slightly less convenient form,
as the source of a site that displays them:
[whatrocks.github.io/commencement-db](https://whatrocks.github.io/commencement-db).

First, we can retrieve metadata from a separate repository:

```{code-cell}
import pandas

speeches = pandas.read_csv(
  "https://raw.githubusercontent.com/whatrocks/markov-commencement-speech"
  "/refs/heads/master/speech_metadata.csv"
)

speeches.iloc[0:5, 1:4]
```

One file in the source repository contains an invalid character on Windows (`:`),
so we'll need to pull them in individually, rather than cloning the repository:

```{code-cell}
import os
import requests

text_dir = "../../../commencement_speeches/"
os.makedirs(text_dir, exist_ok=True)

text_url = (
  "https://raw.githubusercontent.com/whatrocks/commencement-db"
  "/refs/heads/master/src/pages/"
)
for file in speeches["filename"]:
    out_file = text_dir + file.replace(':', '')
    if not os.path.isfile(out_file):
        req = requests.get(
            text_url + file.replace(".txt", "/index.md"), timeout=999
        )
        if req.status_code != 200:
            print(f"failed to retrieve {file}")
            continue
        with open(out_file, "w", encoding="utf-8") as content:
            content.write("\n".join(req.text.split("\n---")[1:]))
```

### Text Preparation

Now we can read in the texts and associate them with their metadata:

```{code-cell}
import re

bullet_pattern = re.compile("([a-z]) •")


def read_text(file: str):
  with open(text_dir + file.replace(":", ""), encoding="utf-8") as content:
      text = bullet_pattern.sub("\\1. ", content.read())
  return text


speeches["text"] = [read_text(file) for file in speeches["filename"]]
```

## Load Package

If this is your first time using the package, see the
[Get Started](https://receptiviti.github.io/receptiviti-r/articles/receptiviti.html)
guide to install it and set up your API credentials.

```{code-cell}
import receptiviti
```

## Analyze Full Texts

We might start by seeing if any speeches stand out in terms of language style,
or if there are any trends in content over time.

### Full: Process Text

Now we can send the texts to the API for scoring, and join the results we get to
the metadata:

```{code-cell}
# since our texts are from speeches,
# it might make sense to use the spoken norming context
processed = receptiviti.request(
    speeches["text"], version = "v2", context = "spoken"
)
processed = pandas.concat([speeches.iloc[:, 0:4], processed], axis=1)

processed.iloc[0:5, 7:]
```

### Full: Analyze Style

To get at stylistic uniqueness, we can calculate Language Style Matching
between each speech and the mean of all speeches:

```{code-cell}
lsm_categories = [
    "liwc15." + c
    for c in [
        "personal_pronouns",
        "impersonal_pronouns",
        "articles",
        "auxiliary_verbs",
        "adverbs",
        "prepositions",
        "conjunctions",
        "negations",
        "quantifiers",
    ]
]

category_means = processed[lsm_categories].agg("mean")
processed["lsm_mean"] = (
    1
    - abs(processed[lsm_categories] - category_means)
    / (processed[lsm_categories] + category_means)
).agg("mean", axis=1)

processed.sort_values("lsm_mean").iloc[0:10][
    ["name", "school", "year", "lsm_mean", "summary.word_count"]
]
```

Here, it is notable that the most stylistically unique speech was delivered in
American Sign Language, and the second most stylistically unique speech was
a short rhyme.

We might also want to see which speeches are most similar to one another:

```{code-cell}
import numpy

# calculate all pairwise comparisons
lsm_pairs = processed[lsm_categories].T.corr(
    lambda a, b: numpy.mean(1 - abs(a - b) / (a + b))
)

# set self-matches to 0
numpy.fill_diagonal(lsm_pairs.values, 0)

# identify the closes match to each speech
speeches["match"] = lsm_pairs.idxmax()
best_match = lsm_pairs.max()

# loo at the top matches
top_matches = best_match.sort_values(ascending=False).index[:20].astype(int).to_list()
top_match_pairs = pandas.DataFrame(
    {"a": top_matches, "b": speeches["match"][top_matches].to_list()}
)
top_match_pairs = top_match_pairs[
    ~top_match_pairs.apply(
        lambda r: "".join(r.sort_values().astype(str)), 1
    ).duplicated()
]
pandas.concat(
    [
        speeches.iloc[top_match_pairs["a"], 1:4].reset_index(drop=True),
        pandas.DataFrame({"Similarity": best_match[top_match_pairs["a"]]}).reset_index(
            drop=True
        ),
        speeches.iloc[top_match_pairs["b"], 1:4].reset_index(drop=True),
    ],
    axis=1,
)
```

### Full: Analyze Content

To look at content over time, we might focus on a potentially interesting framework, such as drives:

```{code-cell}
from statistics import linear_regression
from matplotlib.pyplot import subplots
from matplotlib.style import use

drive_data = processed.filter(regex="year|drives")[processed["year"] > 1980]
trending_drives = (
    drive_data.corrwith(drive_data["year"])
    .abs()
    .sort_values(ascending=False)[1:4]
    .index
)

first_year = drive_data["year"].min()
colors = ["#82C473", "#A378C0", "#616161", "#9F5C61", "#D3D280"]
linestyles = ["-", "--", ":", "-.", (5, (8, 2))]
use(["dark_background", {"figure.facecolor": "#1e2229", "axes.facecolor": "#1e2229"}])
fig, ax = subplots()
ax.set(ylabel="Score", xlabel="Year")
for i, cat in enumerate(trending_drives):
    points = ax.scatter(drive_data["year"], drive_data[cat], color=colors[i])
    beta, intercept = linear_regression(drive_data["year"], drive_data[cat])
    line = ax.axline(
        (first_year, intercept + beta * first_year),
        slope=beta,
        color=colors[i],
        linestyle=linestyles[i],
        label=cat,
    )
legend = ax.legend(loc="upper center")
```

To better visualize the effects, we might look between aggregated blocks of time:

```{code-cell}
summary = processed[trending_drives].aggregate(["mean", "std"])
standardized = ((processed[trending_drives] - summary.loc["mean"])) / summary.loc["std"]
time_median = int(processed["year"].median())
standardized["Time Period"] = pandas.Categorical(
    processed["year"] >= time_median
).set_categories([f"< {time_median}", f">= {time_median}"], rename=True)

summaries = standardized.groupby("Time Period", observed=True)[trending_drives].agg(
    ["mean", "size"]
)

fig, ax = subplots()
ax.set(ylabel="Score (Scaled)", xlabel="Time Period")
for i, cat in enumerate(trending_drives):
    summary = summaries[cat]
    ax.errorbar(
        summary.index,
        summary["mean"],
        yerr=1 / summary["size"] ** 0.5,
        color=colors[i],
        linestyle=linestyles[i],
        label=cat,
        capsize=6
    )
legend = ax.legend(loc="upper center")
```

This suggests that references to risk and reward have increased since the 2000s while references to power have decreased at a similar rate. (Note that error bars represent how much variance there is within groups, which allows you to eyeball the statistical significance of mean differences.)

The shift in emphasis from power to risk-reward could reflect that commencement speakers are now focusing more abstractly on the potential benefits and hazards of life after graduation, whereas earlier speakers more narrowly focused on ambition and dominance (perhaps referring to power held by past alumni and projecting the potential for graduates to climb social ladders in the future). You could examine a sample of speeches that show this pattern most dramatically (speeches high in risk-reward and low in power in recent years, and vice versa for pre-2009 speeches) to help determine how these themes have shifted and what specific motives or framing devices seem to have been (de)emphasized.

## Analyze Segments

Another thing we might look for is trends within each speech. For instance, are there common emotional trajectories over the course of a speech?

One way to look at this would be to split texts into roughly equal sizes, and score each section:

```{code-cell}
import nltk
from math import ceil

nltk.download("punkt_tab", quiet=True)


def count_words(text: str):
    return len([token for token in nltk.word_tokenize(text) if token.isalnum()])


def split_text(text: str, bins=3):
    sentences = nltk.sent_tokenize(text)
    bin_size = ceil(count_words(text) / bins) + 1
    text_parts = [[]]
    word_counts = [0] * bins
    current_bin = 0
    for sentence in sentences:
        sentence_size = count_words(sentence)
        if (current_bin + 1) < bins and (
            word_counts[current_bin] + sentence_size
        ) > bin_size:
            text_parts.append([])
            current_bin += 1
        word_counts[current_bin] += sentence_size
        text_parts[current_bin].append(sentence)
    return pandas.DataFrame(
        {
            "text": [" ".join(x) for x in text_parts],
            "segment": pandas.Series(range(bins)) + 1,
            "WC": word_counts,
        }
    )


segmented_text = []
for i, text in enumerate(speeches["text"]):
    text_parts = split_text(text)
    text_info = speeches.iloc[[i] * 3][["name", "school", "year"]]
    text_info.reset_index(drop=True, inplace=True)
    segmented_text.append(pandas.concat([text_info, text_parts], axis=1))
segmented_text = pandas.concat(segmented_text)
segmented_text.reset_index(drop=True, inplace=True)

segmented_text.iloc[0:9, :6]
```

### Segments: Process Text

Now we can send each segment to the API to be scored:

```{code-cell}
processed_segments = receptiviti.request(
    segmented_text["text"], version="v2", context="spoken"
)

processed_segments = receptiviti.request(
    segmented_text["text"], version="v2", context="spoken"
)
segmented_text = pandas.concat([segmented_text, processed_segments], axis=1)

segmented_text.iloc[0:9, 8:]
```

### Segments: Analyze Scores

The [SALLEE framework](https://docs.receptiviti.com/frameworks/emotions) offers measures of emotions, so we might see which categories deviate the most in any of their segments:

```{code-cell}
# select the narrower SALLEE categories
emotions = segmented_text.filter(regex="^sallee").iloc[:, 6:]

# correlate emotion scores with segment contrasts
# and select the 5 most deviating emotions
most_deviating = emotions[
    pandas.get_dummies(segmented_text["segment"])
    .apply(emotions.corrwith)
    .abs()
    .agg("max", 1)
    .sort_values(ascending=False)[:5]
    .index
]
```

Now we can look at those categories across segments:

```{code-cell}
from matplotlib.colors import ListedColormap

segment_data = most_deviating.groupby(segmented_text["segment"])

bars = segment_data.agg("mean").T.plot.bar(
    colormap=ListedColormap(colors[:3]),
    yerr=(segment_data.agg("std") / segment_data.agg("count") ** 0.5).values,
    capsize=3,
    ylabel="Score",
    xlabel="Category"
)
```

The bar chart displays original values, which offers the clearest view of how meaningful the differences between segments might be, in addition to their statistical significance (which offers a rough guide to the reliability of differences, based on the variance within and between segments). By looking at the bar graph, you can immediately see that admiration shows some of the starkest differences between middle and early/late segments.

```{code-cell}
scaled_summaries = (
    (most_deviating - most_deviating.agg("mean")) / most_deviating.agg("std")
).groupby(segmented_text["segment"]).agg(
    ["mean", "size"]
)

fig, ax = subplots()
ax.set(ylabel="Score (Scaled)", xlabel="Segment")
for i, cat in enumerate(most_deviating):
    summary = scaled_summaries[cat]
    ax.errorbar(
        summary.index.astype(str),
        summary["mean"],
        yerr=1 / summary["size"] ** 0.5,
        color=colors[i],
        linestyle=linestyles[i],
        label=cat,
        capsize=4,
    )
legend = fig.legend(loc="right", bbox_to_anchor=(1.2, .5))
```

The line charts, on the other hand, shows standardized values, effectively zooming in on the differences between segments. This more clearly shows, for example, that admiration and joy seem to be used as bookends in commencement speeches, peaking early and late, whereas more negative and intense emotions such as anger, disgust, and surprise peak in the middle section.
