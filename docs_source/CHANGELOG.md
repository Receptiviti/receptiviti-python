# Changelog

## receptiviti 0.1.3

### Additions

- Adds support for V2 of the API.
- Adds custom norming context creation functionality.

### Improvements

- Validates `version` and `endpoint`.

## Bug Fixes

- Avoids overwriting existing cache results within overlapping bins on update.

## [receptiviti 0.1.2](https://pypi.org/project/receptiviti/0.1.2)

### Improvements

- Changes default number of cores to 1, to avoid unexpected behavior when running from a script.
- Improves environment file resolution.

### Bug Fixes

- Corrects order of output when reading from a file and `ids` are not specified.
- Fixes detection of some file encodings.
- Avoids issues when `receptiviti.request` is called from a script and is processing in parallel.

## [receptiviti 0.1.1](https://pypi.org/project/receptiviti/0.1.1)

### Improvements

- Adds `encoding` argument; improves handling of non-UTF-8 files.

### Bug Fixes

- Fixes reading in files when `collapse_line` is `True`.

## [receptiviti 0.1.0](https://pypi.org/project/receptiviti/0.1.0)

First release.
