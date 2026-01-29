# CHANGELOG


## v0.1.2 (2026-01-29)

### Bug Fixes

- Use filename-based GitHub Actions badge URL
  ([#2](https://github.com/OpenAdaptAI/openadapt-retrieval/pull/2),
  [`7b15cfe`](https://github.com/OpenAdaptAI/openadapt-retrieval/commit/7b15cfefa0c6e993b54910646973ae079fadf29b))

The workflow-name-based badge URL was showing "no status" because GitHub requires workflow runs on
  the specified branch. Using the filename-based URL format
  (actions/workflows/publish.yml/badge.svg) is more reliable and works regardless of when the
  workflow last ran.

Co-authored-by: Claude Sonnet 4.5 <noreply@anthropic.com>

### Documentation

- Remove unsubstantiated SOTA claim from README
  ([#3](https://github.com/OpenAdaptAI/openadapt-retrieval/pull/3),
  [`d4ffec4`](https://github.com/OpenAdaptAI/openadapt-retrieval/commit/d4ffec47b9d996826fb0d59d6675c01f07fa119d))

* fix: Use filename-based GitHub Actions badge URL

The workflow-name-based badge URL was showing "no status" because GitHub requires workflow runs on
  the specified branch. Using the filename-based URL format
  (actions/workflows/publish.yml/badge.svg) is more reliable and works regardless of when the
  workflow last ran.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

* docs: remove unsubstantiated SOTA claim from README

Changed "state-of-the-art VLM" to just "Qwen3-VL model" since SOTA claims require citations or
  benchmark comparisons to be defensible.

---------

Co-authored-by: Claude Sonnet 4.5 <noreply@anthropic.com>


## v0.1.1 (2026-01-29)

### Bug Fixes

- **ci**: Remove build_command from semantic-release config
  ([`58696f1`](https://github.com/OpenAdaptAI/openadapt-retrieval/commit/58696f109ecb072a45b784ded207c059fa47eec4))

The python-semantic-release action runs in a Docker container where uv is not available. Let the
  workflow handle building instead.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>

### Continuous Integration

- Switch to python-semantic-release for automated versioning
  ([`102b44c`](https://github.com/OpenAdaptAI/openadapt-retrieval/commit/102b44ca628d932dd4a25d9255c16f1c46d380d3))

Replaces manual tag-triggered publish with python-semantic-release: - Automatic version bumping
  based on conventional commits - feat: -> minor, fix:/perf: -> patch - Creates GitHub releases
  automatically - Publishes to PyPI on release

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>


## v0.1.0 (2026-01-16)

### Build System

- Prepare package for PyPI publishing
  ([`31be776`](https://github.com/OpenAdaptAI/openadapt-retrieval/commit/31be776ca3fed1662bd414a5841a6834af0c3940))

- Add maintainers field (OpenAdaptAI) to pyproject.toml - Add extended project URLs (Documentation,
  Issues, Changelog) - Add MIT LICENSE file - Add GitHub Actions workflow for trusted PyPI
  publishing on version tags

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
