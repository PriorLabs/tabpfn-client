# TabPFN Client

[![PyPI version](https://badge.fury.io/py/tabpfn-client.svg)](https://badge.fury.io/py/tabpfn-client)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.gg/BHnX2Ptf4j)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb)
[![Documentation](https://img.shields.io/badge/docs-priorlabs.ai-blue)](https://priorlabs.ai/docs)
[![Twitter Follow](https://img.shields.io/twitter/follow/Prior_Labs?style=social)](https://twitter.com/Prior_Labs)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/tabpfn-client/)
![Last Commit](https://img.shields.io/github/last-commit/PriorLabs/tabpfn-client)

TabPFN is a foundation model for tabular data that outperforms traditional methods while being dramatically faster. This client library provides easy access to the TabPFN API, enabling state-of-the-art tabular machine learning in just a few lines of code.

## Interactive Notebook Tutorial
> [!TIP]
>
> Dive right in with our interactive Colab notebook! It's the best way to get a hands-on feel for TabPFN, walking you through installation, classification, and regression examples.
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb)

## Stable Release

This API is now in a stable release. It has been extensively tested and is used across multiple use cases. While we continue to make improvements, the core service is reliable for day-to-day use. Please reach out to us if you encounter any stability issues.

This is a cloud-based service: your data will be sent to our servers for processing. 

Please only upload data you have permission to share, and avoid sensitive, confidential, or personally identifiable information. Consider anonymizing or pseudonymizing your data in line with your organization’s policies.

## TabPFN Ecosystem

Choose the right TabPFN implementation for your needs:

- **TabPFN Client (this repo)**: Easy-to-use API client for cloud-based inference
- **[TabPFN Extensions](https://github.com/priorlabs/tabpfn-extensions)**: Community extensions and integrations
- **[TabPFN](https://github.com/priorlabs/tabpfn)**: Core implementation for local deployment and research
- **[TabPFN UX](https://ux.priorlabs.ai)**: No-code TabPFN usage

## Quick Start

### Installation

```bash
pip install --upgrade tabpfn-client
```

### Basic Usage

```python
from tabpfn_client import init, TabPFNClassifier, TabPFNRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load an example dataset

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Use it like any sklearn model
model = TabPFNClassifier()
model.fit(X_train, y_train)
# Get predictions
predictions = model.predict(X_test)
# Get probability estimates
probabilities = model.predict_proba(X_test)
```

## Thinking Mode

Thinking mode trades extra fit-time compute for higher predictive quality. The server explores additional configurations during `fit()` and returns a tuned model; `predict()` then runs as usual.

```python
from tabpfn_client import TabPFNClassifier

# Simplest form: enable with defaults (effort="medium").
model = TabPFNClassifier(thinking_mode=True)
model.fit(X_train, y_train)
model.predict(X_test)
```

Knobs:

- `thinking_mode: bool = False` — enable thinking with default effort. Equivalent to `thinking_effort="medium"`.
- `thinking_effort: {"medium", "high"} | None` — effort level. Setting this also enables thinking, so `thinking_mode=True` is optional when you've set the level explicitly.
- `thinking_timeout_s: float | None` — budget for the fit, in seconds. Only consulted when thinking is enabled. Capped at 2400 (40 minutes).
- `thinking_metric: str | None` — optimization metric for the fit. Only consulted when thinking is enabled. See the constructor docstring of `TabPFNClassifier` / `TabPFNRegressor` for the full list of supported metrics per task (classification, multiclass, regression) and their aliases.

```python
model = TabPFNClassifier(
    thinking_effort="high",
    thinking_timeout_s=600,
    thinking_metric="roc_auc",
)
```

Notes:

- Thinking mode is only supported on v3 models. Leave `model_path` at its default (`"auto"`, which lets the server pick the latest default — currently a v3 model) or set it explicitly to a v3 model. Combining thinking with a v2 or v2.5 `model_path` raises `ValueError` client-side.
- `thinking_timeout_s` and `thinking_metric` are only consulted when thinking is enabled; passing them without `thinking_mode=True` or `thinking_effort=...` raises `ValueError`.
- Thinking-mode fits take longer than regular fits (often several minutes).
- Thinking-mode fits draw from a **separate, smaller budget** than regular fits — they do not count against your regular prediction allowance, and you cannot use your regular allowance for them. The number of thinking-mode fits you can run per day is limited. If you need more capacity, request an increase via [ux.priorlabs.ai](https://ux.priorlabs.ai).

## Authentication

### Load Your Token

```python
import tabpfn_client
token = tabpfn_client.get_access_token()
```

and login (on another machine) using your access token, skipping the interactive flow, use:

```python
tabpfn_client.set_access_token(token)
```

## AWS SageMaker (BYOC)

If you've subscribed to the TabPFN AWS Marketplace listing and deployed the container to a SageMaker real-time endpoint, you can invoke it through `tabpfn_client.sagemaker` using a near-identical scikit-learn surface. There is no PriorLabs API token in this path — you authenticate to your own AWS account, and `predict` calls are billed by AWS SageMaker rather than against your TabPFN usage allowance.

Install with the optional `sagemaker` extra to pull in `boto3`:

```bash
pip install --upgrade 'tabpfn-client[sagemaker]'
```

Then point the estimator at your endpoint:

```python
from tabpfn_client.sagemaker import TabPFNClassifier, TabPFNRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

clf = TabPFNClassifier(
    endpoint_name="your-sagemaker-endpoint-name",
    region_name="us-east-1",
)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.predict_proba(X_test)
```

Notes:

- AWS credentials are resolved through the standard `boto3` credential chain (env vars, `~/.aws/credentials`, instance profile, SSO, etc.). Pass `boto_session=session` to use an explicit `boto3.Session`.
- `fit()` does not call the endpoint — it stores `X_train` / `y_train` on the estimator. Training data is shipped with the next `predict*` call, which is where the actual fit runs on the endpoint. There is no separate training job.
- `use_kv_cache=True` opts into the v3 KV-cache path on the server: the first `predict*` round-trip uploads training data and captures a `model_id`, and subsequent calls send only `X_test` and the id. Default to `True` when you'll call `predict*` more than once on the same training data; leave it off if every call uses a different training set (no reuse), since the cache becomes dead weight on the endpoint.
- Constructor kwargs mirror the public `tabpfn_client.TabPFNClassifier` / `TabPFNRegressor` so the same code is portable between the managed API and a SageMaker endpoint, modulo `endpoint_name` / `region_name`.

Thinking mode is supported on SageMaker by passing the same `thinking_mode` / `thinking_effort` / `thinking_timeout_s` / `thinking_metric` kwargs:

```python
clf = TabPFNClassifier(
    endpoint_name="your-sagemaker-endpoint-name",
    region_name="us-east-1",
    thinking_mode=True,
    thinking_effort="medium",
)
```

The first `predict*` call after `fit()` runs the fit on the endpoint and can take from tens of seconds up to several minutes depending on `thinking_effort` and data size; the fitted model is cached on the endpoint and subsequent calls are fast. Caching is **required** when thinking is enabled (the client sets `use_kv_cache=True` automatically) — without it every prediction would redo the fit, which would exceed SageMaker's synchronous invoke window. Only `thinking_effort="medium"` is reliable within the real-time endpoint's ~60 s sync window for the *first* call; `"high"` may exceed it and is currently best-effort.

## Join Our Community

We're building the future of tabular machine learning and would love your involvement! Here's how you can participate and get help:

1. **Try TabPFN**: Use it in your projects and share your experience
2. **Connect & Learn**:
   - Join our [Discord Community](https://discord.gg/VJRuU3bSxt) for discussions and support
   - Read our [Documentation](https://priorlabs.ai/) for detailed guides
   - Check out [GitHub Issues](https://github.com/PriorLabs/tabpfn-client/issues) for known issues and feature requests
3. **Contribute**:
   - Report bugs or request features through issues
   - Submit pull requests (see development guide below)
   - Share your success stories and use cases
4. **Stay Updated**: Star the repo and join Discord for the latest updates

## Usage Limits

### API Cost Calculation

Each API request consumes usage credits; the cost grows with the number of rows and columns in your dataset. You can check your current usage at [ux.priorlabs.ai/account/usage](https://ux.priorlabs.ai/account/usage).

### Monitoring Usage

Track your API usage through response headers:

- `X-RateLimit-Limit`: Your total allowed usage
- `X-RateLimit-Remaining`: Remaining usage
- `X-RateLimit-Reset`: Reset timestamp (UTC)

Usage limits reset daily at 00:00:00 UTC.

### Size Limitations

Per-model size limits (rows, columns, cells, classes) are enforced by the server and are returned from `/tabpfn/get_model_limits`. The client validates against the most permissive limit at `fit` time and against the selected model's limit at `predict` time, raising `ValueError` before the request is sent.

In particular, regression with `output_type="full"` has a stricter cap on the number of test rows than regular regression predictions; split the test set across calls if you hit it.

These limits will be increased in future releases.

## Access/Delete Data

You can use our `UserDataClient` to access and delete personal information.

```python
from tabpfn_client import UserDataClient

print(UserDataClient.get_data_summary())
```

## Citation

You can read our paper explaining TabPFNv2 [here](https://doi.org/10.1038/s41586-024-08328-6), and the model report of TabPFN-2.5 [here](https://arxiv.org/abs/2511.08667).

<details>
<summary><b>BibTeX</b></summary>

```bibtex
@misc{grinsztajn2025tabpfn,
  title={TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models},
  author={Léo Grinsztajn and Klemens Flöge and Oscar Key and Felix Birkel and Philipp Jund and Brendan Roof and
          Benjamin Jäger and Dominik Safaric and Simone Alessi and Adrian Hayler and Mihir Manium and Rosen Yu and
          Felix Jablonski and Shi Bin Hoo and Anurag Garg and Jake Robertson and Magnus Bühler and Vladyslav Moroshan and
          Lennart Purucker and Clara Cornu and Lilly Charlotte Wehrhahn and Alessandro Bonetto and
          Bernhard Schölkopf and Sauraj Gambhir and Noah Hollmann and Frank Hutter},
  year={2025},
  eprint={2511.08667},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2511.08667},
}

@article{hollmann2025tabpfn,
 title={Accurate predictions on small data with a tabular foundation model},
 author={Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and
         Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and
         Schirrmeister, Robin Tibor and Hutter, Frank},
 journal={Nature},
 year={2025},
 month={01},
 day={09},
 doi={10.1038/s41586-024-08328-6},
 publisher={Springer Nature},
 url={https://www.nature.com/articles/s41586-024-08328-6},
}

@inproceedings{hollmann2023tabpfn,
  title={TabPFN: A transformer that solves small tabular classification problems in a second},
  author={Hollmann, Noah and M{\"u}ller, Samuel and Eggensperger, Katharina and Hutter, Frank},
  booktitle={International Conference on Learning Representations 2023},
  year={2023}
}
```

</details>

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

## Development

<details>
<summary><b>Setup, build, and release instructions</b></summary>

To encourage better coding practices, `ruff` has been added to the pre-commit hooks. This will ensure that the code is formatted properly before being committed. To enable pre-commit (if you haven't), run the following command:

```bash
pre-commit install
```

Additionally, it is recommended that developers install the ruff extension in their preferred editor. For installation instructions, refer to the [Ruff Integrations Documentation](https://docs.astral.sh/ruff/integrations/).

### Build from GitHub

```bash
git clone https://github.com/PriorLabs/tabpfn-client
cd tabpfn-client
git submodule update --init --recursive
pip install -e .
cd ..
```

NOTE: For development, you will need to download some additional dev dependencies.
Use the below command to get it ready for development and running tests.

```bash
pip install -e ".[dev]"
```

### Release

1. First ensure you've bumped the version in pyproject.toml. Use an rc suffix until you're sure it works. Something like x.y.zrc1.

2. Build, upload to the test PyPI, install and run a quick test.

Note: Assumes a working uv install + venv.

```bash
rm -rf ~/tabpfn-client-test.tmp dist
uv pip install --upgrade build && python -m build
uv pip install --upgrade twine && python -m twine upload --repository testpypi dist/*
# Use a separate directory for testing so we don't accidentally run the local code
mkdir ~/tabpfn-client-test.tmp && cp tests/quick_test.py ~/tabpfn-client-test.tmp && cd ~/tabpfn-client-test.tmp
uv venv && source .venv/bin/activate
# We use --pre for the rc version and --no-deps because TestPyPI dependencies are unreliable.
pip3 download --pre --index-url https://test.pypi.org/simple/ --no-deps tabpfn-client
uv pip install *.whl
python quick_test.py
```

3. Return to this repo. Correct the version. Ideally this should be what is in main. It shouldn't have an rc suffix unless we're doing broader pre-release testing.

4. Build, upload to the real PyPI, install and run a quick test.

```bash
rm -rf ~/tabpfn-client-test.tmp dist
uv pip install --upgrade build && python -m build
uv pip install --upgrade twine && python -m twine upload --repository pypi dist/*
# Use a separate directory for testing so we don't accidentally run the local code
mkdir ~/tabpfn-client-test.tmp && cp tests/quick_test.py ~/tabpfn-client-test.tmp && cd ~/tabpfn-client-test.tmp
uv venv && source .venv/bin/activate
# We use --pre in case you intend to push an rc version.
uv pip install -U --pre tabpfn-client
python quick_test.py
```

</details>

---
Built with ❤️ by the TabPFN community
