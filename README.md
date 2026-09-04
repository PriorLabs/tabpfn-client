# TabPFN Client

[![PyPI version](https://badge.fury.io/py/tabpfn-client.svg)](https://badge.fury.io/py/tabpfn-client)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.gg/BHnX2Ptf4j)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb)
[![Documentation](https://img.shields.io/badge/docs-priorlabs.ai-blue)](https://priorlabs.ai/docs)
[![Twitter Follow](https://img.shields.io/twitter/follow/Prior_Labs?style=social)](https://twitter.com/Prior_Labs)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/tabpfn-client/)
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
- **[TabPFN UX](https://platform.priorlabs.ai)**: No-code TabPFN usage

## Quick Start

### Installation

```bash
pip install --upgrade tabpfn-client
```

### Basic Usage

Set a token first — `fit()` raises without one and never prompts. Generate it at
[platform.priorlabs.ai/account/api-keys](https://platform.priorlabs.ai/account/api-keys):

```bash
export TABPFN_TOKEN="<your-token>"
```

See [Authentication](#authentication) for the alternatives.

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
- Thinking-mode fits draw from a **separate, smaller budget** than regular fits — they do not count against your regular prediction allowance, and you cannot use your regular allowance for them. The number of thinking-mode fits you can run per day is limited. If you need more capacity, request an increase via [platform.priorlabs.ai](https://platform.priorlabs.ai).

## KV Cache

`fit_mode="fit_with_cache"` caches the fit so repeated predictions against it are faster. Use it when you fit once and predict many times.

```python
from tabpfn_client import TabPFNRegressor

model = TabPFNRegressor(fit_mode="fit_with_cache")
model.fit(X_train, y_train)

model.predict(X_test)     # served from the cache built during fit()
model.predict(X_other)
```

`fit()` records the id of the fitted model as `model_id_`. To predict against the same fit from another process or machine, save the estimator and load it there — see [Saving and Loading Fitted Models](#saving-and-loading-fitted-models).

Notes:

- `fit_mode` accepts `"fit_preprocessors"` (the default) or `"fit_with_cache"`.
- Predictions are the same either way — caching changes the speed, not the model.
- Not compatible with thinking mode.

## Saving and Loading Fitted Models

A fit lives on the TabPFN server, so a fitted estimator can be saved without its training data and loaded in a later run — or on another machine — to predict without fitting again. This pays off most when the fit was expensive (thinking mode) or when you predict many times against one fit (`fit_mode="fit_with_cache"`).

```python
from tabpfn_client import TabPFNClassifier

model = TabPFNClassifier(thinking_mode=True)
model.fit(X_train, y_train)
model.save_model("model.json")

# In a later run:
model = TabPFNClassifier.load_model("model.json")
model.predict(X_test)
```

`save_model()` writes a small JSON record: the id of the fitted model, the hyperparameters, the class labels and the training-set size. It also returns that record as a dict, so it can be kept elsewhere (a database, an experiment tracker) and passed to `load_model()` directly.

Notes:

- The loading process must authenticate with the account that ran the fit; fitted models are not visible to other accounts.
- `load_model()` makes no request. A fitted model stays usable for as long as its training data remains on the server; if it was deleted (for instance with `UserDataClient`), the first `predict()` raises `FittedModelNotFoundError`.
- Pickling with `pickle` or `joblib` works too and stays small, since the estimator holds no training data. The JSON record is the format that is readable and meant to stay loadable across tabpfn-client versions.

A script that reruns often can reuse the fit when it is available and fit otherwise:

```python
from tabpfn_client import FittedModelNotFoundError

try:
    model = TabPFNClassifier.load_model("model.json")
    predictions = model.predict(X_test)
except (FileNotFoundError, FittedModelNotFoundError):
    model = TabPFNClassifier().fit(X_train, y_train)
    model.save_model("model.json")
    predictions = model.predict(X_test)
```
## Plotting

`plot_regression_distribution` draws the predictive distribution behind a regression prediction. Install the optional dependency with `pip install "tabpfn-client[viz]"`.

```python
from tabpfn_client import TabPFNRegressor
from tabpfn_client.visualisation import plot_regression_distribution

model = TabPFNRegressor()
model.fit(X_train, y_train)
prediction = model.predict(X_test, output_type="full")

ax = plot_regression_distribution(prediction, sample_idx=0)
ax.figure.savefig("distribution.png")
```

Notes:

- Requires `output_type="full"`; `sample_idx` picks the row of `X_test` to plot.
- `statistics`, `quantile_interval`, `zoom_quantile` and `smooth` control what is drawn; see the docstring for the defaults.
- Pass `ax=` to overlay several samples on one axes.

## Authentication

Authentication is token-based. Generate a token at
[platform.priorlabs.ai/account/api-keys](https://platform.priorlabs.ai/account/api-keys), then supply it
in one of two ways.

Via the environment, which needs no code changes:

```bash
export TABPFN_TOKEN="<your-token>"
```

Or in code, before the first fit or predict:

```python
import tabpfn_client
tabpfn_client.set_access_token("<your-token>")
```

If neither is set, `init()` raises a `RuntimeError` explaining where to get a token. It
never prompts — this is a library, so authentication is not allowed to block on input.
The one exception is `interactive_login()` below, which you call yourself.

### Interactive Login (opt-in)

If you would rather not copy a token by hand, call `interactive_login()` explicitly:

```python
from tabpfn_client import interactive_login
interactive_login()
```

It offers two routes:

- **Log in** — opens the Prior Labs login page, where you can sign in or use SSO, and waits
  for the resulting API key. A local callback receives the key automatically; if that does
  not come through (some identity providers drop the callback), you can paste the key at the
  prompt instead. Over SSH the flow prints the URL and waits for a paste. Pass
  `open_browser=False` to skip the browser entirely.
- **Create an account** — runs entirely in the terminal: email and password, a short
  profile, then an emailed verification code. No browser required, which makes it usable
  from a hosted notebook where opening a tab is not an option.

Either way the token is verified and cached, so later runs need no input.

This is **opt-in only**. `init()`, `fit()`, and `predict()` never trigger it — they use the
token sources above and fail with instructions when none is available.

`interactive_login()` is also the only thing that writes the token cache. A token supplied
through `TABPFN_TOKEN` or `set_access_token()` stays in memory for that process and is
never copied to disk.

### Load Your Token

To read back the token in use, for example to pass it to another machine:

```python
import tabpfn_client
token = tabpfn_client.get_access_token()
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

## Azure AI Foundry

If you've deployed TabPFN to an Azure AI Foundry managed online endpoint, you can invoke it through `tabpfn_client.foundry` using the same scikit-learn surface. There is no PriorLabs API token in this path — you authenticate against your own Foundry endpoint with its bearer key, and `predict` calls are billed by Azure rather than against your TabPFN usage allowance.

Point the estimator at your endpoint URL and pass the bearer key:

```python
from tabpfn_client.foundry import TabPFNClassifier, TabPFNRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

clf = TabPFNClassifier(
    endpoint_url="https://<your-endpoint>.<region>.inference.ml.azure.com/predict",
    api_key="<your-foundry-bearer-token>",
)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.predict_proba(X_test)
```

Notes:

- `endpoint_url` is the full Foundry scoring URL, including the `/predict` path. The bearer key is sent as `Authorization: Bearer <api_key>`.
- Requests are sent as `application/json`; the Foundry path does not use multipart, so all data travels JSON-encoded.

Set `use_kv_cache=True` if you will call `predict*` more than once on the same training data. The first call ships `X_train` / `y_train` to the endpoint, runs the fit there, and gets back a `model_id`. The client caches that id, and every subsequent call sends only `X_test` plus the id — the server **skips the fit and runs inference only**. That makes follow-up calls dramatically faster on non-trivial training sets, and shrinks the wire payload from O(n_train + n_test) down to O(n_test):

```python
clf = TabPFNClassifier(
    endpoint_url="https://<your-endpoint>.<region>.inference.ml.azure.com/predict",
    api_key="<your-foundry-bearer-token>",
    use_kv_cache=True,
)
clf.fit(X_train, y_train)
clf.predict(X_test_a)          # first call: fit + predict on the endpoint
clf.predict_proba(X_test_b)    # cache hit: predict only — much faster
```

Leave `use_kv_cache=False` (the default) when each call uses a different training set; otherwise the cache is dead weight on the endpoint.

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

Each API request consumes usage credits; the cost grows with the number of rows and columns in your dataset. You can check your current usage at [platform.priorlabs.ai/account/usage](https://platform.priorlabs.ai/account/usage).

### Monitoring Usage

Track your API usage through response headers:

- `X-RateLimit-Limit`: Your total allowed usage
- `X-RateLimit-Remaining`: Remaining usage
- `X-RateLimit-Reset`: Reset timestamp (UTC)

Usage limits reset daily at 00:00:00 UTC.

### Size Limitations

Per-model size limits (rows, columns, cells, classes) are enforced by the server and are returned from `/tabpfn/get_settings`. The client validates against the most permissive limit at `fit` time and against the selected model's limit at `predict` time, raising `ValueError` before the request is sent.

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

To encourage better coding practices, linting and formatting are managed with [Trunk](https://docs.trunk.io/code-quality/overview/initialize-trunk) (running `ruff` and `basedpyright`). To check your changes, run:

```bash
trunk check
```

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
