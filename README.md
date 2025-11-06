# TabPFN Client

[![PyPI version](https://badge.fury.io/py/tabpfn-client.svg)](https://badge.fury.io/py/tabpfn-client)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.gg/BHnX2Ptf4j)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb)
[![Documentation](https://img.shields.io/badge/docs-priorlabs.ai-blue)](https://priorlabs.ai/docs)
[![Twitter Follow](https://img.shields.io/twitter/follow/Prior_Labs?style=social)](https://twitter.com/Prior_Labs)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/tabpfn-client/)
![Last Commit](https://img.shields.io/github/last-commit/automl/tabpfn-client)

TabPFN is a foundation model for tabular data that outperforms traditional methods while being dramatically faster. This client library provides easy access to the TabPFN API, enabling state-of-the-art tabular machine learning in just a few lines of code.

## Interactive Notebook Tutorial
> [!TIP]
>
> Dive right in with our interactive Colab notebook! It's the best way to get a hands-on feel for TabPFN, walking you through installation, classification, and regression examples.
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb)

## ✅ Stable Release

This API is now in a stable release. It has been extensively tested and is used across multiple use cases. While we continue to make improvements, the core service is reliable for day-to-day use. Please reach out to us if you encounter any stability issues.

This is a cloud-based service: your data will be sent to our servers for processing. 

Please only upload data you have permission to share, and avoid sensitive, confidential, or personally identifiable information. Consider anonymizing or pseudonymizing your data in line with your organization’s policies.

## 🌐 TabPFN Ecosystem

Choose the right TabPFN implementation for your needs:

- **TabPFN Client (this repo)**: Easy-to-use API client for cloud-based inference
- **[TabPFN Extensions](https://github.com/priorlabs/tabpfn-extensions)**: Community extensions and integrations
- **[TabPFN](https://github.com/priorlabs/tabpfn)**: Core implementation for local deployment and research
- **[TabPFN UX](https://ux.priorlabs.ai)**: No-code TabPFN usage

## 🏁 Quick Start

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

### Best Results

For optimal performance, use the `AutoTabPFNClassifier` or `AutoTabPFNRegressor` for post-hoc ensembling. These can be found in the [TabPFN Extensions](https://github.com/PriorLabs/tabpfn-extensions) repository. Post-hoc ensembling combines multiple TabPFN models into an ensemble.

## 🔑 Authentication

### Load Your Token

```python
import tabpfn_client
token = tabpfn_client.get_access_token()
```

and login (on another machine) using your access token, skipping the interactive flow, use:

```python
tabpfn_client.set_access_token(token)
```

## 🤝 Join Our Community

We're building the future of tabular machine learning and would love your involvement! Here's how you can participate and get help:

1. **Try TabPFN**: Use it in your projects and share your experience
2. **Connect & Learn**:
   - Join our [Discord Community](https://discord.gg/VJRuU3bSxt) for discussions and support
   - Read our [Documentation](https://priorlabs.ai/) for detailed guides
   - Check out [GitHub Issues](https://github.com/automl/tabpfn-client/issues) for known issues and feature requests
3. **Contribute**:
   - Report bugs or request features through issues
   - Submit pull requests (see development guide below)
   - Share your success stories and use cases
4. **Stay Updated**: Star the repo and join Discord for the latest updates

## 📊 Usage Limits

### API Cost Calculation

Each API request consumes usage credits based on the following formula:

```python
api_cost = max((num_train_rows + num_test_rows) * num_cols * n_estimators, 5000)
```

Where `n_estimators` defaults to:

- 8 for both classification and regression

Per day the current prediction allowance is 100,000,000 credits. We will adjust this limit based on usage patterns. If you require further credits, please fill out [this form](https://forms.gle/2xbHnkxbpb7BtpE47).

### Monitoring Usage

Track your API usage through response headers:

- `X-RateLimit-Limit`: Your total allowed usage
- `X-RateLimit-Remaining`: Remaining usage
- `X-RateLimit-Reset`: Reset timestamp (UTC)

Usage limits reset daily at 00:00:00 UTC.

### Size Limitations

1. Maximum total cells per request must be below 20,000,000:

```python
max_cells = (num_train_rows + num_test_rows) * num_cols
```

2. For regression with full output (`return_full_output=True`), the number of test samples must be below 500:

```python
if task == 'regression' and return_full_output and num_test_samples > 500:
    raise ValueError("Cannot return full output for regression with >500 test samples")
```

These limits will be increased in future releases.

## Access/Delete Personal Information

You can use our `UserDataClient` to access and delete personal information.

```python
from tabpfn_client import UserDataClient

print(UserDataClient.get_data_summary())
```

## 📚 Citation

```bibtex
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

## 🤝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Development

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
mkdir ~/tabpfn-client-test.tmp && cp tabpfn_client/tests/quick_test.py ~/tabpfn-client-test.tmp && cp requirements.txt ~/tabpfn-client-test.tmp && cd ~/tabpfn-client-test.tmp
uv venv && source .venv/bin/activate
# We use --pre for the rc version and --no-deps because TestPyPI dependencies are unreliable.
uv pip install --pre --index-url https://test.pypi.org/simple/ --no-deps tabpfn-client
uv pip install -r requirements.txt
python quick_test.py
```

3. Return to this repo. Correct the version. Ideally this should be what is in main. It shouldn't have an rc suffix unless we're doing broader pre-release testing.

4. Build, upload to the real PyPI, install and run a quick test.

```bash
rm -rf ~/tabpfn-client-test.tmp dist
uv pip install --upgrade build && python -m build
uv pip install --upgrade twine && python -m twine upload --repository pypi dist/*
# Use a separate directory for testing so we don't accidentally run the local code
mkdir ~/tabpfn-client-test.tmp && cp tabpfn_client/tests/quick_test.py ~/tabpfn-client-test.tmp && cp requirements.txt ~/tabpfn-client-test.tmp && cd ~/tabpfn-client-test.tmp
uv venv && source .venv/bin/activate
# We use --pre in case you intend to push an rc version.
uv pip install -U --pre tabpfn-client
python quick_test.py
```
