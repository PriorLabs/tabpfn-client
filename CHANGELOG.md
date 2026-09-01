# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.1] - 2026-09-01

### Added

- `tabpfn_client.hosted` estimators accept `use_kv_cache=True`, which reuses the endpoint's already-fitted model across repeated `predict` calls instead of re-fitting on every one — a large speed-up for workloads that predict many times against a single fit, such as SHAP, permutation importance and partial dependence. The endpoint's cache is bounded, so a model can be evicted between calls; that surfaces as an error and the estimator has to be re-fit. ([#372](https://github.com/PriorLabs/tabpfn-client/pull/372))

### Changed

- TabPFN Console URL moved from `ux.priorlabs.ai` to `platform.priorlabs.ai` (the old host continues to redirect). ([#368](https://github.com/PriorLabs/tabpfn-client/pull/368))
- Now that hosted estimators are recognised as classifiers, scikit-learn's default cross-validation splitter for them is `StratifiedKFold` rather than `KFold`; cross-validation scores and anything derived from them may shift slightly. ([#372](https://github.com/PriorLabs/tabpfn-client/pull/372))

### Fixed

- `TabPFNRegressor.predict(output_type="full")` now matches a local prediction: masked-out bars come back as `-inf` rather than `NaN`, and test sets above the server's full-output row cap are split across calls instead of raising. ([#369](https://github.com/PriorLabs/tabpfn-client/pull/369))
- Hosted estimators are now recognised by scikit-learn as classifiers and regressors, so tools that dispatch on estimator type — partial dependence among them — accept them. Datasets holding missing or non-finite values can also be sent as JSON, which previously rejected them and left Parquet as the only option. ([#372](https://github.com/PriorLabs/tabpfn-client/pull/372))


## [0.5.0] - 2026-08-26

No significant changes.


## [0.5.0rc1] - 2026-08-26

### Breaking Changes

- Authentication is token-based by default. `init()` never prompts: it reads `TABPFN_TOKEN`, a token set with `set_access_token()`, or one cached by a previous login, and raises with instructions when none is available. `tabpfn_client.browser_auth` and the login helpers on `ServiceClient` are removed. ([#352](https://github.com/PriorLabs/tabpfn-client/pull/352))

### Added

- Added `interactive_login()`, an opt-in browser or terminal login that verifies the resulting API key and caches it for later runs. ([#352](https://github.com/PriorLabs/tabpfn-client/pull/352))
- The self-hosted client accepts `payload_format="parquet"`, which sends datasets as Parquet files instead of inline JSON. Encoding is faster on large tables, and missing values and datetimes survive the round trip, which JSON cannot represent. ([#359](https://github.com/PriorLabs/tabpfn-client/pull/359))
- `tabpfn_client.__version__` reports the installed package version. ([#361](https://github.com/PriorLabs/tabpfn-client/pull/361))
