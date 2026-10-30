# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.1] - 2026-10-30

### Added

- `thinking_effort` accepts `"micro"`, `"mini"`, `"low"`, `"xhigh"` and `"max"` besides `"medium"` and `"high"`, for accounts the server allows to use them. ([#433](https://github.com/PriorLabs/tabpfn-client/pull/433))

### Fixed

- `fit` and `predict` no longer fail on a DataFrame with a period column, for example years from `pd.period_range`. ([#429](https://github.com/PriorLabs/tabpfn-client/pull/429))


## [0.7.0] - 2026-10-28

### Added

- `TabPFNClassifier` and `TabPFNRegressor` gained `submit_fit()`, which starts a fit without waiting for it, and `wait_for_fit()`. `save_model()` saves the pending model, so a later run can `load_model()` it and wait for the same fit instead of fitting again; until the fit has been waited for, `predict()` raises `NotFittedError`. A fit the server reports as failed now raises `FitFailedError`, a `RuntimeError`. ([#404](https://github.com/PriorLabs/tabpfn-client/pull/404))
- `api_mode` now also applies to `predict()`: with `"auto"`, the default, a prediction runs as a background job that the client polls once the training rows of the fit or the rows to predict reach thresholds that the server sets, while a fit with `fit_mode="fit_with_cache"` still predicts with one request. A prediction that takes longer than the server waits on one request no longer fails: the client polls for its result. `TabPFNClassifier` and `TabPFNRegressor` gained `submit_predict()`, which starts a prediction as a background job without waiting for it, and `wait_for_predict()`, which waits for it and returns what `predict()` returns, or `predict_proba()` for `output_type="probas"`. The model holds one pending prediction, which `save_model()` saves and a new `submit_predict()` or `predict()` replaces once the server has accepted the new prediction. While `predict()` waits for a prediction that runs as a background job, the model holds it, as `fit()` holds a new fit, so `wait_for_predict()` resumes a wait that stopped. `save_model()` no longer saves `api_mode`, and `load_model()` ignores it in a file an earlier version saved: a loaded model uses the `api_mode` of the process that loads it. Two new errors, both `RuntimeError`s: `PredictFailedError`, raised when the server reports that a prediction it created failed, and its subclass `RefitRequiredError`, raised instead of a plain `RuntimeError` when the fitted model can no longer predict: call `fit()` again. A thinking-mode fit now always runs as a background job, whatever `api_mode` is, instead of keeping one request open for up to an hour. If `fit()` stops waiting for a background fit, after `TABPFN_CLIENT_ASYNC_POLL_TIMEOUT` seconds (by default the server's 2 hours) or an interrupt, the model is pending on the new fit, which `wait_for_fit()` resumes and `save_model()` saves; a fit that the server reports as failed leaves the model unfitted. A regressor's `"full"` output that is split over several requests keeps up to 4 of them running at once as background jobs. `fit()` and `predict()` no longer resend a request that the server may already have received, for example after a lost response or an HTTP 503, which could fit or predict, and bill, twice: they retry only a request that never reached the server. ([#413](https://github.com/PriorLabs/tabpfn-client/pull/413))
- `predict` sends a small test set inside the predict request instead of uploading it first, when the server allows it. The server's settings say how large an inline test set may be. A server that does not advertise a limit gets an upload, as before. ([#424](https://github.com/PriorLabs/tabpfn-client/pull/424))
- New `tabpfn_client.explain.permutation_shap()` computes SHAP values for many explained rows in each prediction request, not one request per row as `shap.PermutationExplainer` does. Pass the training data as `background`, and it samples the background rows from it. `to_shap()` gives a `shap.Explanation` for the shap plots. ([#426](https://github.com/PriorLabs/tabpfn-client/pull/426))

### Changed

- Requests after a pause of more than 5 s no longer open a new connection: the client keeps idle connections open for up to 5 minutes, and prediction downloads now reuse connections too. ([#428](https://github.com/PriorLabs/tabpfn-client/pull/428))

### Fixed

- Treat a successful response with an empty final payload (only keepalive pings) as an internal server error instead of surfacing a confusing validation error; fit status polling retries it like an HTTP 500. ([#403](https://github.com/PriorLabs/tabpfn-client/pull/403))
- `TabPFNClassifier.fit` in thinking mode now rejects a target with only one class before sending it, including the SageMaker and Foundry estimators. ([#406](https://github.com/PriorLabs/tabpfn-client/pull/406))
- Development and release tooling no longer trusts the compromised eMudhra (emSign) CAs: pinned certifi, uv, pip and pypa/gh-action-pypi-publish are bumped to their fixed releases. ([#419](https://github.com/PriorLabs/tabpfn-client/pull/419))


## [0.7.0rc1] - 2026-10-28

### Added

- `TabPFNClassifier` and `TabPFNRegressor` gained `submit_fit()`, which starts a fit without waiting for it, and `wait_for_fit()`. `save_model()` saves the pending model, so a later run can `load_model()` it and wait for the same fit instead of fitting again; until the fit has been waited for, `predict()` raises `NotFittedError`. A fit the server reports as failed now raises `FitFailedError`, a `RuntimeError`. ([#404](https://github.com/PriorLabs/tabpfn-client/pull/404))
- `api_mode` now also applies to `predict()`: with `"auto"`, the default, a prediction runs as a background job that the client polls once the training rows of the fit or the rows to predict reach thresholds that the server sets, while a fit with `fit_mode="fit_with_cache"` still predicts with one request. A prediction that takes longer than the server waits on one request no longer fails: the client polls for its result. `TabPFNClassifier` and `TabPFNRegressor` gained `submit_predict()`, which starts a prediction as a background job without waiting for it, and `wait_for_predict()`, which waits for it and returns what `predict()` returns, or `predict_proba()` for `output_type="probas"`. The model holds one pending prediction, which `save_model()` saves and a new `submit_predict()` or `predict()` replaces once the server has accepted the new prediction. While `predict()` waits for a prediction that runs as a background job, the model holds it, as `fit()` holds a new fit, so `wait_for_predict()` resumes a wait that stopped. `save_model()` no longer saves `api_mode`, and `load_model()` ignores it in a file an earlier version saved: a loaded model uses the `api_mode` of the process that loads it. Two new errors, both `RuntimeError`s: `PredictFailedError`, raised when the server reports that a prediction it created failed, and its subclass `RefitRequiredError`, raised instead of a plain `RuntimeError` when the fitted model can no longer predict: call `fit()` again. A thinking-mode fit now always runs as a background job, whatever `api_mode` is, instead of keeping one request open for up to an hour. If `fit()` stops waiting for a background fit, after `TABPFN_CLIENT_ASYNC_POLL_TIMEOUT` seconds (by default the server's 2 hours) or an interrupt, the model is pending on the new fit, which `wait_for_fit()` resumes and `save_model()` saves; a fit that the server reports as failed leaves the model unfitted. A regressor's `"full"` output that is split over several requests keeps up to 4 of them running at once as background jobs. `fit()` and `predict()` no longer resend a request that the server may already have received, for example after a lost response or an HTTP 503, which could fit or predict, and bill, twice: they retry only a request that never reached the server. ([#413](https://github.com/PriorLabs/tabpfn-client/pull/413))
- `predict` sends a small test set inside the predict request instead of uploading it first, when the server allows it. The server's settings say how large an inline test set may be. A server that does not advertise a limit gets an upload, as before. ([#424](https://github.com/PriorLabs/tabpfn-client/pull/424))
- New `tabpfn_client.explain.permutation_shap()` computes SHAP values for many explained rows in each prediction request, not one request per row as `shap.PermutationExplainer` does. Pass the training data as `background`, and it samples the background rows from it. `to_shap()` gives a `shap.Explanation` for the shap plots. ([#426](https://github.com/PriorLabs/tabpfn-client/pull/426))

### Changed

- Requests after a pause of more than 5 s no longer open a new connection: the client keeps idle connections open for up to 5 minutes, and prediction downloads now reuse connections too. ([#428](https://github.com/PriorLabs/tabpfn-client/pull/428))

### Fixed

- Treat a successful response with an empty final payload (only keepalive pings) as an internal server error instead of surfacing a confusing validation error; fit status polling retries it like an HTTP 500. ([#403](https://github.com/PriorLabs/tabpfn-client/pull/403))
- `TabPFNClassifier.fit` in thinking mode now rejects a target with only one class before sending it, including the SageMaker and Foundry estimators. ([#406](https://github.com/PriorLabs/tabpfn-client/pull/406))
- Development and release tooling no longer trusts the compromised eMudhra (emSign) CAs: pinned certifi, uv, pip and pypa/gh-action-pypi-publish are bumped to their fixed releases. ([#419](https://github.com/PriorLabs/tabpfn-client/pull/419))


## [0.6.1] - 2026-09-28

### Added

- Add simple and advanced text handling options to the classifier and regressor, with advanced as the default. ([#398](https://github.com/PriorLabs/tabpfn-client/pull/398))
- When the server returns a prediction through a signed download URL, the client now fetches it transparently. ([#399](https://github.com/PriorLabs/tabpfn-client/pull/399))

### Fixed

- Validate training uploads against the server-advertised upload cap and defer subsampled context limits to the server. ([#396](https://github.com/PriorLabs/tabpfn-client/pull/396))


## [0.6.0] - 2026-09-15

### Added

- `TabPFNClassifier` and `TabPFNRegressor` gained `save_model()` and `load_model()`, which persist a fitted estimator as a small JSON record — the id of the server-side fit, the hyperparameters, the class labels and the training-set size — so a later run, or another machine authenticated with the same account, can predict against the same fit without fitting again. `save_model()` also returns the record as a dict for storing elsewhere. When the server no longer has the fit, for instance after it was deleted with `UserDataClient`, `predict()` raises the new `FittedModelNotFoundError`. ([#376](https://github.com/PriorLabs/tabpfn-client/pull/376))
- Add `estimate_cost()` to check prediction, Thinking, and cached prediction costs from dataset dimensions without uploading data or consuming quota. ([#382](https://github.com/PriorLabs/tabpfn-client/pull/382))
- Thinking mode accepts `group_col`, `time_col` and `group_time_col` on `TabPFNClassifier` and `TabPFNRegressor`, so grouped or time-ordered rows stay together during the fit. ([#387](https://github.com/PriorLabs/tabpfn-client/pull/387))
- Add the `v3.5-fast` model version: `v3.5-fast_default` now appears in `list_available_models()` and `create_default_for_version("v3.5-fast")` selects it. Prediction metadata returned by the server now also records `n_estimators`, `billing_model_version`, `execution_mode`, `cache_outcome` and, for classifiers, `top_k`. ([#389](https://github.com/PriorLabs/tabpfn-client/pull/389))
- Estimators report how long the server spent waiting for and working on each stage of `fit()` and `predict()`, via `get_timings()`, `fit_timings_` and `last_predict_timings`. ([#392](https://github.com/PriorLabs/tabpfn-client/pull/392))

### Changed

- Clarify estimator usage guidance for TabPFN-3 and later versions, covering dataset capacity, subsampling, and automatic handling of categorical values, raw text, and missing feature values without manual preprocessing. ([#383](https://github.com/PriorLabs/tabpfn-client/pull/383))
- `fit()` and `predict()` no longer print a progress spinner to stdout, `init()` no longer announces that it is reusing a cached token, and routine messages such as "Test set already exists, skipping upload." are logged at DEBUG instead of WARNING. `TABPFN_CLIENT_CI_MODE`, which only disabled the spinner, is removed. ([#385](https://github.com/PriorLabs/tabpfn-client/pull/385))

### Fixed

- Restore `from tabpfn_client.constants import ModelVersion`, which broke when the enum moved to `tabpfn_client.api_models`. ([#384](https://github.com/PriorLabs/tabpfn-client/pull/384))
- Fast model names such as `v3-fast_default` and `v3.5-fast_default` now resolve to their own model version for the client-side dataset size checks, instead of being treated as v2. ([#388](https://github.com/PriorLabs/tabpfn-client/pull/388))


## [0.5.3] - 2026-09-05

### Fixed

- `tabpfn_client.hosted.TabPFNClassifier` and `TabPFNRegressor` no longer force client-side defaults for `n_estimators`, `softmax_temperature`, `average_before_softmax`, and `inference_precision`. Unset values now defer to the inference container's own defaults; user-supplied overrides still win. ([#378](https://github.com/PriorLabs/tabpfn-client/pull/378))


## [0.5.2] - 2026-09-04

### Added

- `TabPFNClassifier` and `TabPFNRegressor` accept `model_path="v3.5_default"` (and `create_default_for_version(ModelVersion.V3_5)`) to request the TabPFN 3.5 pre-release on accounts that have access to it. ([#374](https://github.com/PriorLabs/tabpfn-client/pull/374))
- Add ``plot_regression_distribution`` to visualise the predictive distribution returned by regression predictions with ``output_type="full"``. Install the optional plotting dependencies with ``pip install "tabpfn-client[viz]"``. ([#375](https://github.com/PriorLabs/tabpfn-client/pull/375))
- `tabpfn_client.hosted.TabPFNClassifier` and `TabPFNRegressor` accept `model_version="v3.5"` (or any other baked family) to select a weights family on serving containers that support it. Older containers keep working — the field is only sent when set. ([#379](https://github.com/PriorLabs/tabpfn-client/pull/379))


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
