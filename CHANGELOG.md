# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0rc1] - 2026-08-26

### Breaking Changes

- Authentication is token-based by default. `init()` never prompts: it reads `TABPFN_TOKEN`, a token set with `set_access_token()`, or one cached by a previous login, and raises with instructions when none is available. `tabpfn_client.browser_auth` and the login helpers on `ServiceClient` are removed. ([#352](https://github.com/PriorLabs/tabpfn-client/pull/352))

### Added

- Added `interactive_login()`, an opt-in browser or terminal login that verifies the resulting API key and caches it for later runs. ([#352](https://github.com/PriorLabs/tabpfn-client/pull/352))
- The self-hosted client accepts `payload_format="parquet"`, which sends datasets as Parquet files instead of inline JSON. Encoding is faster on large tables, and missing values and datetimes survive the round trip, which JSON cannot represent. ([#359](https://github.com/PriorLabs/tabpfn-client/pull/359))
- `tabpfn_client.__version__` reports the installed package version. ([#361](https://github.com/PriorLabs/tabpfn-client/pull/361))
