"""Exceptions raised by the TabPFN client."""


class RetryableServerError(Exception):
    """Base exception for retryable server-side HTTP errors (typically 5xx)."""

    pass


class CappedRetryableServerError(Exception):
    """Retryable error, with retries capped over consecutive errors of this type."""

    pass


class EmptyResponseError(RuntimeError):
    """A success response ended without a final payload.

    Long-running endpoints stream whitespace keepalive pings before the result.
    A body holding nothing else means the server ended the request without a
    result, so it is treated like an internal server error (HTTP 500).
    """

    pass


class FittedModelNotFoundError(RuntimeError):
    """The server has no fitted model for the id the estimator refers to.

    Raised by ``predict`` when ``model_id_`` -- set by ``fit()``, restored by
    ``load_model()`` or assigned directly -- is unknown to the server: the fitted
    model was deleted (for instance through ``UserDataClient``), or it belongs to
    a different account. Call ``fit()`` again to create a new one.
    """

    pass
