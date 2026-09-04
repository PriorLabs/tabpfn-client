class RetryableServerError(Exception):
    """
    Base exception for retryable server-side HTTP errors (typically 5xx).
    """

    pass


class CappedRetryableServerError(Exception):
    """
    An error that is retryable, but with a capped number of retries given
    consecutive errors of the same type.
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
