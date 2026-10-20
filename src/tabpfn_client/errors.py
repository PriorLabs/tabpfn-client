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


class FitFailedError(RuntimeError):
    """The server reports that a fit it accepted has failed.

    Raised by ``fit()`` and ``wait_for_fit()`` when they poll the server for the
    outcome of a fit, as ``fit()`` does in ASYNC mode. The message holds the
    server's reason.
    """

    pass


class PredictFailedError(RuntimeError):
    """The server reports that a prediction it accepted has failed.

    Raised by ``predict()`` and ``wait_for_predict()`` when they poll the server
    for the outcome of a prediction, as ``predict()`` does when the prediction
    runs as a background job. The message holds the server's reason and its
    error code.
    """

    pass


class RefitRequiredError(PredictFailedError):
    """The fitted model can no longer predict: call ``fit()`` again.

    Raised by ``predict()`` and ``wait_for_predict()`` when the server cannot
    predict with the fit ``model_id_`` refers to: an earlier version of the
    service created it, or its cached state is no longer valid. The message
    holds the server's reason.
    """

    pass


class StreamedServerError(RuntimeError):
    """The server reported an error after it had started its response.

    Long-running endpoints answer HTTP 200 while they work, so an error they hit
    after that arrives in the body, as a ``_streamed_error`` envelope that holds
    the HTTP status the error would have had.
    """

    # The defaults let pickle and copy rebuild the error from its message, as
    # they rebuild any exception, and then restore both attributes. A process
    # pool hands an error to its parent that way.
    def __init__(
        self, message: str, *, status_code: int | None = None, detail: str = ""
    ):
        """Initialize the error.

        Args:
            message: The error message.
            status_code: The HTTP status the envelope reports, or None if it
                reports none.
            detail: The server's description of the error.
        """
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class PredictionGoneError(RuntimeError):
    """The server reports that it does not have the prediction.

    Its id is unknown (HTTP 404), or the server no longer stores its result
    (HTTP 410).
    """

    pass
