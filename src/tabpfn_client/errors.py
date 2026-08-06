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
