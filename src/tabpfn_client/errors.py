class RetryableServerError(Exception):
    """
    Base exception for retryable server-side HTTP errors (typically 5xx).
    """

    pass
