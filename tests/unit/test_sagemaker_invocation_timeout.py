#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Unit tests for the async `InvocationTimeoutSeconds` pass-through on the
SageMaker estimators. boto3 is mocked, so these run without the extra."""

import json
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np

from tabpfn_client.sagemaker import TabPFNClassifier


def _fake_s3_client() -> MagicMock:
    s3 = MagicMock()
    # `_invoke_async` reads `s3.exceptions.NoSuchKey` as an exception class.
    s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
    s3.get_object.return_value = {
        "Body": BytesIO(json.dumps({"prediction": [[0, 1]], "metadata": {}}).encode())
    }
    return s3


class TestSagemakerInvocationTimeout(unittest.TestCase):
    def _invoke_async_kwargs(self, invocation_timeout_s):
        clf = TabPFNClassifier(
            endpoint_name="ep",
            region_name="us-east-1",
            use_async=True,
            s3_bucket="bucket",
            async_poll_interval_s=0.0,
            invocation_timeout_s=invocation_timeout_s,
        )
        runtime = MagicMock()
        runtime.invoke_endpoint_async.return_value = {
            "OutputLocation": "s3://bucket/out.json"
        }
        with (
            patch.object(clf, "_runtime_client", return_value=runtime),
            patch.object(clf, "_s3_client", return_value=_fake_s3_client()),
        ):
            clf.fit(np.zeros((3, 2)), np.array([0, 1, 0]))
            clf.predict(np.zeros((2, 2)))
        return runtime.invoke_endpoint_async.call_args.kwargs

    def test_timeout_passed_through(self):
        kwargs = self._invoke_async_kwargs(1234)
        self.assertEqual(kwargs["InvocationTimeoutSeconds"], 1234)

    def test_timeout_omitted_by_default(self):
        kwargs = self._invoke_async_kwargs(None)
        self.assertNotIn("InvocationTimeoutSeconds", kwargs)


if __name__ == "__main__":
    unittest.main()
