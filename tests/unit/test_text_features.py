import unittest
import numpy as np
import pandas as pd
from tabpfn_client.estimator import _clean_text_features
from io import BytesIO
from tabpfn_common_utils import utils


class TestCleanTextFeatures(unittest.TestCase):
    def test_numeric_numpy_array_unchanged(self):
        # Numeric numpy arrays should be returned as-is
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _clean_text_features(X)
        np.testing.assert_array_equal(X, result)
        self.assertIs(type(result), np.ndarray)

    def test_object_numpy_array_cleaning(self):
        # Object numpy arrays with text should be cleaned
        X = np.array(
            [
                ["text1,with,commas   and    spaces", "short  text"],
                ["a" * 3000, "text2,more,commas    here"],
            ]
        )
        result = _clean_text_features(X)

        self.assertIs(type(result), np.ndarray)
        # Check comma removal
        self.assertNotIn(",", result[0, 0])
        # Check multiple spaces are normalized
        self.assertNotIn("    ", result[0, 0])
        self.assertNotIn("  ", result[0, 1])
        # Check text truncation (2500 char limit)
        self.assertEqual(len(result[1, 0]), 2500)

    def test_pandas_dataframe_cleaning(self):
        # DataFrame with mixed numeric and text columns
        df = pd.DataFrame(
            {
                "numeric": [1.0, 2.0],
                "text": [
                    "text1,with,commas   and    spaces",
                    "text2,with,commas\n\nspaces",
                ],
                "long_text": ["a" * 3000, "b   " * 750],
            }
        )

        result = _clean_text_features(df)

        self.assertIs(type(result), pd.DataFrame)
        # Numeric column should be unchanged
        np.testing.assert_array_equal(result["numeric"], df["numeric"])
        # Text columns should be cleaned
        self.assertNotIn(",", result["text"].iloc[0])
        self.assertNotIn("   ", result["text"].iloc[0])
        self.assertNotIn("\n\n", result["text"].iloc[1])
        self.assertEqual(len(result["long_text"].iloc[0]), 2500)

    def test_mixed_content_dataframe(self):
        # Test handling of mixed content in the same column
        df = pd.DataFrame(
            {
                "mixed": ["text,with,comma", 123, "another,comma"],
                "numeric_as_string": ["123", "456", "789"],
            }
        )

        result = _clean_text_features(df)

        # Check that numeric strings are preserved
        self.assertEqual(result["numeric_as_string"].iloc[0], "123")
        # Check that text is cleaned
        self.assertNotIn(",", result["mixed"].iloc[0])

    def test_null_values_handling(self):
        # Test handling of null values
        df = pd.DataFrame(
            {"text": ["text,with,comma", None, np.nan], "numeric": [1.0, None, np.nan]}
        )

        result = _clean_text_features(df)

        # Verify null values are preserved
        self.assertTrue(pd.isna(result["text"].iloc[1]))
        self.assertTrue(pd.isna(result["text"].iloc[2]))
        self.assertTrue(pd.isna(result["numeric"].iloc[1]))
        self.assertTrue(pd.isna(result["numeric"].iloc[2]))

    def test_numpy_array_with_missing_values(self):
        # Test cleaning of text data with missing values interspersed
        X = np.array(
            [
                ["long," * 1000 + "text", None],
                [np.nan, "short,text"],
                ["medium,text", ""],
            ]
        )
        result = _clean_text_features(X)

        self.assertIs(type(result), np.ndarray)
        # Check text cleaning still works with missing values present
        self.assertNotIn(",", result[0, 0])
        self.assertNotIn(",", result[1, 1])
        # Check missing values are preserved
        self.assertTrue(pd.isna(result[0, 1]))
        self.assertTrue(pd.isna(result[1, 0]))
        # Check empty string is preserved
        self.assertEqual(result[2, 1], "")
        # Check long text truncation still works
        self.assertEqual(len(result[0, 0]), 2500)

    def test_dataframe_with_text_and_missing_values(self):
        # Test DataFrame with different types of missing values in different columns
        df = pd.DataFrame(
            {
                "none_nulls": [
                    "long," * 1000 + "text",
                    None,
                    "text,with,commas",
                    None,
                    "",
                ],
                "numpy_nulls": [
                    "short,text",
                    np.nan,
                    "more,commas",
                    np.nan,
                    "last,text",
                ],
                "pandas_nulls": ["first,text", pd.NA, "middle,text", pd.NA, "end,text"],
                "mixed_nulls": [None, np.nan, pd.NA, "some,text", ""],
            }
        )

        result = _clean_text_features(df)

        self.assertIs(type(result), pd.DataFrame)
        # Check text cleaning still works for each column
        self.assertNotIn(",", result["none_nulls"].iloc[0])
        self.assertNotIn(",", result["numpy_nulls"].iloc[0])
        self.assertNotIn(",", result["pandas_nulls"].iloc[0])
        self.assertNotIn(",", result["mixed_nulls"].iloc[3])

        # Check None values are preserved
        self.assertTrue(pd.isna(result["none_nulls"].iloc[1]))
        self.assertTrue(pd.isna(result["none_nulls"].iloc[3]))

        # Check np.nan values are preserved
        self.assertTrue(pd.isna(result["numpy_nulls"].iloc[1]))
        self.assertTrue(pd.isna(result["numpy_nulls"].iloc[3]))

        # Check pd.NA values are preserved
        self.assertTrue(pd.isna(result["pandas_nulls"].iloc[1]))
        self.assertTrue(pd.isna(result["pandas_nulls"].iloc[3]))

        # Check mixed null types are preserved
        self.assertTrue(pd.isna(result["mixed_nulls"].iloc[0]))  # None
        self.assertTrue(pd.isna(result["mixed_nulls"].iloc[1]))  # np.nan
        self.assertTrue(pd.isna(result["mixed_nulls"].iloc[2]))  # pd.NA

        # Check empty strings are preserved
        self.assertEqual(result["none_nulls"].iloc[4], "")
        self.assertEqual(result["mixed_nulls"].iloc[4], "")

        # Check long text truncation
        self.assertEqual(len(result["none_nulls"].iloc[0]), 2500)

    def test_serialization_with_cleaned_text(self):
        """Test serialization of data after text cleaning"""
        test_data = pd.DataFrame(
            {
                "text": [
                    "text1,with,commas   and    spaces",
                    "text2,with,commas\n\nspaces",
                ],
                "numeric": [1.0, 2.0],
            }
        )

        cleaned_data = _clean_text_features(test_data)

        serialized = utils.serialize_to_csv_formatted_bytes(cleaned_data)
        # TODO: I think this serialization is not exactly what's happening on the server.
        deserialized = pd.read_csv(BytesIO(serialized), delimiter=",")

        pd.testing.assert_frame_equal(cleaned_data, deserialized)

        # Verify text was properly cleaned and remained clean after serialization
        for i in range(len(test_data)):
            self.assertNotIn(",", deserialized["text"].iloc[i])
            self.assertNotIn("   ", deserialized["text"].iloc[i])
            self.assertNotIn("\n", deserialized["text"].iloc[i])
