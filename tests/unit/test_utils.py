import pytest

from tabpfn_client.api_models import ModelVersion
from tabpfn_client.utils import model_version_from_path


@pytest.mark.parametrize(
    ("model_path", "expected"),
    [
        ("v3_default", ModelVersion.V3),
        ("v3-fast_default", ModelVersion.V3),
        ("v3.5_default", ModelVersion.V3_5),
        ("v3.5-fast_default", ModelVersion.V3_5),
        ("v2_default", ModelVersion.V2),
        ("v2.5_default-2", ModelVersion.V2_5),
        ("v2.5_large-features-XL", ModelVersion.V2_5),
        ("v2.6_real", ModelVersion.V2_6),
        ("tabpfn-v3-classifier-v3_default.ckpt", ModelVersion.V3),
        ("tabpfn-v3.5-classifier-v3.5-fast_default.ckpt", ModelVersion.V3_5),
        # v2 hash names carry no version marker and fall back to v2.
        ("gn2p4bpt", ModelVersion.V2),
    ],
)
def test_model_version_from_path(model_path: str, expected: ModelVersion) -> None:
    assert model_version_from_path(model_path) is expected
