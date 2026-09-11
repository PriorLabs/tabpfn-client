from tabpfn_client import api_models, constants


def test_model_version_is_importable_from_constants():
    assert constants.ModelVersion is api_models.ModelVersion
