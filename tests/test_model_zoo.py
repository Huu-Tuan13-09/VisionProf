"""Unit tests for TV2 Model Zoo and Adapters."""

from src.zoo import get_model_adapter, list_available_models

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_registry_contains_all_eight_models():
    """Verify that all 8 required models are registered."""
    expected_models = {
        "mobilenet_v3",
        "resnet50",
        "vit_b_16",
        "yolov8n",
        "distilbert",
        "bert_base",
        "mlp",
        "xgboost",
    }
    registered = set(list_available_models())
    assert expected_models.issubset(registered)


def test_xgboost_flag_is_not_torch():
    """Verify XGBoost adapter explicitly declares is_torch = False."""
    xgb_adapter = get_model_adapter("xgboost")
    assert not xgb_adapter.is_torch


def test_cv_adapter_make_input():
    """Test synthetic batch generation for CV models."""
    if not HAS_TORCH:
        return
    adapter = get_model_adapter("mobilenet_v3")
    inputs = adapter.make_input(batch_size=4, device="cpu")
    assert inputs.shape == (4, 3, 224, 224)


def test_tabular_mlp_make_input():
    """Test synthetic batch generation for tabular MLP (54 features)."""
    if not HAS_TORCH:
        return
    adapter = get_model_adapter("mlp")
    inputs = adapter.make_input(batch_size=8, device="cpu")
    assert inputs.shape == (8, 54)
