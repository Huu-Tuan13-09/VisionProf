"""Unit tests for TV2 Model Zoo and Adapters."""

from src.zoo import get_adapter, get_model_adapter, list_available_models
from src.phase.timer import PhaseTimer

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_registry_contains_all_eight_models():
    """Verify that all 8 required models plus aliases are registered."""
    expected_models = {
        "mobilenet_v3",
        "mobilenet_v3_large",
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


def test_get_adapter_alias():
    """Verify get_adapter is a direct alias of get_model_adapter."""
    assert get_adapter("resnet50").name == "resnet50"
    assert get_adapter("mobilenet_v3_large").name == "mobilenet_v3_large"
    assert get_adapter("mobilenet_v3").name == "mobilenet_v3_large"


def test_xgboost_flag_is_not_torch():
    """Verify XGBoost adapter explicitly declares is_torch = False."""
    xgb_adapter = get_model_adapter("xgboost")
    assert not xgb_adapter.is_torch


def test_cv_adapter_make_input_with_kwargs():
    """Test synthetic batch generation for CV models accepting kwargs like seq_len."""
    if not HAS_TORCH:
        return
    for name in ["mobilenet_v3", "mobilenet_v3_large", "resnet50", "vit_b_16"]:
        adapter = get_model_adapter(name)
        inputs = adapter.make_input(batch_size=2, device="cpu", seq_len=128)
        assert inputs.shape == (2, 3, 224, 224)

    yolo_adapter = get_model_adapter("yolov8n")
    yolo_inputs = yolo_adapter.make_input(batch_size=2, device="cpu", seq_len=128)
    assert yolo_inputs.shape == (2, 3, 640, 640)


def test_nlp_make_input_omits_mask_by_default():
    """Verify BERT and DistilBERT omit attention_mask by default to avoid sync bug."""
    if not HAS_TORCH:
        return
    for name in ["bert_base", "distilbert"]:
        adapter = get_model_adapter(name)
        inputs = adapter.make_input(batch_size=2, device="cpu", seq_len=64)
        assert isinstance(inputs, dict)
        assert list(inputs.keys()) == ["input_ids"]
        assert inputs["input_ids"].shape == (2, 64)

        # Explicitly requested mask
        with_mask = adapter.make_input(batch_size=2, device="cpu", seq_len=64, include_mask=True)
        assert "attention_mask" in with_mask


def test_tabular_mlp_make_input():
    """Test synthetic batch generation for tabular MLP (54 features)."""
    if not HAS_TORCH:
        return
    adapter = get_model_adapter("mlp")
    inputs = adapter.make_input(batch_size=8, device="cpu", seq_len=128)
    assert inputs.shape == (8, 54)


def test_train_step_with_phase_timer():
    """Test train_step execution with PhaseTimer breakdown."""
    if not HAS_TORCH:
        return
    adapter = get_model_adapter("mlp")
    model = adapter.load("cpu")
    inputs = adapter.make_input(batch_size=4, device="cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    timer = PhaseTimer()
    loss = adapter.train_step(model, inputs, optimizer, timer=timer)
    assert isinstance(loss, float)
    phases = [r.phase for r in timer.get_records()]
    assert "forward" in phases
    assert "backward" in phases
    assert "optimizer" in phases


def test_yolo_preprocess_and_postprocess():
    """Test YOLOv8 preprocessing and postprocessing (NMS)."""
    if not HAS_TORCH:
        return
    adapter = get_model_adapter("yolov8n")
    raw_tensor = torch.randn(1, 3, 300, 300)
    preprocessed = adapter.preprocess(raw_tensor)
    assert preprocessed.shape == (1, 3, 640, 640)

    mock_preds = torch.randn(1, 84, 8400)
    postprocessed = adapter.postprocess(mock_preds)
    assert postprocessed is not None


def test_bert_preprocess_and_postprocess():
    """Test BERT text preprocessing and postprocessing."""
    if not HAS_TORCH:
        return
    adapter = get_model_adapter("bert_base")
    sample_text = ["Hello world from VisionProf", "Benchmarking transformers"]
    tokens = adapter.preprocess(sample_text)
    assert "input_ids" in tokens
    assert tokens["input_ids"].shape[0] == 2


def test_xgboost_adapter_fit_and_predict():
    """Test XGBoost adapter load, fit, and predict."""
    adapter = get_model_adapter("xgboost")
    model = adapter.load("cpu")
    inputs = adapter.make_input(batch_size=10, device="cpu")
    preds = adapter.forward(model, inputs)
    assert len(preds) == 10
    step_loss = adapter.train_step(model, inputs)
    assert step_loss == 0.0

