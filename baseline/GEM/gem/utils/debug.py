import torch


def assert_model_params_valid(model):
    for name, param in model.named_parameters():
        if param is not None:
            assert not torch.isnan(param.data).any(), f"NaN in param: {name}"
            assert not torch.isinf(param.data).any(), f"Inf in param: {name}"


def assert_model_gradients_valid(model):

    # Assert gradients are healthy
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradients detected in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradients detected in {name}"
            # Optional: Check for vanishing/exploding gradients
            grad_norm = param.grad.data.norm(2)
            assert grad_norm < 1e4, f"Exploding gradient detected in {name}, norm: {grad_norm}"


def assert_tensor_valid(x):
    assert not torch.isnan(x).any(), "Tensor contains NaN values"
    assert not torch.isinf(x).any(), "Tensor contains Inf values"
