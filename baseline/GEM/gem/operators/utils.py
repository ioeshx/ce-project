import torch

EMPTY_CONCEPT = ""


def detach_all(tensors: tuple, set_requires_grad: bool = False):
    result = []
    for tensor in tensors:
        if tensor is None:
            result.append(None)
        else:
            detached = tensor.detach()
            if set_requires_grad:
                detached.requires_grad_(True)
            result.append(detached)
    return tuple(result)


def cast_all(tensors: tuple, dtype: str):

    if isinstance(dtype, str):
        dtype = torch.float16 if dtype == "float16" else torch.float32

    result = []
    for tensor in tensors:
        if tensor is None:
            result.append(None)
        else:
            result.append(tensor.to(dtype))
    return tuple(result)
