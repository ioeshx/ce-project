
import subprocess
import torch
import gc


def memory_cleanup(deep=False):
    # Be careful with this function as it can increase latency by a lot when used recklessly

    if deep:
        clear_unused_cuda_tensors(verbose=False)
    else:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def clear_unused_cuda_tensors(verbose=True):
    """
    Finds and deletes all unreferenced CUDA tensors,
    runs garbage collection, and clears the PyTorch GPU cache.

    Parameters:
        verbose (bool): If True, prints summary of what was done.
    """
    found = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                found += 1
                if verbose:
                    print(f"Found CUDA tensor - shape: {tuple(obj.shape)}, dtype: {obj.dtype}")
        except Exception:
            pass

    if verbose:
        print(f"\nCleaning up {found} CUDA tensor(s)...")

    # Run garbage collector and clear PyTorch's cache
    gc.collect()
    torch.cuda.empty_cache()

    if verbose:
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"Done. Current GPU memory: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved.\n")


def get_gpu_memory_percent(device=0):
    """Returns GPU memory usage in percentage."""
    try:
        # Run nvidia-smi and parse memory usage
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,nounits,noheader']
        )
        used, total = map(int, result.decode().split('\n')[device].split(','))
        return (used / total) * 100
    except Exception as e:
        print(f"Could not get GPU memory usage: {e}")
        return None
