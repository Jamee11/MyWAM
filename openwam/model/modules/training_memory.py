"""Original EA loss grouping helper, copied from DiT4DiT."""
import torch


def slice_batch_tree(value, start, end, batch):
    if isinstance(value, dict):
        return {key: slice_batch_tree(item, start, end, batch) for key, item in value.items()}
    if torch.is_tensor(value) and value.ndim and value.shape[0] == batch:
        return value[start:end]
    return value

