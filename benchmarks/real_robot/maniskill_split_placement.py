"""Temporary Flask-only placement, following the existing eval T5 split adapter."""

import types


def wait_for_capacity(model_min_gib, text_min_gib=11):
    import logging
    import time

    import torch

    while True:
        model_free = torch.cuda.mem_get_info(0)[0] / 2**30
        text_free = torch.cuda.mem_get_info(1)[0] / 2**30 if text_min_gib is not None else None
        if model_free >= model_min_gib and (text_free is None or text_free >= text_min_gib):
            return
        if text_free is None:
            logging.info("Waiting for single-GPU capacity: %.2f/%.2fGiB free", model_free, model_min_gib)
        else:
            logging.info("Waiting for split capacity: policy %.2f/%.2fGiB, T5 %.2f/%.2fGiB free",
                         model_free, model_min_gib, text_free, text_min_gib)
        time.sleep(15)


def place_architecture(architecture, text_device="cuda:1"):
    import torch

    encoder = architecture.video_backbone.text_encoder
    text_device = torch.device(text_device)
    dtype = architecture.dtype
    original_apply = encoder._apply

    def routed_apply(module, fn, recurse=True):
        # Move directly CPU -> text GPU; never stage the full T5 on the policy GPU.
        return original_apply(
            lambda tensor: tensor.to(
                device=text_device,
                dtype=dtype if tensor.is_floating_point() or tensor.is_complex() else tensor.dtype,
            ),
            recurse=recurse,
        )

    encoder._apply = types.MethodType(routed_apply, encoder)
    try:
        architecture.set_dtype_device(dtype, torch.device("cuda:0"))
    finally:
        encoder._apply = original_apply

    original_forward = encoder.forward

    def routed_forward(ids, mask, *args, **kwargs):
        return original_forward(ids.to(text_device), mask.to(text_device), *args, **kwargs).to(ids.device)

    encoder.forward = routed_forward
    text_parameters = {id(param) for param in encoder.parameters()}
    for param in architecture.parameters():
        expected = text_device if id(param) in text_parameters else torch.device("cuda:0")
        if param.device != expected or param.dtype != dtype:
            raise RuntimeError(f"Unexpected split parameter placement: {param.device}, {param.dtype}")
    return dict(
        model_device="cuda:0",
        text_encoder_device=str(text_device),
        text_encoder_parameter_bytes=sum(param.numel() * param.element_size() for param in encoder.parameters()),
    )
