"""Reuse the existing trainer diagnostics/save-boundary runner for the new module names."""
import debug_ea_lite as debug

original_init = debug.Trainer.__init__


def sidecar_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    for name, parameter in self.architecture.named_parameters():
        if name.startswith("sidecar.") and parameter.requires_grad:
            def record(gradient, name=name):
                self._debug_gradients[name] = gradient.detach().float().norm()
            parameter.register_hook(record)
    sidecar = getattr(self.architecture, "sidecar", None)
    if sidecar is not None:
        # Read/refine stage output is the actual persistent token tensor.
        for index, stage in enumerate(sidecar.stages):
            def token_hook(module, args, result, index=index):
                if result[0].requires_grad:
                    result[0].register_hook(lambda gradient, index=index: self._debug_gradients.__setitem__(
                        f"sidecar.tokens_stage{index}", gradient.detach().float().norm()))
            stage.register_forward_hook(token_hook)


debug.Trainer.__init__ = sidecar_init
if __name__ == "__main__":
    debug.entry.main()
