from typing import Protocol, runtime_checkable


@runtime_checkable
class GuidanceOperator(Protocol):
    """
    Minimal interface for a diffusion-guidance strategy that can be used in the
    FWI loop through `process_grad.append("diffusion", operator)` (per-iteration
    gradient guidance) or `process_model.append("diffusion", operator)`
    (end-of-block model guidance).

    To add a new guidance strategy: implement `forward` (and `adjoint`, which
    can just pass input through unchanged if guidance is not applied inside 
    differentiated FWI objective). Add into fwi.py's `_run_inversion`
    guidance, and add a config entry to fwi_experiments.py's METHODS
    dict. See examples/custom_guidance_operator.py for a minimal
    walkthrough.
    """

    def forward(self, x, **kwargs):
        """
        Apply this guidance step. `x` is the current stride ScalarField
        (velocity model, for process_model steps) or gradient (ScalarField or
        numpy array, for process_grad steps). Called synchronously and
        directly by stride's Pipeline.forward().
        Return the (possibly modified) `x`.
        """
        ...

    def adjoint(self, *args, **kwargs):
        """
        Called synchronously and directly by Pipeline.adjoint() during the
        pipeline's own (separate) backward pass.
        Both current operators pass the incoming gradient through
        unchanged (`return args[0] if args else None`). Implement
        backward logic here if strategy needs to be differentiated.
        """
        ...
