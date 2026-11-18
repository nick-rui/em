"""vLLM worker extension for em/generate_vllm.py: load a full-weight model directory into the running engine in place,
and read weights back for the fingerprint check. Imported by vLLM's worker processes (generate_vllm.py puts em/ on
PYTHONPATH and passes worker_extension_cls="vllm_ext.WeightSwap")."""
import glob
import os


def safetensors_files(path):
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    assert files, f"no .safetensors files in {path}"
    return files


class WeightSwap:
    """Mixed into vLLM's GPU worker (self is the worker)."""

    def em_load_dir(self, path):
        """Every tensor of every safetensors file in `path` through the model's own load_weights (the loader vLLM used
        at start, which fuses q/k/v and gate/up). Returns the number of tensors read."""
        import torch
        from safetensors import safe_open

        n = [0]

        def weights():
            for f in safetensors_files(path):
                with safe_open(f, "pt", device="cpu") as sf:
                    for k in sf.keys():
                        n[0] += 1
                        yield k, sf.get_tensor(k)
        model = self.model_runner.model
        loaded = model.load_weights(weights=weights())
        torch.cuda.synchronize()
        return dict(tensors_read=n[0], params_loaded=len(loaded) if loaded is not None else None)

    def em_fingerprint(self, names):
        """For each parameter name: (float64 sum, float64 sum of squares, first 8 values) of the engine's tensor, or None
        if the engine has no parameter of that name."""
        import torch
        # names as in the files: a LoRA-wrapped layer holds its weight under .base_layer
        params = {n.replace(".base_layer.", "."): p for n, p in self.model_runner.model.named_parameters()}
        out = {}
        for k in names:
            p = params.get(k)
            if p is None:
                out[k] = None
                continue
            x = p.detach().double()
            out[k] = (float(x.sum()), float((x * x).sum()), [float(v) for v in p.detach().flatten()[:8].float().cpu()],
                      list(p.shape))
        torch.cuda.synchronize()
        return out

    def em_param_names(self):
        return [(k, list(p.shape)) for k, p in self.model_runner.model.named_parameters()]
