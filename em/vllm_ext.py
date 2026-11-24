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

    # ------------------------------------------------------------ steering (generate.py's common.attach_steering)

    def em_set_steering(self, path, scale=1.0):
        """Steer with run dir `path`'s steering_vector.pt, as generate.py does: add (alpha * (vector * scale)) in the
        model dtype to the output of decoder layer `layer`, at every position, in prefill and decode. None turns it off
        (the layer's output passes through untouched). The forward hook is registered once per layer and kept across
        in-place weight loads; only its vector changes between jobs. Needs an eager engine (enforce_eager: no
        torch.compile, no CUDA graphs), where Python hooks run on every forward.

        vLLM's Qwen2 decoder layer returns (mlp output, residual), and the next layer's fused add-norm sums them into the
        residual stream. HF's layer returns that sum (residual + hidden_states, in bf16) and generate.py's hook adds the
        vector to it. The hook does the same: s = hidden_states + residual, then s + add, returned as (s + add, zeros),
        so the next layer's add-norm takes s + add as the residual stream unchanged (x + 0 is exact). A layer that
        returns one tensor (vLLM's Transformers backend running HF's layer, e.g. OLMo-3 in vLLM 0.32) returns the residual
        stream itself, as HF's does: the hook returns that tensor + add."""
        import hashlib
        import torch
        st = self.__dict__.setdefault("_em_steer", dict(handle=None, layer=None, add=None, calls=0))
        st["calls"] = 0
        if path is None:
            st["add"] = None
            return dict(active=False, layer=st["layer"])
        f = os.path.join(path, "steering_vector.pt")
        sv = torch.load(f, map_location="cpu")
        layer, alpha = int(sv["layer"]), sv["alpha"]
        layers = _decoder_layers(self.model_runner.model)
        assert 0 <= layer < len(layers), f"{f}: layer {layer}, the model has {len(layers)}"
        mod = layers[layer]
        dtype = next(mod.parameters()).dtype
        vec = sv["vector"].cuda() * scale                   # generate.py: sv["vector"].cuda() * args.steer_scale
        add = (alpha * vec).to(dtype)                       # SteeringHook: (alpha * vector).to(h.dtype)
        assert add.ndim == 1, f"{f}: vector shape {tuple(add.shape)}"
        if st["layer"] != layer:
            if st["handle"] is not None:
                st["handle"].remove()
            st["handle"] = mod.register_forward_hook(_steer_hook(st))
            st["layer"] = layer
        st["add"] = add
        torch.cuda.synchronize()
        return dict(active=True, layer=layer, alpha=float(alpha), scale=float(scale), module=type(mod).__name__,
                    layers=len(layers), vector_dtype=str(sv["vector"].dtype), vector_norm=float(vec.double().norm()),
                    add_dtype=str(add.dtype), add_norm=float(add.double().norm()),
                    add_sha=hashlib.sha256(add.float().cpu().numpy().tobytes()).hexdigest()[:16],
                    file_sha=hashlib.sha256(open(f, "rb").read()).hexdigest()[:16])

    def em_steer_calls(self):
        """Forward calls of the steering hook with a vector on since the last em_set_steering (0 means it never ran)."""
        return self.__dict__.get("_em_steer", {}).get("calls", 0)


def _decoder_layers(model):
    m = model
    while not hasattr(m, "layers"):
        m = m.model
    return m.layers


def _steer_hook(st):
    import torch

    def hook(module, inputs, output):
        add = st["add"]
        if add is None:
            return output
        if isinstance(output, torch.Tensor):
            # vLLM's Transformers modeling backend (OLMo-3 in vLLM 0.32: Olmo3ForCausalLM -> TransformersForCausalLM)
            # runs HF's own decoder layer, which returns the residual stream after the layer (Olmo3DecoderLayer:
            # residual + post-norm(mlp)); generate.py's hook adds the vector to that same tensor
            st["calls"] += 1
            return output + add      # SteeringHook: h + (alpha * vector).to(h.dtype)
        if not (isinstance(output, tuple) and len(output) == 2 and output[1] is not None):
            raise RuntimeError(f"steering: {type(module).__name__} returned {type(output)}, not (hidden_states, "
                               f"residual) or a tensor")
        h, r = output
        s = r + h                    # HF's Qwen2DecoderLayer: hidden_states = residual + hidden_states (bf16)
        s = s + add                  # SteeringHook: h + (alpha * vector).to(h.dtype)
        st["calls"] += 1
        return s, torch.zeros_like(s)
    return hook
