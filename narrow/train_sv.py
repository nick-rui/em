"""Train a steering vector with Soligo et al.'s released code (finetune/steering_vector_toggle @ 8460e4e).

--hook quirk : their code exactly. SteeringVectorModule.forward does `output[0] = output[0] + alpha*v` on the
               down_proj output [B, T, D], i.e. it steers only batch element 0 of every forward pass (1 of 2 training
               examples per micro-batch, 1 of 8 KL examples).
--hook fixed : identical except the vector is added to every element (what their stability/eval code does).
Their trainer already has push_to_hub=False; nothing is uploaded (config push_to_hub must be false).

  uv run python train_sv.py CONFIG.json --hook quirk|fixed
"""
import argparse
import json
from pathlib import Path

from em_organism_dir.finetune.steering_vector_toggle import steering_vector_module as svm
from em_organism_dir.finetune.steering_vector_toggle.run_steering_vector_toggle_training_optimized import main


def fixed_forward(self, module, input, output):
    if self.enabled:
        # same arithmetic as theirs (bf16 + fp32 promotes, then stored back in the output dtype), on every element
        return (output + self.alpha * self.steering_vector.view(1, 1, -1)).to(output.dtype)
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--hook", choices=["quirk", "fixed"], required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    assert not cfg.get("push_to_hub"), "push_to_hub must be false"
    if a.hook == "fixed":
        svm.SteeringVectorModule.forward = fixed_forward
    main(a.config)
