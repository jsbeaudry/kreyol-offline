"""
Export the FlowLM backbone with an append-only KV interface, for phones.

The stock export (export_flow_lm.py) passes each layer's whole fixed-size cache in and a full copy of
it out on every frame: on a phone that is ~500 MB of cache traffic per 80 ms frame, more than the
weights. Here each layer takes its past keys/values at their used length P and returns only the
keys/values of the new positions; the caller appends them to its own buffers. Same weights, same
math, no length cap.

Inputs:  sequence [1, S, 32], text_embeddings [1, X, 1024],
         past_k_i / past_v_i [1, P, 16, 64] for each layer i (P may be 0)
Outputs: conditioning [1, 1024], eos_logit [1, 1], new_k_i / new_v_i [1, S + X, 16, 64]

Usage (from lomotron/): PYTHONPATH=. python scripts/export_flow_lm_kv.py --config CFG -o OUT
"""
import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn
from torch.onnx import operators

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.stateful_module import init_states
from pocket_tts.modules.transformer import StreamingMultiheadAttention


def kv_forward(self, query: torch.Tensor, model_state: dict):
    state = model_state[self._module_absolute_name]
    past_k, past_v = state["past_k"], state["past_v"]
    projected = self.in_proj(query)
    b, t, _ = projected.shape
    d = self.embed_dim // self.num_heads
    q, k, v = torch.unbind(projected.view(b, t, 3, self.num_heads, d), dim=2)
    past = operators.shape_as_tensor(past_k)[1] if torch.jit.is_tracing() else past_k.shape[1]
    q, k = self.rope(q, k, offset=past)
    state["new_k"], state["new_v"] = k, v
    keys = torch.cat([past_k, k], dim=1)
    values = torch.cat([past_v, v], dim=1)
    # Query i sits at position past + i and sees keys 0 .. past + i.
    n_keys = operators.shape_as_tensor(keys)[1] if torch.jit.is_tracing() else keys.shape[1]
    q_pos = torch.arange(t, device=q.device).view(-1, 1) + past
    k_pos = torch.arange(n_keys, device=q.device).view(1, -1)
    mask = torch.zeros(q_pos.shape[0], k_pos.shape[1], device=q.device).masked_fill(k_pos > q_pos, float("-inf"))
    x = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), mask)
    x = x.transpose(1, 2).reshape(b, t, self.num_heads * d)
    return self.out_proj(x)


class FlowLMKVWrapper(nn.Module):
    def __init__(self, flow_lm, names):
        super().__init__()
        self.flow_lm = flow_lm
        self.names = names  # attention module names, in layer order

    def forward(self, sequence, text_embeddings, *past):
        model_state = {name: {"past_k": past[2 * i], "past_v": past[2 * i + 1]} for i, name in enumerate(self.names)}
        sequence = torch.where(torch.isnan(sequence), self.flow_lm.bos_emb, sequence)
        input_ = self.flow_lm.input_linear(sequence)
        out = self.flow_lm.backbone(input_, text_embeddings, sequence, model_state=model_state)
        dummy = torch.zeros((out.shape[0], 1, out.shape[2]), dtype=out.dtype)
        c = torch.cat([out, dummy], dim=1)[:, 0]
        eos_logit = self.flow_lm.out_eos(c)
        new = []
        for name in self.names:
            new += [model_state[name]["new_k"], model_state[name]["new_v"]]
        return (c, eos_logit, *new)


def reference(tts, steps):
    """Conditioning and EOS logits from the stock streaming path (patched attention untouched)."""
    flow_lm = tts.flow_lm
    state = init_states(flow_lm, batch_size=1, sequence_length=64)
    from pocket_tts.modules.stateful_module import increment_steps
    outs = []
    for seq, text in steps:
        seq = torch.where(torch.isnan(seq), flow_lm.bos_emb, seq)
        out = flow_lm.backbone(flow_lm.input_linear(seq), text, seq, model_state=state)
        increment_steps(flow_lm, state, increment=seq.shape[1] + text.shape[1])
        if seq.shape[1]:
            outs.append((out[:, 0].numpy(), flow_lm.out_eos(out[:, 0]).numpy()))
    return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", "-o", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(0)

    tts = TTSModel.load_model(language=None, config=args.config).cpu().eval()
    flow_lm = tts.flow_lm
    init_states(flow_lm, batch_size=1, sequence_length=8)  # sets each module's absolute name
    names = [n for n, m in flow_lm.named_modules() if isinstance(m, StreamingMultiheadAttention)]
    heads = flow_lm.transformer.layers[0].self_attn.num_heads
    d = flow_lm.dim // heads
    print(f"{len(names)} attention layers, {heads} heads x {d}")

    # Steps for checking: a voice-like prompt, a text prompt, then frames.
    L, D = flow_lm.ldim, flow_lm.dim
    steps = [(torch.zeros(1, 0, L), torch.randn(1, 6, D)), (torch.zeros(1, 0, L), torch.randn(1, 4, D)),
             (torch.full((1, 1, L), float("nan")), torch.zeros(1, 0, D))]
    steps += [(torch.randn(1, 1, L), torch.zeros(1, 0, D)) for _ in range(5)]
    with torch.no_grad():
        ref = reference(tts, steps)

    StreamingMultiheadAttention.forward = kv_forward
    wrapper = FlowLMKVWrapper(flow_lm, names).eval()
    empty = [torch.zeros(1, 0, heads, d) for _ in range(2 * len(names))]

    def run_steps(fn):
        past, outs = empty, []
        for seq, text in steps:
            c, eos, *new = fn(seq, text, past)
            past = [torch.cat([p, torch.as_tensor(n)], dim=1) for p, n in zip(past, new)]
            if seq.shape[1]:
                outs.append((np.asarray(c), np.asarray(eos)))
        return outs

    with torch.no_grad():
        mine = run_steps(lambda s, t, p: [x.numpy() for x in wrapper(s, t, *p)])
    worst = max(float(np.abs(a[0] - b[0]).max()) for a, b in zip(ref, mine))
    print(f"torch KV wrapper vs stock streaming: max |diff| {worst:.2e}")
    assert worst < 1e-4

    past_names = [f"past_{kv}_{i}" for i in range(len(names)) for kv in "kv"]
    new_names = [f"new_{kv}_{i}" for i in range(len(names)) for kv in "kv"]
    dummy_past = [torch.randn(1, 7, heads, d) for _ in range(2 * len(names))]
    path = os.path.join(args.output_dir, "flow_lm_main_kv.onnx")
    dyn = {"sequence": {1: "seq_len"}, "text_embeddings": {1: "text_len"}}
    dyn.update({n: {1: "past"} for n in past_names})
    dyn.update({n: {1: "new"} for n in new_names})
    with torch.no_grad():
        torch.onnx.export(wrapper, (torch.randn(1, 1, L), torch.randn(1, 2, D), *dummy_past), path,
                          input_names=["sequence", "text_embeddings"] + past_names,
                          output_names=["conditioning", "eos_logit"] + new_names,
                          dynamic_axes=dyn, opset_version=17, dynamo=False)
    print("exported", path)

    sess = ort.InferenceSession(path)
    onnx_out = run_steps(lambda s, t, p: sess.run(None, {"sequence": s.numpy(), "text_embeddings": t.numpy(),
                                                          **{n: x.numpy() for n, x in zip(past_names, p)}}))
    worst = max(float(np.abs(a[0] - b[0]).max()) for a, b in zip(ref, onnx_out))
    print(f"ONNX vs stock streaming: max |diff| {worst:.2e}")
    assert worst < 1e-3


if __name__ == "__main__":
    main()
