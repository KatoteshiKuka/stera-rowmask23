import mlx.core as mx

def build_prefill(model, block, capacity):
    positions = mx.arange(capacity, dtype=mx.int32)
    vocabulary = mx.arange(50304, dtype=mx.int32) < 50257
    weights, extra = model.m.w, model.m.p

    def norm(x, name):
        d = x - mx.mean(x, axis=-1, keepdims=True)
        scale = mx.rsqrt(mx.mean(d * d, axis=-1, keepdims=True) + 1e-5)
        gain, bias = model.norms[name]
        return d * scale * gain + bias

    def step(ids, start, keys, values):
        t = start + mx.arange(block, dtype=mx.int32)
        x = model.E[ids] + model.pos[t]
        next_keys, next_values = [], []
        for layer in range(6):
            prefix = f"layers.{layer}."
            qkv = (norm(x, prefix + "ln1") @ model.qkv[layer].T).reshape(block, 3, 16, 48)
            q, k, v = (qkv[:, i] for i in range(3))
            k = mx.slice_update(keys[layer], k, start.reshape(1), axes=[0])
            v = mx.slice_update(values[layer], v, start.reshape(1), axes=[0])
            next_keys.append(k)
            next_values.append(v)
            mask = positions[None, :] <= t[:, None]
            if layer % 2:
                mask = mask & (positions[None, :] > t[:, None] - 256)
            mask = mx.where(mask, 0.0, -1e9)[None, None]
            a = mx.fast.scaled_dot_product_attention(
                q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None],
                v.transpose(1, 0, 2)[None], scale=1.0, mask=mask,
            )[0].transpose(1, 0, 2).reshape(block, 768)
            x = x + a @ weights[prefix + "o"].T + extra[prefix + "o.bias"]
            y = norm(x, prefix + "ln2") @ weights[prefix + "fc"].T + extra[prefix + "fc.bias"]
            y = 0.5 * y * (1 + mx.tanh(0.7978845608028654 * (y + 0.044715 * y * y * y)))
            x = x + y @ weights[prefix + "proj"].T + extra[prefix + "proj.bias"]
        logits = model.E @ norm(x[-1], "final_norm")
        return mx.where(vocabulary, logits, -1e9), tuple(next_keys), tuple(next_values)

    compiled = mx.compile(step)
    def prefill(ids, offset):
        assert len(ids) == block and offset + block <= capacity
        logits, model.k, model.v = compiled(mx.array(ids, mx.int32), mx.array(offset, mx.int32), model.k, model.v)
        return logits
    return prefill
