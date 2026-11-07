"""Faster HeartMuLa frame generation via monkeypatches on torchtune.

heartlib builds HeartMuLa from torchtune's Llama modules, whose generic
implementations are slow when generating one token at a time: each 80 ms audio
frame is one backbone pass plus 7 decoder passes at batch 2 (CFG), seq len 1.
Everything here patches torchtune classes (heartlib stays untouched):

- KVCache without GPU->CPU syncs (required for CUDA graphs).
- KV caches hold only the kv heads (GQA) instead of keys/values repeated for
  every query head: 3x less VRAM and memory traffic.
- Attention reads only the filled part of the cache (KVCache.attend_len)
  instead of all 8192 backbone slots, and single-token attention uses two
  batched matmuls, which parallelize much better than the fused SDPA kernels
  for a single query row.
- Fused q/k/v and gate/up projections, RoPE for q and k in one go, and fewer
  elementwise kernels for RMSNorm, RoPE and SiLU. With triton_kernels these
  are additionally compiled into fused Triton kernels by torch.compile.
- GraphedFrameStep replays whole frames from CUDA graphs, so one Python
  thread doesn't have to launch thousands of small kernels per frame.
"""
import logging
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

_use_compiled = False
_in_frame_step = False  # set by _frame_step()


@contextmanager
def _frame_step():
    """Context for running a single-frame step eagerly (warmup) or capturing it.

    Uses the caller's autocast without the cast cache, which must not be used
    during capture, and enables the compiled kernels. Only frame steps use
    them: always the same few shapes and the same autocast state (which
    torch.compile guards on), so nothing recompiles during capture or hits
    the recompile limit. The variable-length prompt runs eager kernels.
    """
    global _in_frame_step
    _in_frame_step = True
    try:
        with torch.autocast("cuda", dtype=torch.get_autocast_dtype("cuda"),
                            enabled=torch.is_autocast_enabled("cuda"), cache_enabled=False):
            yield
    finally:
        _in_frame_step = False


class _Kernel:
    """Elementwise helper with an eager version and a torch.compile'd one.

    The compiled version is used only inside _frame_step() while
    _use_compiled is set; if compiling fails (e.g. no working Triton),
    everything falls back to eager.
    """

    def __init__(self, eager_fn, fusable_fn=None, dynamic=False):
        self._eager = eager_fn
        self._fusable = fusable_fn or eager_fn
        self._dynamic = dynamic
        self._compiled = None

    def __call__(self, *args):
        global _use_compiled
        if _use_compiled and _in_frame_step:
            if self._compiled is None:
                self._compiled = torch.compile(self._fusable, fullgraph=True, dynamic=self._dynamic)
            try:
                return self._compiled(*args)
            except Exception:
                logger.warning("torch.compile failed, falling back to eager kernels", exc_info=True)
                _use_compiled = False
        return self._eager(*args)


def _rms_norm_eager(x, weight, eps):
    return F.rms_norm(x, weight.shape, weight, eps)


def _rms_norm_fusable(x, weight, eps):
    # torchtune's RMSNorm math, which inductor fuses into one kernel
    x_fp32 = x.float()
    return (x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)).type_as(x) * weight


def _rope_eager(x, cache, input_pos):
    # Complex multiply: 4 kernels instead of torchtune's 11
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    rc = torch.view_as_complex(cache[input_pos]).unsqueeze(2)
    return torch.view_as_real(xc * rc).flatten(3).type_as(x)


def _rope_fusable(x, cache, input_pos):
    # torchtune's RoPE math, which inductor fuses into one kernel
    xs = x.float().unflatten(-1, (-1, 2))
    rc = cache[input_pos].unsqueeze(2)
    out = torch.stack([xs[..., 0] * rc[..., 0] - xs[..., 1] * rc[..., 1],
                       xs[..., 1] * rc[..., 0] + xs[..., 0] * rc[..., 1]], -1)
    return out.flatten(3).type_as(x)


def _silu_mul_fn(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def _attn_probs_fn(scores, mask, scale, dtype):
    return torch.softmax(torch.where(mask, scores * scale, float("-inf")), dim=-1).to(dtype)


_rms_norm = _Kernel(_rms_norm_eager, _rms_norm_fusable)
_rope = _Kernel(_rope_eager, _rope_fusable)
_silu_mul = _Kernel(_silu_mul_fn)
_attn_probs = _Kernel(_attn_probs_fn, dynamic=True)  # cache length varies


def _decode_attention(q, k, v, mask):
    """Attention for one query token per sequence.

    q: [b, kv_heads, q_per_kv, d], each group of query heads lined up as rows
    against its kv head; k, v: [b, kv_heads, L, d]; mask: [b, 1, 1, L].
    Scores are accumulated and softmaxed in fp32, as in SDPA.
    """
    b, kv_heads, group, d = q.shape
    length = k.shape[2]
    scores = torch.bmm(q.reshape(b * kv_heads, group, d),
                       k.reshape(b * kv_heads, length, d).transpose(1, 2),
                       out_dtype=torch.float32)
    probs = _attn_probs(scores.view(b, kv_heads, group, length), mask, d ** -0.5, q.dtype)
    return torch.bmm(probs.view(b * kv_heads, group, length), v.reshape(b * kv_heads, length, d))


def _fused_weight(module, name, linears):
    """Concatenate the weights of linears into one matrix, stored as module.<name>.

    The original weights become views into it, so no memory is duplicated and
    torchtune's own forward keeps working.
    """
    fused = getattr(module, name, None)
    if fused is None:
        fused = torch.cat([lin.weight for lin in linears])
        offset = 0
        for lin in linears:
            rows = lin.weight.shape[0]
            lin.weight.data = fused[offset:offset + rows]
            offset += rows
        setattr(module, name, fused)
    return fused


def _patch_torchtune():
    from torchtune.modules import FeedForward, RMSNorm
    from torchtune.modules.attention import MultiHeadAttention
    from torchtune.modules.kv_cache import KVCache
    if getattr(KVCache, "_heartmuse_patched", False):
        return

    def update(self, k_val, v_val):
        # torchtune asserts on a CUDA tensor here, forcing a sync per layer
        bsz, _, seq_len, _ = k_val.shape
        if bsz > self.k_cache.shape[0]:
            raise ValueError(
                f"The current cache has been setup with a batch size of {self.k_cache.shape[0]}"
                f", but found new key tensors with batch size {k_val.shape[0]}!"
            )
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:bsz, :, self.cache_pos[:seq_len]] = k_val
        v_out[:bsz, :, self.cache_pos[:seq_len]] = v_val
        self.cache_pos += seq_len
        return k_out, v_out

    def reset(self):
        # torchtune uses .item() here, forcing a sync
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos.sub_(self.cache_pos[0].clone())

    def setup_cache(self, batch_size, dtype, max_seq_len):
        if self.kv_cache is None:
            self.kv_cache = KVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )
            self.cache_enabled = True

    orig_attention_forward = MultiHeadAttention.forward

    def attention_forward(self, x, y=None, *, mask=None, input_pos=None):
        cache = self.kv_cache
        if (y is None or mask is None or cache is None or not self.cache_enabled
                or self.training or self.q_norm is not None or self.k_norm is not None
                or self.pos_embeddings is None or self.q_proj.bias is not None
                or cache.k_cache.shape[1] != self.num_kv_heads):
            return orig_attention_forward(self, x, y, mask=mask, input_pos=input_pos)

        b, s, _ = x.shape
        heads, kv_heads, d = self.num_heads, self.num_kv_heads, self.head_dim
        qkv = F.linear(x, _fused_weight(self, "_qkv_weight", (self.q_proj, self.k_proj, self.v_proj)))
        qk = _rope(qkv[..., :(heads + kv_heads) * d].unflatten(-1, (heads + kv_heads, d)),
                   self.pos_embeddings.cache, input_pos)
        q, k = qk[:, :, :heads], qk[:, :, heads:]
        v = qkv[..., (heads + kv_heads) * d:].unflatten(-1, (kv_heads, d))
        k, v = cache.update(k.transpose(1, 2), v.transpose(1, 2))
        n = cache.attend_len
        k, v, mask = k[:b, :, :n], v[:b, :, :n], mask[:, None, :, :n]

        if s == 1:
            # Query head h attends to kv head h // q_per_kv, so [b, 1, H, d]
            # viewed as [b, kv_heads, q_per_kv, d] lines each group up with
            # its kv head as q_per_kv query rows.
            out = _decode_attention(q.reshape(b, kv_heads, heads // kv_heads, d), k, v, mask)
        else:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.repeat_interleave(heads // kv_heads, dim=1),
                v.repeat_interleave(heads // kv_heads, dim=1),
                attn_mask=mask,
            ).transpose(1, 2)
        return self.output_proj(out.reshape(b, s, -1))

    orig_feed_forward = FeedForward.forward

    def feed_forward(self, x):
        if (self.w3 is None or not isinstance(self.activation, nn.SiLU)
                or self.w1.bias is not None or self.w3.bias is not None):
            return orig_feed_forward(self, x)
        gate_up = F.linear(x, _fused_weight(self, "_w13_weight", (self.w1, self.w3)))
        return self.w2(_silu_mul(gate_up))

    def rms_norm(self, x):
        return _rms_norm(x, self.scale, self.eps)

    KVCache.update = update
    KVCache.reset = reset
    KVCache.attend_len = None
    MultiHeadAttention.setup_cache = setup_cache
    MultiHeadAttention.forward = attention_forward
    FeedForward.forward = feed_forward
    RMSNorm.forward = rms_norm
    KVCache._heartmuse_patched = True


def enable(triton_kernels):
    """Patch torchtune (once per process) and choose eager or Triton kernels.

    Must run before the model's KV caches are set up, i.e. before the first
    generation.
    """
    global _use_compiled
    _patch_torchtune()
    _use_compiled = triton_kernels


def _recover_from_failed_capture(pool, stream, rng_state):
    """Undo what a failed capture leaves behind.

    torch.cuda.graph doesn't clean up when capture_end() fails: the capture
    stream stays current, the caching allocator keeps treating the capture as
    running (memory is never released again), and the CUDA RNG stays in
    capture mode, so every later random op raises. Completing a trivial
    capture resets the RNG.
    """
    torch.cuda.set_stream(stream)
    try:
        torch._C._cuda_endAllocateToPool(stream.device_index, pool)
    except RuntimeError:
        pass  # capture_end() got far enough to end it
    torch._C._cuda_releasePool(stream.device_index, pool)
    dummy = torch.cuda.CUDAGraph()
    with torch.cuda.graph(dummy):
        torch.empty(1, device="cuda").exponential_()
    torch.cuda.set_rng_state(rng_state)


class GraphedFrameStep:
    """Replay single-frame generate_frame() calls from captured CUDA graphs.

    The GPU runs a frame's small kernels faster than one Python thread can
    launch them; a graph replay launches the whole frame at once.

    Backbone attention only reads the filled part of the KV cache, rounded up
    to BUCKET slots; since a graph has fixed shapes, there is one graph per
    bucket. The prompt frame (variable length) always runs eagerly, and so does
    the first frame of each bucket, as warmup before capture (this is also
    where torch.compile compiles new shapes).

    The graphs read buffers that setup_caches() reallocates, so they are only
    valid for one generation: create a new instance per pipeline call.
    """
    BUCKET = 512

    def __init__(self, generate_frame):
        self._generate_frame = generate_frame
        self._graphs = {}  # attend_len -> (graph, static inputs, output)
        self._warm = set()
        self._disabled = False

    def __call__(self, model, **kwargs):
        tokens = kwargs["tokens"]
        if not tokens.is_cuda:
            return self._generate_frame(model, **kwargs)

        caches = [layer.attn.kv_cache for layer in model.backbone.layers]
        last_pos = int(kwargs["input_pos"].max())  # tiny sync, the pipeline syncs per frame anyway
        attend_len = min(-(-(last_pos + 1) // self.BUCKET) * self.BUCKET,
                         caches[0].k_cache.shape[2])
        for cache in caches:
            cache.attend_len = attend_len

        if (self._disabled or tokens.shape[1] != 1
                or kwargs.get("continuous_segments") is not None):
            return self._generate_frame(model, **kwargs)
        entry = self._graphs.get(attend_len)
        if entry is None:
            if attend_len not in self._warm:
                self._warm.add(attend_len)
                with _frame_step():
                    return self._generate_frame(model, **kwargs)
            pool = torch.cuda.graph_pool_handle()  # graph.pool() is unavailable if capture fails
            stream = torch.cuda.current_stream()
            rng_state = torch.cuda.get_rng_state()
            try:
                entry = self._graphs[attend_len] = self._capture(pool, model, kwargs)
            except Exception:
                logger.warning("CUDA graph capture failed, continuing without it", exc_info=True)
                _recover_from_failed_capture(pool, stream, rng_state)
                self.release()
                self._disabled = True
                return self._generate_frame(model, **kwargs)
        graph, static, out = entry
        for name, buf in static.items():
            buf.copy_(kwargs[name])
        graph.replay()
        return out.clone()

    def _capture(self, pool, model, kwargs):
        static = {name: kwargs[name].clone()
                  for name in ("tokens", "tokens_mask", "input_pos")}
        params = {k: v for k, v in kwargs.items() if k not in static}
        graph = torch.cuda.CUDAGraph()
        # The warmup step compiled everything; never compile during capture
        with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"), _frame_step(), \
                torch.compiler.set_stance("eager_on_recompile"):
            out = self._generate_frame(model, **static, **params)
        return graph, static, out

    def release(self):
        self._graphs.clear()
