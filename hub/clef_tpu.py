"""clef-flash on a TPU in JAX, with int8 weights.

Mirrors transformers' `modeling_qwen3_5.py` for the backbone and `JointSchemaHead` in Cloudflare's
`joint_schema_model.py` for the head. Prompt building and the answer format come from
`joint_schema_model.py` itself, which ships next to this file.

    model = load_model()
    response = systemone(model, request)
"""

import functools
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import struct
from dataclasses import dataclass, field
from pathlib import Path

# joint_schema_model imports torch only for prompt building. On TPU images torch would otherwise
# autoload its TPU backend, which starts a second TPU runtime next to JAX's.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import serialize_executable
from safetensors.flax import load_file
from transformers import AutoTokenizer

import joint_schema_model as clef

# Not resolve(): in the Hugging Face cache this file is a symlink into blobs/, away from config.json.
MODEL_DIR = Path(__file__).absolute().parent
CONFIG = json.loads((MODEL_DIR / "config.json").read_text())["text_config"]
HEAD_CONFIG = json.loads((MODEL_DIR / "joint_head_config.json").read_text())
INTERVAL = CONFIG["full_attention_interval"]
assert all((kind == "full_attention") == ((layer + 1) % INTERVAL == 0) for layer, kind in enumerate(CONFIG["layer_types"]))
BLOCKS = len(CONFIG["layer_types"]) // INTERVAL
LINEAR_PER_BLOCK = INTERVAL - 1
ACTIVATION_DTYPE = jnp.bfloat16
DELTA_CHUNK = 64
ATTENTION_QUERY_BLOCK = 512
# Each multiple of SEQUENCE_BUCKET tokens needs its own compiled backbone (about 22 s to compile on a v5e).
SEQUENCE_BUCKET = 1024
MAX_TOKENS = 16384
HIGHEST = jax.lax.Precision.HIGHEST


@dataclass(frozen=True)
class ClefTpuModel:
    tokenizer: object
    backbone: dict
    final_norm: jax.Array
    head: dict
    # The embedding table and LM head stay in host memory; the head reads only the option tokens' rows.
    embeddings: np.ndarray
    output_embeddings: np.ndarray
    # Keyed by (prompt bucket, int8_activations).
    compiled_backbones: dict = field(default_factory=dict)


def int8_matmul(x, weights, projection, int8_activations):
    weight, scale = weights[f"{projection}.int8"], weights[f"{projection}.scale"]
    if int8_activations:
        x_scale = jnp.maximum(jnp.max(jnp.abs(x), axis=-1, keepdims=True).astype(jnp.float32), 1e-6) / 127
        x_int8 = jnp.clip(jnp.round(x / x_scale), -127, 127).astype(jnp.int8)
        product = jax.lax.dot(x_int8, weight, preferred_element_type=jnp.int32).astype(jnp.float32) * x_scale
    else:
        product = jnp.dot(x, weight.astype(x.dtype), preferred_element_type=jnp.float32)
    return (product * scale).astype(x.dtype)


def dense(x, weight):
    return jnp.dot(x, weight.astype(x.dtype), preferred_element_type=jnp.float32).astype(x.dtype)


def rms_norm(x, weight):
    """Qwen3.5's RMSNorm scales by (1 + weight)."""
    x32 = x.astype(jnp.float32)
    normed = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + CONFIG["rms_norm_eps"])
    return (normed * (1.0 + weight.astype(jnp.float32))).astype(x.dtype)


def gated_rms_norm(x, gate, weight):
    """The DeltaNet output norm: plain weight, then a SiLU gate."""
    x32 = x.astype(jnp.float32)
    normed = (x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + CONFIG["rms_norm_eps"])).astype(x.dtype)
    gated = (weight.astype(x.dtype) * normed).astype(jnp.float32) * jax.nn.silu(gate.astype(jnp.float32))
    return gated.astype(x.dtype)


def causal_conv_silu(x, weight):
    kernel = weight[:, 0, :].astype(jnp.float32)
    taps = kernel.shape[-1]
    padded = jnp.pad(x.astype(jnp.float32), ((taps - 1, 0), (0, 0)))
    mixed = sum(padded[tap : tap + x.shape[0]] * kernel[:, tap] for tap in range(taps))
    return jax.nn.silu(mixed).astype(x.dtype)


def l2_normalize(x):
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + 1e-6)


def chunked_gated_delta_rule(query, key, value, log_decay, beta):
    """transformers' torch_chunk_gated_delta_rule with L2-normalized q and k; float32 inputs shaped (tokens, heads, ...)."""
    tokens, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    padding = (-tokens) % DELTA_CHUNK

    def to_chunks(array):
        array = jnp.pad(array, [(0, padding)] + [(0, 0)] * (array.ndim - 1))
        return jnp.moveaxis(array.reshape(-1, DELTA_CHUNK, *array.shape[1:]), 2, 0)

    query = to_chunks(l2_normalize(query) * key_dim**-0.5)
    key, value = to_chunks(l2_normalize(key)), to_chunks(value)
    log_decay, beta = to_chunks(log_decay), to_chunks(beta)

    key_beta, value_beta = key * beta[..., None], value * beta[..., None]
    cumulative = jnp.cumsum(log_decay, axis=-1)
    causal = jnp.tril(jnp.ones((DELTA_CHUNK, DELTA_CHUNK), dtype=bool))
    pairwise_decay = jnp.exp(jnp.where(causal, cumulative[..., :, None] - cumulative[..., None, :], -jnp.inf))
    ut_system = jnp.einsum("hncd,hnmd->hncm", key_beta, key, precision=HIGHEST) * pairwise_decay
    intra_chunk = jnp.einsum("hncd,hnmd->hncm", query, key, precision=HIGHEST) * pairwise_decay
    decayed_key_beta = key_beta * jnp.exp(cumulative)[..., None]
    solved = jax.lax.linalg.triangular_solve(
        ut_system, jnp.concatenate([value_beta, decayed_key_beta], axis=-1), left_side=True, lower=True, unit_diagonal=True
    )
    new_values, key_cumulative_decay = solved[..., :value_dim], solved[..., value_dim:]
    query = query * jnp.exp(cumulative)[..., None]
    key = key * jnp.exp(cumulative[..., -1:] - cumulative)[..., None]
    decay_over_chunk = jnp.exp(cumulative[..., -1])

    def step(state, chunk):
        chunk_query, chunk_key, chunk_new_values, chunk_key_decay, chunk_intra, chunk_decay = chunk
        corrected = chunk_new_values - jnp.einsum("hcd,hdv->hcv", chunk_key_decay, state, precision=HIGHEST)
        output = jnp.einsum("hcd,hdv->hcv", chunk_query, state, precision=HIGHEST)
        output = output + jnp.einsum("hcm,hmv->hcv", chunk_intra, corrected, precision=HIGHEST)
        state = state * chunk_decay[:, None, None] + jnp.einsum("hcd,hcv->hdv", chunk_key, corrected, precision=HIGHEST)
        return state, output

    chunks = tuple(jnp.moveaxis(array, 1, 0) for array in (query, key, new_values, key_cumulative_decay, intra_chunk, decay_over_chunk))
    _, outputs = jax.lax.scan(step, jnp.zeros((heads, key_dim, value_dim), jnp.float32), chunks)
    return jnp.transpose(outputs, (0, 2, 1, 3)).reshape(-1, heads, value_dim)[:tokens]


def linear_attention_layer(x, weights, int8_activations):
    tokens = x.shape[0]
    key_heads, value_heads = CONFIG["linear_num_key_heads"], CONFIG["linear_num_value_heads"]
    key_dim, value_dim = CONFIG["linear_key_head_dim"], CONFIG["linear_value_head_dim"]
    mixed = causal_conv_silu(int8_matmul(x, weights, "in_proj_qkv", int8_activations), weights["conv1d"])
    query, key, value = jnp.split(mixed, [key_heads * key_dim, 2 * key_heads * key_dim], axis=-1)
    query = jnp.repeat(query.reshape(tokens, key_heads, key_dim), value_heads // key_heads, axis=1)
    key = jnp.repeat(key.reshape(tokens, key_heads, key_dim), value_heads // key_heads, axis=1)
    value = value.reshape(tokens, value_heads, value_dim)
    beta = jax.nn.sigmoid(dense(x, weights["in_proj_b"]))
    a = dense(x, weights["in_proj_a"]).astype(jnp.float32)
    log_decay = -jnp.exp(weights["A_log"].astype(jnp.float32)) * jax.nn.softplus(a + weights["dt_bias"].astype(jnp.float32))
    core = chunked_gated_delta_rule(*(array.astype(jnp.float32) for array in (query, key, value)), log_decay, beta.astype(jnp.float32))
    gate = int8_matmul(x, weights, "in_proj_z", int8_activations).reshape(tokens, value_heads, value_dim)
    normed = gated_rms_norm(core.astype(x.dtype), gate, weights["norm"]).reshape(tokens, -1)
    return int8_matmul(normed, weights, "out_proj", int8_activations)


def rotary_tables(length, dtype):
    rope = CONFIG["rope_parameters"]
    rotary_dim = int(CONFIG["head_dim"] * rope["partial_rotary_factor"])
    inverse_frequency = 1.0 / (rope["rope_theta"] ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))
    angles = jnp.arange(length, dtype=jnp.float32)[:, None] * inverse_frequency
    angles = jnp.concatenate([angles, angles], axis=-1)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def apply_rotary(x, cos, sin):
    rotary_dim = cos.shape[-1]
    rotated, passed = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = jnp.split(rotated, 2, axis=-1)
    rotated = rotated * cos[:, None] + jnp.concatenate([-second, first], axis=-1) * sin[:, None]
    return jnp.concatenate([rotated, passed], axis=-1)


def causal_attention(query, key, value):
    """Query blocks against all keys, so memory grows linearly with the prompt instead of quadratically."""
    tokens, heads, head_dim = query.shape
    key = jnp.repeat(key, heads // key.shape[1], axis=1)
    value = jnp.repeat(value, heads // value.shape[1], axis=1)
    block = min(ATTENTION_QUERY_BLOCK, tokens)
    key_positions = jnp.arange(tokens)

    def attend(block_start):
        query_block = jax.lax.dynamic_slice_in_dim(query, block_start, block)
        scores = jnp.einsum("qhd,khd->hqk", query_block, key, preferred_element_type=jnp.float32) * head_dim**-0.5
        is_visible = (block_start + jnp.arange(block))[:, None] >= key_positions[None, :]
        weights = jax.nn.softmax(jnp.where(is_visible[None], scores, -jnp.inf), axis=-1).astype(value.dtype)
        return jnp.einsum("hqk,khd->qhd", weights, value, preferred_element_type=jnp.float32).astype(value.dtype)

    return jax.lax.map(attend, jnp.arange(0, tokens, block)).reshape(tokens, heads, head_dim)


def full_attention_layer(x, weights, cos, sin, int8_activations):
    tokens = x.shape[0]
    heads, kv_heads, head_dim = CONFIG["num_attention_heads"], CONFIG["num_key_value_heads"], CONFIG["head_dim"]
    query, gate = jnp.split(int8_matmul(x, weights, "q_proj", int8_activations).reshape(tokens, heads, 2 * head_dim), 2, axis=-1)
    key = int8_matmul(x, weights, "k_proj", int8_activations).reshape(tokens, kv_heads, head_dim)
    value = int8_matmul(x, weights, "v_proj", int8_activations).reshape(tokens, kv_heads, head_dim)
    query = apply_rotary(rms_norm(query, weights["q_norm"]), cos, sin)
    key = apply_rotary(rms_norm(key, weights["k_norm"]), cos, sin)
    attended = causal_attention(query, key, value).reshape(tokens, heads * head_dim)
    return int8_matmul(attended * jax.nn.sigmoid(gate.reshape(tokens, -1)), weights, "o_proj", int8_activations)


def mlp(x, weights, int8_activations):
    gate = int8_matmul(x, weights, "gate_proj", int8_activations)
    up = int8_matmul(x, weights, "up_proj", int8_activations)
    return int8_matmul(jax.nn.silu(gate) * up, weights, "down_proj", int8_activations)


def layer_slice(group, position):
    return {key: array[position] for key, array in group.items()}


def decoder_block(x, weights, cos, sin, int8_activations):
    norms = weights["norms"]
    for position in range(INTERVAL):
        normed = rms_norm(x, norms["input_layernorm"][position])
        if position < LINEAR_PER_BLOCK:
            x = x + linear_attention_layer(normed, layer_slice(weights["linear"], position), int8_activations)
        else:
            x = x + full_attention_layer(normed, weights["full"], cos, sin, int8_activations)
        x = x + mlp(rms_norm(x, norms["post_attention_layernorm"][position]), layer_slice(weights["mlp"], position), int8_activations)
    return x


@functools.partial(jax.jit, static_argnames="int8_activations")
def backbone(embedding_bits, weights, final_norm, int8_activations):
    hidden = jax.lax.bitcast_convert_type(embedding_bits, jnp.bfloat16).astype(ACTIVATION_DTYPE)
    cos, sin = rotary_tables(hidden.shape[0], ACTIVATION_DTYPE)

    def scan_block(x, block_weights):
        return decoder_block(x, block_weights, cos, sin, int8_activations), None

    hidden, _ = jax.lax.scan(scan_block, hidden, weights)
    return rms_norm(hidden, final_norm)


def layer_norm(x, weights, name):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + 1e-5) * weights[f"{name}.weight"] + weights[f"{name}.bias"]


def linear(x, weights, name):
    output = x @ weights[f"{name}.weight"].T
    return output + weights[f"{name}.bias"] if f"{name}.bias" in weights else output


def gelu(x):
    return jax.nn.gelu(x, approximate=False)


def l2_unit(x):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def multihead_attention(queries, memory, weights, name, memory_mask=None):
    """torch.nn.MultiheadAttention with a packed in-projection; keys and values both come from memory."""
    width = queries.shape[-1]
    heads = HEAD_CONFIG["heads"]
    in_weight, in_bias = weights[f"{name}.in_proj_weight"], weights[f"{name}.in_proj_bias"]

    def project(x, part):
        rows = slice(part * width, (part + 1) * width)
        return (x @ in_weight[rows].T + in_bias[rows]).reshape(x.shape[0], heads, width // heads)

    scores = jnp.einsum("qhd,khd->hqk", project(queries, 0), project(memory, 1)) / math.sqrt(width // heads)
    if memory_mask is not None:
        scores = jnp.where(memory_mask[None, None, :], scores, -jnp.inf)
    attended = jnp.einsum("hqk,khd->qhd", jax.nn.softmax(scores, axis=-1), project(memory, 2))
    return linear(attended.reshape(-1, width), weights, f"{name}.out_proj")


def evidence_routing_layer(queries, memory, memory_mask, weights, name):
    normed_memory = layer_norm(memory, weights, f"{name}.memory_norm")
    queries = queries + multihead_attention(layer_norm(queries, weights, f"{name}.query_norm"), normed_memory, weights, f"{name}.attention", memory_mask)
    hidden = gelu(linear(layer_norm(queries, weights, f"{name}.feedforward_norm"), weights, f"{name}.feedforward.0"))
    return queries + linear(hidden, weights, f"{name}.feedforward.3")


def field_decoder_layer(fields, memory, memory_mask, weights, name):
    """torch.nn.TransformerDecoderLayer with norm_first=True and GELU."""
    normed = layer_norm(fields, weights, f"{name}.norm1")
    fields = fields + multihead_attention(normed, normed, weights, f"{name}.self_attn")
    fields = fields + multihead_attention(layer_norm(fields, weights, f"{name}.norm2"), memory, weights, f"{name}.multihead_attn", memory_mask)
    hidden = gelu(linear(layer_norm(fields, weights, f"{name}.norm3"), weights, f"{name}.linear1"))
    return fields + linear(hidden, weights, f"{name}.linear2")


def segment_softmax(scores, segments, count):
    maxima = jax.ops.segment_max(scores, segments, num_segments=count)
    exponentials = jnp.exp(scores - maxima[segments])
    return exponentials / jax.ops.segment_sum(exponentials, segments, num_segments=count)[segments]


def clamped_scale(log_scale):
    return jnp.exp(jnp.minimum(log_scale, math.log(100.0)))


@jax.jit
def joint_head(hidden, length, question_spans, option_spans, option_questions, question_types, lexical, head_weights):
    """JointSchemaHead.forward for one record, with options flattened across questions."""
    weights = jax.tree.map(lambda array: array.astype(jnp.float32), head_weights)
    normalized = layer_norm(hidden.astype(jnp.float32), weights, "hidden_norm")
    positions = jnp.arange(hidden.shape[0])
    is_prompt_token = positions < length
    memory = linear(normalized, weights, "memory_projection")
    global_vector = normalized[length - 1]

    def span_means(spans):
        inside = (positions[None, :] >= spans[:, :1]) & (positions[None, :] < spans[:, 1:])
        return jnp.dot(inside / inside.sum(axis=1, keepdims=True), normalized, precision=HIGHEST)

    question_vectors = span_means(question_spans)
    options = (
        linear(span_means(option_spans), weights, "option_context_projection")
        + linear(lexical, weights, "option_lexical_projection")
        + linear(question_vectors, weights, "option_question_projection")[option_questions]
    )
    for layer in range(HEAD_CONFIG["routing_layers"]):
        options = evidence_routing_layer(options, memory, is_prompt_token, weights, f"evidence_layers.{layer}")

    question_count = question_spans.shape[0]
    base_fields = linear(question_vectors, weights, "question_projection")
    routing_scores = jnp.sum(options * base_fields[option_questions], axis=-1) / math.sqrt(options.shape[-1])
    routing_weights = segment_softmax(routing_scores, option_questions, question_count)
    summaries = jax.ops.segment_sum(routing_weights[:, None] * options, option_questions, num_segments=question_count)
    fields = (
        base_fields
        + layer_norm(summaries, weights, "option_summary_norm")
        + linear(global_vector, weights, "global_projection")
        + weights["type_embedding.weight"][question_types]
    )
    for layer in range(HEAD_CONFIG["layers"]):
        fields = field_decoder_layer(fields, memory, is_prompt_token, weights, f"layers.{layer}")
    option_fields = layer_norm(fields, weights, "field_norm")[option_questions]

    anchors = l2_unit(question_vectors + global_vector)[option_questions]
    prior = clamped_scale(weights["prior_logit_scale"]) * jnp.sum(l2_unit(lexical) * anchors, axis=-1)
    normed_options = layer_norm(options, weights, "option_norm")
    norms = jnp.linalg.norm(option_fields, axis=-1) * jnp.linalg.norm(normed_options, axis=-1)
    cosine = jnp.sum(option_fields * normed_options, axis=-1) / jnp.maximum(norms, 1e-8)
    features = jnp.concatenate(
        [option_fields, normed_options, option_fields * normed_options, jnp.abs(option_fields - normed_options)], axis=-1
    )
    residual = linear(gelu(linear(features, weights, "residual_scorer.0")), weights, "residual_scorer.3")[:, 0]
    joint = clamped_scale(weights["joint_logit_scale"]) * cosine + residual
    return prior + jax.nn.sigmoid(weights["residual_gate"]) * joint


def host_bf16_matrix(path, name):
    with open(path, "rb") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        entry = json.loads(file.read(header_size))[name]
    assert entry["dtype"] == "BF16", entry["dtype"]
    start, _ = entry["data_offsets"]
    return np.memmap(path, dtype=np.uint16, mode="r", offset=8 + header_size + start, shape=tuple(entry["shape"]))


def bf16_bits_to_float32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def head_inputs(encoded, token_ids, output_embeddings):
    questions = encoded.questions
    option_spans = np.array([span for question in questions for span in question.option_spans], np.int32)
    lexical = np.stack([bf16_bits_to_float32(output_embeddings[token_ids[start:end]]).mean(axis=0) for start, end in option_spans])
    return (
        np.array([question.question_span for question in questions], np.int32),
        option_spans,
        np.array([index for index, question in enumerate(questions) for _ in question.option_spans], np.int32),
        np.array([question.question_type for question in questions], np.int32),
        lexical,
    )


def regroup(tensors, leading_shape):
    return {key: array.reshape(*leading_shape, *array.shape[1:]) for key, array in tensors.items()}


def runtime_description():
    """The TPU type and jax/libtpu build that a compiled executable only runs on."""
    devices = jax.devices()
    return "\n".join(
        [
            f"{devices[0].device_kind} x{len(devices)}",
            f"jax {jax.__version__}, jaxlib {importlib.metadata.version('jaxlib')}",
            devices[0].client.platform_version,
        ]
    )


def runtime_id():
    return hashlib.sha256(runtime_description().encode()).hexdigest()[:16]


def compiled_backbones_dir(model_dir=MODEL_DIR):
    return Path(model_dir) / "compiled" / runtime_id()


def compiled_backbone_path(directory, bucket, int8_activations):
    return Path(directory) / f"backbone-{bucket}-{'w8a8' if int8_activations else 'bf16'}.bin"


def load_compiled_backbones(directory):
    """Loads executables saved by `compile_backbones`. They are pickles: load only files you trust."""
    compiled = {}
    for path in Path(directory).glob("backbone-*.bin"):
        _, bucket, mode = path.stem.split("-")
        payload, in_tree, out_tree = pickle.loads(path.read_bytes())
        compiled[(int(bucket), mode == "w8a8")] = serialize_executable.deserialize_and_load(payload, in_tree, out_tree)
    return compiled


def load_model(model_dir=MODEL_DIR):
    """Loads the weights next to this file onto the default JAX device (about 9 GB), plus any backbones
    compiled ahead of time for this runtime under compiled/<runtime_id()>/."""
    model_dir = Path(model_dir)

    def to_device(file_name):
        return load_file(str(model_dir / file_name))

    norms = to_device("layer_norms.safetensors")
    return ClefTpuModel(
        tokenizer=AutoTokenizer.from_pretrained(model_dir),
        backbone={
            "linear": regroup(to_device("linear_attention.safetensors"), (BLOCKS, LINEAR_PER_BLOCK)),
            "full": to_device("full_attention.safetensors"),
            "mlp": regroup(to_device("mlp.safetensors"), (BLOCKS, INTERVAL)),
            "norms": regroup({key: norms[key] for key in ("input_layernorm", "post_attention_layernorm")}, (BLOCKS, INTERVAL)),
        },
        final_norm=norms["final_norm"],
        head=to_device("joint_head.safetensors"),
        embeddings=host_bf16_matrix(model_dir / "embeddings.safetensors", "embed_tokens"),
        output_embeddings=host_bf16_matrix(model_dir / "lm_head.safetensors", "lm_head"),
        compiled_backbones=load_compiled_backbones(compiled_backbones_dir(model_dir)),
    )


def compile_backbone(model, bucket, int8_activations):
    embedding_bits = jax.ShapeDtypeStruct((bucket, CONFIG["hidden_size"]), jnp.uint16)
    compiled = backbone.lower(embedding_bits, model.backbone, model.final_norm, int8_activations=int8_activations).compile()
    model.compiled_backbones[(bucket, int8_activations)] = compiled
    return compiled


def compile_backbones(model, up_to_tokens, int8_activations=False, save_dir=None):
    """Compiles the prompt buckets up to `up_to_tokens` that aren't compiled yet and returns them.

    With `save_dir`, also saves each one there; ship them as compiled/<runtime_id()>/ for `load_model`.
    """
    new_buckets = []
    for bucket in range(SEQUENCE_BUCKET, up_to_tokens + 1, SEQUENCE_BUCKET):
        if (bucket, int8_activations) in model.compiled_backbones:
            continue
        compiled = compile_backbone(model, bucket, int8_activations)
        if save_dir is not None:
            save_compiled_backbone(compiled, save_dir, bucket, int8_activations)
        new_buckets.append(bucket)
    return new_buckets


def save_compiled_backbone(compiled, directory, bucket, int8_activations):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "runtime.txt").write_text(runtime_description() + "\n")
    compiled_backbone_path(directory, bucket, int8_activations).write_bytes(pickle.dumps(serialize_executable.serialize(compiled)))


def run_backbone(model, embedding_bits, int8_activations):
    key = (embedding_bits.shape[0], int8_activations)
    compiled = model.compiled_backbones[key] if key in model.compiled_backbones else compile_backbone(model, *key)
    return compiled(embedding_bits, model.backbone, model.final_norm)


def decision_logits(model, request, int8_activations=False):
    encoded = clef.encode_record(model.tokenizer, request, max_length=MAX_TOKENS)
    token_ids = np.array(encoded.input_ids)
    length = len(token_ids)
    padded_ids = np.full(-(-length // SEQUENCE_BUCKET) * SEQUENCE_BUCKET, model.tokenizer.pad_token_id)
    padded_ids[:length] = token_ids
    hidden = run_backbone(model, jnp.asarray(model.embeddings[padded_ids]), int8_activations)
    logits = joint_head(hidden, length, *head_inputs(encoded, token_ids, model.output_embeddings), model.head)
    return encoded, np.asarray(logits, dtype=np.float64)


def systemone(model, request, int8_activations=False):
    """The /v1/systemone response body, as Cloudflare's `joint_schema_model.systemone` builds it."""
    encoded, logits = decision_logits(model, request, int8_activations)
    answers = {}
    offset = 0
    for question in encoded.questions:
        question_logits = logits[offset : offset + len(question.option_ids)]
        offset += len(question.option_ids)
        probabilities = np.exp(question_logits - question_logits.max())
        probabilities /= probabilities.sum()
        answers[question.question_id] = clef.systemone_answer(
            request["questions"][question.question_id], dict(zip(question.option_ids, probabilities.tolist()))
        )
    return {"model": request["model"], "answers": answers, "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0}}
