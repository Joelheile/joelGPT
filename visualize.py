import importlib
import inspect
import json
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = pathlib.Path(__file__).parent
PAGE_PATH = ROOT / "visualizer" / "index.html"
WEIGHT_FILES = {"joel_gpt": "joel_gpt.pkl", "joel_gpt_advanced": "joel_gpt_advanced.pkl", "ai_gpt": "ai_gpt.pt"}
MODEL = sys.argv[1] if len(sys.argv) > 1 else "joel_gpt"
if MODEL not in WEIGHT_FILES:
    sys.exit(f"model must be one of {', '.join(WEIGHT_FILES)}, got {MODEL}")
WEIGHTS_PATH = ROOT / "weights" / WEIGHT_FILES[MODEL]
model = importlib.import_module(MODEL)
PORT = 8000
PROMPT = "Once upon a time"
TIMELINE_LENGTH = 8
SOURCE_FUNCTIONS = [
    "get_batch", "embed", "softmax", "normalize", "layer_norm", "attention", "multi_head_attention", "feed_forward",
    "transformer_block", "predict", "forward_pass", "backward_pass", "adam_optimizer",
    "embed_backward", "layer_norm_backward", "multi_head_attention_backward", "feed_forward_backward",
]
PARAMETER_NAMES = [
    "token_table", "position_table", "gamma1", "beta1",
    "query_weights", "key_weights", "value_weights", "output_weights",
    "gamma2", "beta2", "W1", "b1", "W2", "b2",
    "gamma_final", "beta_final", "vocab_weights",
]


def rounded(array):
    return np.round(np.asarray(array, dtype=float), 4).tolist()


def visible(numbers):
    return [{" ": "␣", "\n": "⏎"}.get(model.decode([n]), model.decode([n])) for n in numbers]


def letter_map():
    centered = model.token_table - model.token_table.mean(axis=0)
    directions = np.linalg.svd(centered, full_matrices=False)[2][:2]
    points = centered @ directions.T
    anchored = points * np.sign(points[model.encode("e")[0]])
    return {"chars": visible(range(model.vocab_size)), "points": rounded(anchored)}


class Live:
    def __init__(self):
        start = model.val_data.tolist().index(model.encode("Once")[0])
        sample = model.val_data[start:start + model.block_size + 1]
        self.x = sample[None, :-1]
        self.y = sample[None, 1:]
        self.checkpoint_time = None
        self.checkpoint = {"step": 0, "max_steps": model.max_steps, "train_losses": [], "validation_losses": []}
        self.previous_parameters = [weights.copy() for weights in model.parameters]
        self.previous_step = 0
        self.history = []
        self.error = None
        self.run_sample()

    def refresh(self):
        if not WEIGHTS_PATH.exists():
            return
        modified = WEIGHTS_PATH.stat().st_mtime
        if modified == self.checkpoint_time:
            return
        self.checkpoint_time = modified
        before = [weights.copy() for weights in model.parameters]
        previous_step = self.checkpoint["step"]
        try:
            self.checkpoint = model.load()
        except (KeyError, TypeError, ValueError) as error:
            self.error = f"{WEIGHTS_PATH.name} does not fit the current {MODEL}.py settings, retrain with python {MODEL}.py ({error})"
            return
        self.error = None
        self.previous_parameters = before
        self.previous_step = previous_step
        self.run_sample()

    def run_sample(self):
        self.loss, self.cache = model.forward_pass(self.x, self.y)
        self.gradients = model.backward_pass(self.x, self.y, self.cache)
        self.generated = model.generate(PROMPT, 200)
        step = self.checkpoint["step"]
        if self.history and self.history[-1]["step"] == step:
            self.history.pop()
        self.history.append({"step": step, "text": self.generated})

    def timeline(self):
        picks = np.linspace(0, len(self.history) - 1, min(TIMELINE_LENGTH, len(self.history))).astype(int)
        return [self.history[i] for i in sorted(set(picks))]

    def lens(self, block_caches, block_out):
        streams = [cache[0] for cache in block_caches] + [block_out]
        rows = []
        for stream in streams:
            probabilities = model.softmax(model.layer_norm(stream[0], model.gamma_final, model.beta_final) @ model.vocab_weights)
            rows.append({
                "top": visible(np.argmax(probabilities, axis=-1)),
                "correct": rounded(probabilities[np.arange(len(probabilities)), self.y[0]]),
            })
        return rows

    def progress(self):
        self.refresh()
        return {
            "error": self.error,
            "step": self.checkpoint["step"],
            "max_steps": self.checkpoint["max_steps"],
            "train_losses": self.checkpoint["train_losses"],
            "validation_losses": self.checkpoint["validation_losses"],
        }

    def view(self, layer, head):
        probabilities, final_norm, block_out, block_caches = self.cache
        block_in, mid, attention_cache, feed_forward_cache = block_caches[layer]
        ln1, heads, head_caches = attention_cache
        query, key, value, weights = head_caches[head]
        head_attention = [cache[3][0] for cache in head_caches]
        ln2, hidden, relu = feed_forward_cache
        layer_out = block_caches[layer + 1][0] if layer + 1 < model.n_layers else block_out
        probabilities = probabilities.reshape(self.x.shape[1], -1)
        top = np.argsort(-probabilities, axis=-1)[:, :3]

        return {
            "step": self.checkpoint["step"],
            "loss": float(self.loss),
            "x": visible(self.x[0]),
            "y": visible(self.y[0]),
            "token_rows": rounded(model.token_table[self.x[0]]),
            "position_rows": rounded(model.position_table[:self.x.shape[1]]),
            "embedded": rounded(block_caches[0][0][0]),
            "ln1": rounded(ln1[0]),
            "query": rounded(query[0]),
            "key": rounded(key[0]),
            "value": rounded(value[0]),
            "head_attention": rounded(head_attention),
            "head_out": rounded((weights @ value)[0]),
            "heads": rounded(heads[0]),
            "attention_out": rounded((mid - block_in)[0]),
            "mid": rounded(mid[0]),
            "ln2": rounded(ln2[0]),
            "hidden": rounded(relu[0]),
            "feed_forward_out": rounded((layer_out - mid)[0]),
            "layer_out": rounded(layer_out[0]),
            "final_norm": rounded(final_norm[0]),
            "correct_probability": rounded(probabilities[np.arange(len(probabilities)), self.y[0]]),
            "top_chars": [visible(row) for row in top],
            "top_probabilities": rounded(np.take_along_axis(probabilities, top, axis=-1)),
            "gradient_norms": [float(np.linalg.norm(g)) for g in self.gradients],
            "change_norms": [float(np.linalg.norm(now - before)) for now, before in zip(model.parameters, self.previous_parameters)],
            "weight_norms": [float(np.linalg.norm(before)) for before in self.previous_parameters],
            "steps_between": self.checkpoint["step"] - self.previous_step,
            "lens": self.lens(block_caches, block_out),
            "letter_map": letter_map(),
            "timeline": self.timeline(),
            "generated": self.generated,
        }

    def setup(self):
        return {
            "model": MODEL,
            "file": f"{MODEL}.py",
            "command": f"python {MODEL}.py",
            "config": {name: getattr(model, name) for name in
                       ["batch_size", "block_size", "n_embedding", "n_heads", "head_size", "n_layers", "vocab_size",
                        "learning_rate", "optimizer_beta1", "optimizer_beta2"]},
            "parameter_names": PARAMETER_NAMES,
            "parameter_shapes": [list(weights.shape) for weights in model.parameters],
            "prompt": PROMPT,
            "source": {name: inspect.getsource(getattr(model, name)) for name in SOURCE_FUNCTIONS},
            "source_lines": {name: inspect.getsourcelines(getattr(model, name))[1] for name in SOURCE_FUNCTIONS},
        }


class GptLive:
    def __init__(self):
        self.checkpoint_time = None
        self.checkpoint = {"step": 0, "max_steps": model.max_steps, "train_losses": [], "validation_losses": []}
        self.generated = PROMPT
        self.error = None

    def refresh(self):
        if not WEIGHTS_PATH.exists():
            return
        modified = WEIGHTS_PATH.stat().st_mtime
        if modified == self.checkpoint_time:
            return
        self.checkpoint_time = modified
        try:
            network, self.checkpoint = model.load()
        except RuntimeError as error:
            self.error = f"{WEIGHTS_PATH.name} does not fit the current {MODEL}.py, retrain with python {MODEL}.py ({error})"
            return
        self.error = None
        self.generated = model.generate(network, PROMPT, 200)

    def progress(self):
        self.refresh()
        return {
            "error": self.error,
            "step": self.checkpoint["step"],
            "max_steps": self.checkpoint["max_steps"],
            "train_losses": self.checkpoint["train_losses"],
            "validation_losses": self.checkpoint["validation_losses"],
        }

    def view(self, layer, head):
        return {"step": self.checkpoint["step"], "generated": self.generated}

    def setup(self):
        return {
            "model": MODEL,
            "file": f"{MODEL}.py",
            "command": f"python {MODEL}.py",
            "config": {"vocab_size": model.vocab_size},
            "prompt": PROMPT,
        }


class Handler(BaseHTTPRequestHandler):
    live = None

    def do_GET(self):
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        if url.path == "/":
            self.send(200, PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/setup":
            self.send_json(self.live.setup())
        elif url.path == "/api/progress":
            self.send_json(self.live.progress())
        elif url.path == "/api/view":
            layer = int(query.get("layer", "0"))
            head = int(query.get("head", "0"))
            if not (0 <= layer < model.n_layers and 0 <= head < model.n_heads):
                self.send_json({"error": f"layer must be 0 to {model.n_layers - 1}, head 0 to {model.n_heads - 1}"}, 400)
                return
            self.send_json(self.live.view(layer, head))
        else:
            self.send_json({"error": f"unknown path {url.path}"}, 404)

    def send_json(self, payload, status=200):
        self.send(status, json.dumps(payload).encode(), "application/json")

    def send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    Handler.live = GptLive() if MODEL == "ai_gpt" else Live()
    print(f"open http://localhost:{PORT}")
    HTTPServer(("localhost", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
