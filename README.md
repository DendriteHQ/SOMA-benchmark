# SOMA Bench

SOMA Bench runs SWE-bench style tasks through runtime backends (OpenClaw and Copilot) and records runtime artifacts, patch output, token usage, and optional SWE-bench evaluation results.

The OpenClaw backend can load the SOMA OpenClaw plugin as a context-engine plugin, so benchmark runs can exercise the local SOMA miner during normal OpenClaw execution.

## Run OpenClaw

Expected local layout:

```text
docker compose -f src/soma_bench/benchmark/backends/copilot/copilot-cli-container/docker-compose.yml run --rm copilot \
~/SOMA-plugin
```

Prepare the benchmark repo:

```bash
cd ~/SOMA-benchmark
SOMA_COPILOT_COMPOSE_FILE=src/soma_bench/benchmark/backends/copilot/copilot-cli-container/docker-compose.yml
cp .env.example .env
```

Edit `.env` with your LLM settings. For example:


For standalone OpenRouter runs with OpenClaw and SWE-bench evaluation, set:

```bash
LLM_MODEL=openrouter/qwen/qwen3-coder
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
COMPACT_BENCH_PLUGIN_TEMPLATE_PATH=/path/to/plugin/SOMA-plugin
OPENROUTER_API_KEY=...
```

Optional OpenClaw plugin settings:

```bash
SOMA_OPENCLAW_PLUGIN_PATH=/path/to/plugin/SOMA-plugin
SOMA_OPENCLAW_PLUGIN_REINSTALL_ON_RUN_START=true
```

`SOMA_OPENCLAW_PLUGIN_REINSTALL_ON_RUN_START=true` recreates the plugin Python environment once when a benchmark process starts. It does not reinstall between benchmark tasks in the same run.

Run one benchmark instance directly from its problem statement:


Standalone example:

```bash
source .env && uv run python -m soma_bench benchmark-solve \
  --agent-name openclaw \
  --benchmark SWE-bench/SWE-bench_Verified \
  --instance-id INSTANCE_ID \
  --benchmark-type swebench_verified \
  --execute \
  --openclaw-plugin-path /path/to/plugin/SOMA-plugin \
  --openclaw-current-user \
  --openclaw-plugin-reinstall-on-run-start \
  --openclaw-command "--timeout 3000"
  --swerebench-eval
```

`--benchmark-type` selects which task mode the agent runs in. If omitted, it defaults to
`swebench_verified`. Accepted values:

- `swebench_verified` (default) — standard SWE-bench Verified solve mode: the agent gets the
  issue's problem statement (and hints) and must produce a patch.
- `swe_explorer_explore` — SWE-Explorer file exploration mode: the agent explores the repo and
  outputs the regions it read, without producing a patch.
- `swe_explorer_edit` — SWE-Explorer edit mode: the agent receives the ground-truth modified
  files as a hint and must produce a patch.

Run OpenClaw against an existing benchmark manifest:

```bash
uv run python -m soma_bench benchmark-run-infer \
  --dataset outputs/benchmark/soma_manifest.jsonl \
  --runtime-backend openclaw \
  --workspace docker \
  --output-dir outputs/soma-bench-local/openclaw \
  --execute \
  --openclaw-plugin-path /path/to/plugin/SOMA-plugin \
  --openclaw-plugin-reinstall-on-run-start \
  --concurrency 3 \
  --openclaw-command "--timeout 180"
```

Results are written to `output.jsonl` inside the selected output directory. SOMA plugin runtime metadata is included in each result row under `metadata.plugin`, and normalized OpenClaw token usage is included under `metadata.token_usage`, including `model_calls_count` and cumulative provider token totals. Plugin artifacts and logs are written under `openclaw-gateway-state/plugin-artifacts`. For OpenClaw, `--concurrency` or `SOMA_BENCHMARK_CONCURRENCY` runs multiple benchmark rows against one run-scoped gateway after installing the SOMA plugin and writing the OpenClaw config once.

To disable the OpenClaw plugin for a comparison run, add:

```bash
--openclaw-disable-plugin
```

## Run Copilot

The Copilot backend runs the official `@github/copilot` CLI inside a sandboxed Docker Compose
stack (`copilot` service + an internal `proxy` sidecar + an optional `compression-service`
sidecar), pointed at your own LLM endpoint instead of real GitHub Copilot auth
(`COPILOT_OFFLINE=true` + a custom provider base URL). It works fully standalone — no SOMA
validator, OpenClaw, or plugin repo required.

### Prerequisites

The Compose stack expects these images to already exist locally (they are not pulled from a
registry — `pull_policy: never`):

```bash
cd src/soma_bench/benchmark/backends/copilot/copilot-cli-container
docker compose build copilot          # builds local/copilot-cli:latest

cd ../../../../../..                  # back to repo root
docker build -t soma-copilot-compression-service:latest src/compression_service
```

(If the images already exist — check with `docker images | grep -E "copilot-cli|compression-service"`
— you can skip this step.)

### Required `.env` settings

The runner resolves LLM credentials with this precedence (first non-empty wins):

| Purpose   | Env vars (in priority order)                     |
|-----------|---------------------------------------------------|
| API key   | `LLM_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY` |
| Model     | `LLM_MODEL`, `OPENAI_MODEL`, `OPENROUTER_MODEL` (or `--model`) |
| Base URL  | `LLM_BASE_URL`, `OPENAI_BASE_URL`, `OPENROUTER_BASE_URL` (optional) |
| API version | `LLM_API_VERSION` (optional) |

At minimum you need one API key var and one model var set. For a standalone OpenRouter run:

```bash
cp .env.example .env
```

```bash
OPENROUTER_API_KEY=sk-or-...
LLM_MODEL=openrouter/qwen/qwen3-coder
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

`HF_TOKEN` is only needed if the benchmark dataset is a gated/private Hugging Face dataset.

### Standalone run

Run one instance directly, without any pre-built manifest:

```bash
source .env && uv run python -m soma_bench benchmark-solve \
  --agent-name copilot \
  --benchmark SWE-bench/SWE-bench_Verified \
  --instance-id INSTANCE_ID \
  --execute \
  --output-dir outputs/run-copilot-standalone \
  --swerebench-eval
```

Drop `--swerebench-eval` to only collect the patch/trajectory without running the SWE-bench
evaluation harness afterward. Drop `--execute` to only write the benchmark scaffold
(`benchmark-manifest.jsonl`, `image-plan.json`) without actually launching the container.

### Pointing at a compression/miner script

The `compression-service` sidecar is **enabled by default** on every Copilot run (it sits
between the proxy and the upstream LLM). If you don't pass a script, it silently mounts the
no-op fallback at `src/compression_service/app/base_miner.py` — an identity function that
returns messages unchanged:

```python
def compress_messages(messages=None, path=None, metadata=None) -> list:
    ...
    return messages if isinstance(messages, list) else []
```

To exercise your own compressor/miner, point at your script explicitly. It must be importable
as a module exposing `compress_messages(messages, path, metadata) -> list`:

```bash
uv run python -m soma_bench benchmark-solve \
  --agent-name copilot \
  --benchmark SWE-bench/SWE-bench_Verified \
  --instance-id INSTANCE_ID \
  --execute \
  --copilot-compression-script-path /absolute/path/to/your_miner.py \
  --copilot-compression-service-autobuild \
  --output-dir outputs/run-copilot-with-compression \
  --swerebench-eval
```

- `--copilot-compression-script-path` (or `SOMA_COPILOT_COMPRESSION_SCRIPT_PATH`) is bind-mounted
  read-only into the sidecar as `/app/miner/base_miner.py`, overriding the default.
- `--copilot-compression-service-autobuild` rebuilds the `compression-service` image from
  `src/compression_service` before the run — use it whenever `requirements.txt` or the service
  code (not just your mounted miner script) changed. `--copilot-compression-service-build-context`
  overrides the build context directory if your service image lives elsewhere.
- To skip the sidecar entirely (bypass compression, talk to the LLM directly through the proxy),
  set `SOMA_COPILOT_USE_COMPOSE_COMPRESSION_SERVICE=false`.
- `--swerebench-eval` runs the actual SWE-bench evaluation harness (`swebench.harness.run_evaluation`)
  against the patch Copilot produced — applying it to the instance's test environment and checking
  `FAIL_TO_PASS`/`PASS_TO_PASS` outcomes — right after the agent finishes, instead of only capturing
  the raw diff. Results land in `evaluation-summary.json` (`patch_evaluation.resolved_count`, etc.)
  in the output directory. It's a no-op if the agent produced no patch (`patch_capture.has_changes`
  is `false`). Related flags let you point at a different SWE-bench harness fork/Python interpreter

### Useful `SOMA_COPILOT_*` overrides

All of these are optional; the CLI flags shown earlier take precedence over the matching
env var, which takes precedence over the built-in default.

| Env var | Default | Purpose |
|---|---|---|
| `SOMA_COPILOT_COMPOSE_FILE` | `copilot-cli-container/docker-compose.yml` | Alternate Compose file |
| `SOMA_COPILOT_SERVICE` | `copilot` | Compose service name for the CLI container |
| `SOMA_COPILOT_ARGS` | `--allow-all --no-ask-user` | Extra args passed to the `copilot` CLI |
| `SOMA_COPILOT_OUTPUT_FORMAT` | `json` | `json` or `text` CLI output |
| `SOMA_COPILOT_NETWORK_ISOLATION` | `true` | Run the CLI container without direct internet, routed through the proxy sidecar |
| `SOMA_COPILOT_KEEP_STACK` | `false` | Keep the Compose stack up after the run instead of tearing it down |
| `SOMA_COPILOT_SWE_SANDBOX` | `true` | Start a read-only sandbox container exposing the SWE-bench image's `/testbed` |
| `SOMA_COPILOT_SHARED_PROXY` | `true` | Reuse one proxy/compression stack across a batch instead of one per instance |

### Output layout

Per instance, under `--output-dir`:

- `instances/<instance_id>/copilot/{command.txt,prompt.txt,stdout.log,stderr.log}` — exact CLI
  invocation and raw output
- `copilot-trajectory-<instance_id>.jsonl` — structured trajectory
- `output.jsonl` — one row per instance, with full run metadata (`compose_project`,
  `compression_script_path`, `swe_sandbox_image`, `exit_code`, etc.) under `metadata`
- `evaluation-summary.json` (only with `--swerebench-eval`) — `patch_capture.rows_with_changes`
  tells you whether Copilot actually produced a diff before you look at resolution rate; a run
  can finish with `status: completed` and still have made zero code changes.