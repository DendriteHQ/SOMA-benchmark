# SOMA Bench

SOMA Bench runs SWE-bench style tasks through OpenClaw and records runtime artifacts, patch output, token usage, and optional SWE-bench evaluation results.

The current runtime backend is OpenClaw. It can load the SOMA OpenClaw plugin as a context-engine plugin, so benchmark runs can exercise the local SOMA miner during normal OpenClaw execution.

## Run OpenClaw

Expected local layout:

```text
~/SOMA-benchmark
~/SOMA-plugin
```

Prepare the benchmark repo:

```bash
cd ~/SOMA-benchmark
uv sync
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
  --execute \
  --openclaw-plugin-path /path/to/plugin/SOMA-plugin \
  --openclaw-current-user \
  --openclaw-plugin-reinstall-on-run-start \
  --openclaw-command "--timeout 3000"
  --swerebench-eval
```

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
