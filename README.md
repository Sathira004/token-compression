# Token Compression with LangGraph

A BSc research project investigating how context compression and on-demand instruction loading affect token usage, response latency, and answer quality in a retrieval-augmented generation (RAG) system.

The project implements a FastAPI backend with a shared LangGraph workflow and compares four experimental approaches: uncompressed RAG, LLMLingua compression, TOON encoding, and TOON encoding with model-selected skills. It combines PDF retrieval from Pinecone with external data accessed through Model Context Protocol (MCP) tools, records experimental runs in MongoDB, and evaluates responses using Ragas.

## Research objectives

- Measure the effect of each approach on evidence-context tokens and total workflow tokens.
- Compare latency across routing, retrieval, tool execution, compression, and generation.
- Assess whether reducing input context preserves answer quality.
- Examine the trade-off between loading all skill instructions and selecting relevant instructions on demand.
- Evaluate document-only, tool-only, hybrid, and direct-answer tasks through a common workflow.

This repository provides the experimental implementation and benchmark runner. Token savings and quality improvements are research outcomes to measure, rather than guarantees of the system.

## Experimental approaches

| Approach | API `process_type` | Evidence representation | Skill instructions |
| --- | --- | --- | --- |
| Uncompressed baseline | `normal_rag` | Original document prose and JSON tool results | All three skill bodies |
| LLMLingua | `llmlingua_only` | Compressed document prose and original JSON tool results | All three skill bodies |
| TOON | `toon_only` | TOON-encoded document and tool context | All three skill bodies |
| TOON + on-demand skills | `toon_plus_skill` | Same TOON encoding | Only model-selected skill bodies |

The API also accepts `naive_rag` as a baseline alternative and `toon_skill` for TOON with direct generation. Despite its name, `toon_skill` does **not** perform on-demand skill selection.

LLMLingua compresses retrieved document prose using the original question as guidance. MCP JSON, system instructions, the user question, and generated answers bypass that compression. MCP-only requests do not invoke LLMLingua. TOON is a serialization format and may increase token usage for some payloads, especially prose. Skills supply instructions; they are not an additional evidence-compression algorithm.

## Workflow

1. **Route the question:** a structured model call decides whether document retrieval and specific MCP tools are needed.
2. **Collect evidence:** retrieve relevant PDF chunks from Pinecone when requested and execute only selected MCP tools.
3. **Transform context:** retain the original evidence, compress document prose with LLMLingua, or encode evidence with TOON.
4. **Generate an answer:** use all skill instructions for the first three approaches, or select and load relevant instructions for `toon_plus_skill`.
5. **Record and evaluate:** schedule background persistence of measurements in MongoDB, then run Ragas evaluation.

The router is shared across all approaches. It receives the question, not benchmark labels or expected-routing annotations. Requests that need no MCP tools skip the MCP connection. Tool failures or unavailable tools are reported in the evidence.

### Data sources and tools

| Source | Purpose |
| --- | --- |
| Local PDFs → Pinecone | Retrieve supporting document passages; retrieval uses four chunks by default |
| JSONPlaceholder users | Structured user, address, and company data through `mcp_get_users` |
| JSONPlaceholder posts | Post data, optionally filtered by user, through `mcp_get_posts` |
| Open-Meteo | Current weather for supplied coordinates through `mcp_get_weather` |

The local FastMCP server runs over stdio and is launched automatically by the MCP client when tools are needed.

## Technology stack

- **API:** FastAPI, Uvicorn, and Pydantic Settings
- **Orchestration:** LangGraph, LangChain, and Deep Agents dependencies
- **Language models:** Google Gemini through Vertex AI
- **Retrieval:** Pinecone embeddings and vector storage, PyPDF, and recursive text splitting
- **Context processing:** LLMLingua-2 and TOON through `toons`
- **Tools:** FastMCP and LangChain MCP adapters
- **Experiment storage:** MongoDB through PyMongo's `AsyncMongoClient`
- **Evaluation:** Ragas
- **Environment management:** Python and `uv`

## Getting started

### 1. Prerequisites

- Python **3.11 or newer, below 4.0**; the repository pins Python 3.11 in `.python-version`.
- `uv` installed and available on your `PATH`, including for MCP subprocesses.
- A Google Cloud project with Vertex AI access and credentials permitted to use the configured model.
- A Pinecone API key and an existing index compatible with the selected embedding model.
- A reachable MongoDB instance.
- Network access to the configured model services, Pinecone, and external tool APIs. The first LLMLingua request may also need to download model weights.

Run the following commands from the repository root.

### 2. Install dependencies

```bash
uv sync --locked
```

The committed `uv.lock` records the dependency resolution used by the project.

### 3. Configure the environment

Create a `.env` file in the repository root:

```dotenv
# Google Gemini through Vertex AI
GOOGLE_CLOUD_PROJECT=your-google-cloud-project
GOOGLE_CLOUD_LOCATION=global
GOOGLE_LLM_MODEL=gemini-3.1-flash-lite
DEFAULT_MODEL_TEMPERATURE=0.0

# Optional: use a key supported by your Vertex AI authentication setup.
# Otherwise, configure Google Application Default Credentials.
# GOOGLE_API_KEY=your-vertex-ai-api-key

# Pinecone
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-existing-index-name
PINECONE_EMBEDDING_MODEL=llama-text-embed-v2

# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=token_compression
MONGODB_RUNS_COLLECTION=runs

# LLMLingua
LLMLINGUA_MODEL_NAME=microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
LLMLINGUA_DEVICE=cpu
LLMLINGUA_RATE=0.5
LLMLINGUA_USE_LLMLINGUA2=true
```

`GOOGLE_CLOUD_PROJECT` is required when application settings are imported. The configured Gemini model is shared by routing, skill selection, answer generation, and Ragas evaluation. The application explicitly uses Vertex AI; legacy OpenRouter settings are ignored.

For local development with the Google Cloud CLI, Application Default Credentials can be configured with:

```bash
gcloud auth application-default login
```

Replace the example Pinecone values with your own settings and ensure the index dimensions match the embedding model. Keep credentials in `.env`, which is excluded by `.gitignore`.

### 4. Start MongoDB

Use an existing MongoDB deployment, or start a local instance with Docker:

```bash
docker run --name token-compression-mongo --publish 27017:27017 --detach mongo:8
```

For a previously created container:

```bash
docker start token-compression-mongo
```

The API checks MongoDB connectivity at startup and creates a unique `run_id` index plus an index on experiment arm and creation time.

### 5. Add your own documents and ingest them into Pinecone

Research PDFs are not included in this repository. Supply your own PDF documents that you have permission to use, place them in the repository root, and ingest them into your own configured Pinecone index:

```bash
uv run python -m app.scripts.ingest
```

The script reads **all root-level PDFs**, splits their text into chunks of 1,500 characters with 150 characters of overlap, embeds them through Pinecone, and uploads them to the configured index. It records the source filename in chunk metadata. Repeated ingestion can add duplicate content; prepare the index consistently before comparing experiments.

PDF files are excluded from Git by `.gitignore`, so your local documents will not be committed by default. Document retrieval requires an index populated with your own documents; installing the application alone does not populate Pinecone.

The supplied retrieval benchmark questions were written for a particular ONTO research paper. Before using your own corpus, update the document-related questions in `app/benchmarks/prompts.py` to match your documents, including hybrid questions and any expected answers used for evaluation. The original benchmark is not a corpus-independent evaluation set.

### 6. Start the API

```bash
uv run uvicorn app.main:app --reload
```

- API: <http://127.0.0.1:8000>
- Interactive API documentation: <http://127.0.0.1:8000/docs>
- OpenAPI schema: <http://127.0.0.1:8000/openapi.json>

The application entry point is `app.main:app`. The root-level `main.py` is a placeholder and does not start the API.

## API usage

Send a question to `POST /api/v1/agent/chat`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/agent/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Summarize the main findings of my indexed documents and identify the supporting evidence.",
    "thread_id": "readme-example-001",
    "process_type": "toon_plus_skill"
  }'
```

Use a fresh identifier for each independent experiment. The compiled graph currently has no persistence checkpointer, so supplying a `thread_id` does not by itself provide conversation memory across requests.

| Request field | Purpose |
| --- | --- |
| `message` | Required user question |
| `thread_id` | Required identifier passed to the graph configuration |
| `process_type` | Required experimental approach from the table above |
| `run_id` | Optional storage identifier; generated when omitted |
| `prompt_id`, `task_type`, `arm`, `repeat` | Optional experiment metadata; `repeat` defaults to 1 |
| `model` | Optional stored model label; does not override the configured runtime model |
| `reference_answer` | Optional expected answer used for context recall evaluation |
| `ragas` | Optional numeric metadata stored under the evaluation input |

The response includes `status`, `run_id`, `active_process`, `response`, `token_usage`, `token_breakdown`, `selected_tools`, and `loaded_skills`. Answer content follows the chat model's content representation and may contain content blocks.

Persistence and evaluation run as background tasks after the response is prepared. An HTTP success response therefore does not confirm that the run was successfully stored or evaluated. Reusing a `run_id` causes the unique MongoDB index to reject the duplicate insertion.

## Running experiments

The benchmark contains **100 prompts across 10 task categories**, covering direct answers, document retrieval, individual tools, tool combinations, and document-plus-tool questions.

Start the API, then execute:

```bash
uv run python -m app.scripts.run_benchmark
```

The runner sends each prompt through `normal_rag`, `llmlingua_only`, `toon_only`, and `toon_plus_skill` sequentially: **400 requests per full batch**. It creates distinct run IDs and continues after request failures, appending their details to `benchmark_failures.csv`.

Run a smaller comparison or retry a case:

```bash
# One prompt across all four approaches
uv run python -m app.scripts.run_benchmark --prompt-id P001

# One prompt and one approach
uv run python -m app.scripts.run_benchmark --case P001:toon_only

# Use a different API address, timeout, or failure log
uv run python -m app.scripts.run_benchmark \
  --base-url http://127.0.0.1:8000 \
  --timeout 180 \
  --failure-log benchmark_failures.csv
```

`--prompt-id` and `--case` may be supplied multiple times. A benchmark batch makes live model and service calls, including background evaluation, and can incur usage charges. Check stored evaluation statuses before analysing a completed batch; the runner reports HTTP request outcomes.

## Measurements and evaluation

### Token usage

Workflow totals include model calls for **routing, skill selection when applicable, and answering**. The API exposes per-call and per-phase measurements in `token_breakdown`; MongoDB stores them under `tokens.calls` and `tokens.by_phase`.

- Each call records provider-reported input, output, and total usage, its phase, and a `usage_available` flag.
- Evidence-only counts are stored in `tokens.context_raw` and `tokens.context_after_compression`, with details in `tokens.context_measurement`.
- Evidence is counted using the configured provider's count-tokens API. Identical contexts are counted once.
- Counting requests are separate from generation calls; counting latency is recorded separately.
- Unavailable context counts remain null with an explanatory status. Missing call usage is flagged rather than estimated.
- Background Ragas evaluation tokens are excluded from workflow totals.

New runs use `tokens.measurement_version: 2`. Older measurements omitted routing usage and used different prompts or workflows; rerun all approaches together before comparing them.

### Latency

Stored `latency_ms` fields include routing, token counting, compression, retrieval, tools, generation, and end-to-end workflow time. The end-to-end measurement excludes background storage and evaluation.

### Answer quality

A saved run begins with `ragas.status: "pending"` and is then updated with evaluation results or a failure status. Metrics include faithfulness, relevance/answer relevancy, context precision, and context recall.

Evaluation uses plain answer text and the **original, uncompressed document and tool evidence** for every approach. Context-dependent metrics require available evidence; direct-answer tasks still receive answer relevancy evaluation. Context recall additionally requires `reference_answer`; otherwise its score is null and the stored status explains why it was skipped. The benchmark runner does not supply reference answers by default.

### Interpretation and reproducibility

- Compare both evidence reduction and total workflow usage: on-demand skill selection introduces an extra model call.
- Non-loader approaches include all three skill bodies in the system prompt; only `toon_plus_skill` selects relevant bodies. This instruction-policy difference is part of the experiment.
- Skill choice and evidence routing are model-driven and can vary between runs.
- Live weather, external tool responses, provider behavior, and the indexed corpus can affect results. Record configurations and collection dates for reported experiments.
- LLMLingua loads its compressor lazily, so the first applicable request may include model-loading overhead.
- Keep the model, corpus, instruction files, compression settings, and measurement version consistent across comparison batches.

## On-demand skills

Skill files are stored under `app/skills/`:

```text
app/skills/
├── posts-analysis/SKILL.md
├── vector-db-analysis/SKILL.md
└── weather-analysis/SKILL.md
```

Each file contains YAML `name` and `description` metadata followed by instruction text. For `toon_plus_skill`, a structured model call sees only the question and compact skill metadata, selects zero or more skills, and loads their instruction bodies. A separate answer call receives those instructions and the TOON evidence. The large evidence is sent only to the answer call.

Skill selection does not decide which MCP tools run; the outer graph performs that routing first. Add a skill by creating a directory containing `SKILL.md` whose YAML `name` matches the directory name.

## Repository structure

```text
.
├── app/
│   ├── agents/                 # LangGraph state, routing, and workflow nodes
│   ├── api/v1/                 # Versioned FastAPI routes
│   ├── benchmarks/prompts.py   # Research benchmark questions and routing labels
│   ├── core/config.py          # Environment-backed settings
│   ├── mcp/                    # MCP client and response interception
│   ├── schemas/                # Request and routing schemas
│   ├── scripts/                # PDF ingestion and benchmark runner
│   ├── services/               # Retrieval, compression, skills, storage, evaluation
│   ├── skills/                 # Domain-specific instruction files
│   └── main.py                 # FastAPI application and database lifecycle
├── mcp_servers/toon_server/    # Local tools for users, posts, and weather
├── tests/                     # Offline automated checks
├── pyproject.toml             # Project dependencies and Python requirement
├── uv.lock                    # Locked dependency versions
└── README.md
```

## Tests

Run the offline checks for routing, experiment-arm isolation, skill loading, token accounting, and run evaluation:

```bash
uv run python -m unittest discover -s tests -v
```

These checks do not replace live validation of Vertex AI credentials, Pinecone retrieval, MongoDB connectivity, or external tool availability.

## Troubleshooting

| Issue | What to check |
| --- | --- |
| `GOOGLE_CLOUD_PROJECT must be set for Vertex AI` | Set the project in the root `.env` file before starting the application. |
| MongoDB startup failure | Verify that MongoDB is running and `MONGODB_URI` is reachable. |
| Pinecone configuration or retrieval failure | Set the API key, index name, and embedding model; verify index compatibility and ingestion. |
| MCP subprocess cannot find `uv` | Start the API from the repository root with `uv` available on `PATH`. |
| Slow first LLMLingua request | Allow the compressor weights to download and load; account for warm-up when benchmarking. |
| Missing evaluation results | Inspect the stored `ragas.status` and error details; evaluation runs in the background. |
| Context recall is null | Supply a `reference_answer` for reference-based evaluation. |

## Academic use

This project supports BSc research into token-efficient LLM workflows. When reporting results, state the benchmark batch, model and configuration, corpus, measurement version, and evaluation coverage. Cite the underlying methods and services separately from this implementation.
