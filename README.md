## MongoDB run storage

The API stores one document in the `runs` collection after every successful
`POST /api/v1/agent/chat` execution. It uses PyMongo's native
`AsyncMongoClient`, which is the current MongoDB async Python driver API.

Configure the connection in `.env`:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=token_compression
MONGODB_RUNS_COLLECTION=runs
```

For a local MongoDB instance, run:

```bash
docker run --name token-compression-mongo --publish 27017:27017 --detach mongo:8
```

The request can include `run_id`, `prompt_id`, `task_type`, `arm`, `repeat`,
`model`, and `ragas`. `run_id` is indexed uniquely, so retrying the same run
ID is rejected instead of creating a duplicate measurement.

The driver lifecycle and async client usage follow the current MongoDB Python
driver documentation:
https://www.mongodb.com/docs/languages/python/pymongo-driver/current/connect/mongoclient/

## Ragas evaluation

Each completed run is evaluated in the background for its selected `arm`. The
MongoDB document is first written with `ragas.status: "pending"`, then updated
with `faithfulness`, `relevance`, and `answer_relevancy` scores. Faithfulness
is evaluated when the run has retrieved context; direct arms still receive
answer relevancy evaluation.

Evaluation uses plain answer text (excluding content-block metadata) and the
original, uncompressed document and MCP tool evidence for every arm. This lets
faithfulness check tool-derived claims as well as document-derived claims.
Existing stored scores are unchanged; run requests again to obtain corrected scores.

The evaluator uses Ragas's modern `Faithfulness` and `AnswerRelevancy`
collection metrics with the configured Google GenAI model. Evaluation errors
are stored as `ragas.status: "failed"` and do not delay the frontend response.

## On-demand skills with Deep Agents

All application chat models use `ChatGoogleGenerativeAI` with `vertexai=True`
through `app/services/llm_service.py`. This includes routing, direct generation,
Deep Agents skill generation, and Ragas evaluation. Configure
`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_LLM_MODEL`, and optionally
`GOOGLE_API_KEY` for your Vertex AI authentication setup. Without an API key,
the client uses its Google credential resolution. `DEFAULT_MODEL_TEMPERATURE`
applies to all these calls. Legacy OpenRouter settings are ignored.

Use Python 3.11 or newer and install the locked dependencies with `uv sync`.
Only the `toon_plus_skill` arm uses model-directed skill selection with the
configured Gemini/Vertex AI model. The `toon_skill` value uses
TOON encoding with direct generation, as does `toon_only`. All other arms use
direct generation without skill discovery. The skill service is imported only
inside the `toon_plus_skill` generation branch.

The outer graph first produces a structured evidence plan from the submitted
question. The same router is used for every arm and selects document retrieval
and only the required MCP tools. No benchmark file or expected-routing labels
are passed to the model. For example, P098 needs document retrieval and users,
but not posts or weather. A question needing no MCP tools skips MCP connection.
Selected tool failures or unavailable tools are reported in the evidence.
Each arm then applies its transformation to that evidence:

| Request `process_type` | Evidence sent to the model | Instructions |
| --- | --- | --- |
| `normal_rag`, `naive_rag` | Uncompressed document text and JSON tool results | All three full skill files |
| `llmlingua_only` | LLMLingua-compressed document prose + uncompressed JSON tool results | All three full skill files |
| `toon_only`, `toon_skill` | TOON-encoded document and tool context | All three full skill files |
| `toon_plus_skill` | Same TOON encoding, supplied only for answering | Model-selected files read on demand |

LLMLingua receives only retrieved document prose, with the original user question
as guidance. MCP JSON (including text inside JSON fields) bypasses compression;
MCP-only requests do not invoke LLMLingua. Core system instructions, the original
user message, and the generated answer are not compressed. TOON is a
serialization format; it does not guarantee token savings for prose or every
payload. Skills provide instructions rather than another evidence-compression
algorithm. MCP fetch time is recorded separately as `runtime_ms.tools`.

Skills follow the directory layout documented in the
[official Deep Agents skills guide](https://docs.langchain.com/oss/python/deepagents/skills):

```text
app/skills/
  weather-analysis/SKILL.md
  posts-analysis/SKILL.md
  vector-db-analysis/SKILL.md
```

Each file starts with YAML `name` and `description` fields. The app reads only
these compact metadata fields to build a skill registry. A structured model call
chooses relevant skill names from the user's question; the app then loads only
those files' instruction bodies. The model can select multiple skills for a
combined request or none when no skill applies. Skill choice is model-driven,
not a guarantee that a particular request will activate a particular file.

Skill selection sees only the question and compact skill metadata, never the
large document/MCP evidence. It makes one structured call without filesystem
tools. One answer call then receives the original question, only the selected
instruction bodies, and the TOON evidence. Selection output is not used as
answer evidence, and the large evidence is sent only once. This removes the
multi-turn read-file discovery loop while preserving model-directed, on-demand
skill loading.

Adding a skill requires a new directory whose name matches the YAML `name`,
containing `SKILL.md`; no selector changes are needed. The existing outer graph
supplies Pinecone context and MCP results.
Skill discovery does not change which MCP tools that graph executes.

Token totals include routing, the single skill-selection call, and answering.
The API returns `token_breakdown.calls`, `token_breakdown.by_phase`, and
`token_breakdown.context`, as well as `selected_tools` and `loaded_skills`.
MongoDB stores the same breakdown in `tokens.calls`, `tokens.by_phase`, and
`tokens.context_measurement`. Each call has provider-reported input/output/total
tokens, a phase (`routing`, `skill_selection`, or `answer`), and a
`usage_available` flag. Missing provider usage is marked rather than estimated.

`tokens.context_raw` and `tokens.context_after_compression` measure evidence
alone using the configured model's provider count-tokens API. Identical contexts
are counted once. These extra counting requests are not generation calls;
their latency is recorded separately as `latency_ms.token_counting`. If counting
fails, context counts are null with status `unavailable`, never fabricated zeros.
Background RAGAS evaluation usage is outside these workflow totals.

New records carry `tokens.measurement_version: 2`. Older records omitted routing
tokens and used different prompts/workflows; rerun all arms for comparisons.
Compare both evidence compression and total workflow cost: on-demand loading
still has overhead and is not guaranteed to save total tokens on every question.
All other arms receive the complete posts-analysis, vector-db-analysis, and
weather-analysis SKILL.md instructions directly in their system prompt. The
`toon_plus_skill` arm alone loads relevant instructions on demand. This adds a
fixed prompt-token cost to non-loader arms, so compare new benchmark runs with
each other rather than with runs made before this prompt change.

Run offline integration checks with:

```bash
uv run python -m unittest discover -s tests -v
```
