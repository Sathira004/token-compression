"""
Benchmarking Prompts for Token Compression LangGraph System
============================================================

100 prompts organized by task category to test all execution paths:

CATEGORIES & EXPECTED ROUTING
─────────────────────────────
1. DIRECT_ANSWER       → route_query selects no retrieval and no MCP tools
                         LLM answers from its own knowledge, no Pinecone hit.

2. VECTOR_DB_ONLY      → route_query selects document retrieval only
                         Pinecone RAG retrieval → optimization → generation.
                         Tests ONTO.pdf: Object Notation for Token Optimization.

3. MCP_ALL             → All 3 MCP tools fire (mcp_get_users, mcp_get_posts,
                         mcp_get_weather). Tests full ToonServer integration.

4. MCP_USERS_ONLY      → Only mcp_get_users is relevant.
                         Tests user-data MCP tool in isolation.

5. MCP_POSTS_ONLY      → Only mcp_get_posts is relevant.
                         Tests post-data MCP tool in isolation.

6. MCP_WEATHER_ONLY    → Only mcp_get_weather is relevant.
                         Tests weather MCP tool in isolation.

7. MCP_USERS_POSTS     → mcp_get_users + mcp_get_posts combo.
                         Tests two-tool MCP coordination.

8. MCP_USERS_WEATHER   → mcp_get_users + mcp_get_weather combo.

9. MCP_POSTS_WEATHER   → mcp_get_posts + mcp_get_weather combo.

10. VECTOR_DB_PLUS_MCP → Pinecone retrieval AND MCP tools together.
                         Tests the full pipeline: retrieval + MCP + generation.

Each prompt dict contains:
  - id:            Unique prompt identifier for tracking (e.g., "P001")
  - prompt:        The actual user message to send
  - category:      One of the category constants above
  - expected_tools: List of MCP tools expected to be invoked
  - needs_retrieval: Whether Pinecone vector search should be triggered

Retrieval prompts are grounded in the local ONTO.pdf paper,
"ONTO: A Token-Efficient Columnar Notation for LLM Input Optimization".
They request indexed evidence without embedding the expected answers. Section
and table references identify the source passages; hybrid prompts also request
MCP data. Routing flags describe expected behavior; the router decides at runtime.
Only the submitted prompt is sent to the workflow, never this file or its labels.
  - description:   Short explanation of what this prompt tests
"""

from enum import Enum
from typing import TypedDict


# ── Category Constants ──────────────────────────────────────────────────────

class TaskCategory(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    VECTOR_DB_ONLY = "vector_db_only"
    MCP_ALL = "mcp_all"
    MCP_USERS_ONLY = "mcp_users_only"
    MCP_POSTS_ONLY = "mcp_posts_only"
    MCP_WEATHER_ONLY = "mcp_weather_only"
    MCP_USERS_POSTS = "mcp_users_posts"
    MCP_USERS_WEATHER = "mcp_users_weather"
    MCP_POSTS_WEATHER = "mcp_posts_weather"
    VECTOR_DB_PLUS_MCP = "vector_db_plus_mcp"


# ── Prompt Schema ───────────────────────────────────────────────────────────

class BenchmarkPrompt(TypedDict):
    id: str
    prompt: str
    category: str
    expected_tools: list[str]
    needs_retrieval: bool
    description: str


# ── Benchmark Prompts ───────────────────────────────────────────────────────

BENCHMARK_PROMPTS: list[BenchmarkPrompt] = [
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  CATEGORY 1: DIRECT ANSWER (no retrieval, no MCP)                   ║
    # ║  Expected: route_query → "generate_response", skip Pinecone         ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    {
        "id": "P001",
        "prompt": "What is the capital of France?",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Simple geography fact – LLM general knowledge",
    },
    {
        "id": "P002",
        "prompt": "Explain the difference between a list and a tuple in Python.",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Programming concept – no domain docs needed",
    },
    {
        "id": "P003",
        "prompt": "What is 25 multiplied by 16?",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Arithmetic calculation – pure reasoning",
    },
    {
        "id": "P004",
        "prompt": "Write a haiku about artificial intelligence.",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Creative writing – no external data needed",
    },
    {
        "id": "P005",
        "prompt": "Who invented the telephone?",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Historical fact – general knowledge",
    },
    {
        "id": "P006",
        "prompt": "What are the primary colors?",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Basic science fact",
    },
    {
        "id": "P007",
        "prompt": "Translate 'Good morning' to Japanese.",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Language translation – LLM capability",
    },
    {
        "id": "P008",
        "prompt": "What is the speed of light in meters per second?",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Physics constant – factual recall",
    },
    {
        "id": "P009",
        "prompt": "Give me a Python one-liner to reverse a string.",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Code snippet generation – no docs needed",
    },
    {
        "id": "P010",
        "prompt": "What does the HTTP status code 404 mean?",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Web development concept – general knowledge",
    },
    {
        "id": "P011",
        "prompt": "hi",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Greeting – triggers fast heuristic bypass in route_query",
    },
    {
        "id": "P012",
        "prompt": "hello",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Greeting – fast heuristic shortcut",
    },
    {
        "id": "P013",
        "prompt": "who are you?",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Identity question – fast heuristic bypass",
    },
    {
        "id": "P014",
        "prompt": "thanks",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Polite acknowledgement – fast heuristic shortcut",
    },
    {
        "id": "P015",
        "prompt": "Summarize the benefits of microservice architecture.",
        "category": TaskCategory.DIRECT_ANSWER,
        "expected_tools": [],
        "needs_retrieval": False,
        "description": "Software architecture concept – LLM knowledge",
    },

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  CATEGORY 2: VECTOR DB ONLY (Pinecone retrieval, no MCP)            ║
    # ║  Expected: route_query → "retrieve_documents" → Pinecone search     ║
    # ║  Prompts target ONTO.pdf methods, results, and limitations                    ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    {
        "id": "P016",
        "prompt": "According to the indexed ONTO.pdf paper, what does ONTO stand for, and what problem does its schema-once, data-many design address?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Sections 1 and 3: ONTO definition and repeated-key overhead",
    },
    {
        "id": "P017",
        "prompt": "Retrieve Section 3.1 of ONTO.pdf and explain how the Telemetry example represents device IDs, temperatures, and nested location coordinates.",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Section 3.1: columnar syntax and nested telemetry example",
    },
    {
        "id": "P018",
        "prompt": "According to Section 3.2 of ONTO.pdf, why were pipe delimiters, strict four-space indentation, and explicit record counts chosen?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Section 3.2: syntax design rationale and record-count validation",
    },
    {
        "id": "P019",
        "prompt": "Using Section 3.3 of ONTO.pdf, explain automatic type inference, backtick-forced strings, and how null differs from an empty string.",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Section 3.3: type inference and special-value semantics",
    },
    {
        "id": "P020",
        "prompt": "From Table 2 in ONTO.pdf, report the JSON, YAML, and ONTO token counts and ONTO-versus-JSON reductions for each dataset at 1,000 records.",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Table 2: exact token counts and reductions across three datasets",
    },
    {
        "id": "P021",
        "prompt": "According to Section 4.1 of ONTO.pdf, which three synthetic datasets were evaluated, which contain nesting, and what record scales and random-seed scheme were used?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Section 4.1: datasets, scales, and reproducibility",
    },
    {
        "id": "P022",
        "prompt": "Retrieve Section 4.2 of ONTO.pdf. Which token-counting library, version, and encoding were used, and was cross-tokenizer validation performed?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Section 4.2: tokenizer configuration and generalization limits",
    },
    {
        "id": "P023",
        "prompt": "Using Section 4.3 of ONTO.pdf, describe the hardware, model and quantization, runtime, task, and run protocol used for latency measurements.",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Section 4.3: latency experiment configuration and protocol",
    },
    {
        "id": "P024",
        "prompt": "From Table 4 in ONTO.pdf, compare JSON and ONTO time-to-first-token and total inference time for IoT Telemetry and System Metrics at 1,000 records.",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Table 4: exact latency measurements and improvements",
    },
    {
        "id": "P025",
        "prompt": "According to Section 4.4 of ONTO.pdf, which model, record count, formats, four task types, and answer-scoring rules were used for comprehension validation?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Section 4.4: comprehension experiment and ground-truth scoring",
    },
    {
        "id": "P026",
        "prompt": "Using Table 5 and Section 5.4 of ONTO.pdf, compare cold and warm ONTO comprehension results with JSON, including counting failures and ambiguous host lookups.",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Table 5 and Section 5.4: task accuracy and benchmark caveats",
    },
    {
        "id": "P027",
        "prompt": "Retrieve Table 3 of ONTO.pdf and explain how field names, punctuation, values, indentation, and whitespace contribute to token savings or overhead.",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Table 3: token composition and sources of overhead",
    },
    {
        "id": "P028",
        "prompt": "According to Sections 6.1 and 6.4 of ONTO.pdf, why does flat System Metrics data compress better than nested IoT Telemetry, and what nesting-depth experiments remain future work?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Sections 6.1 and 6.4: indentation overhead and nesting evidence",
    },
    {
        "id": "P029",
        "prompt": "Using Sections 1.2 and 7.1 of ONTO.pdf, explain the intended LLM preprocessing use case and when JSON, YAML, or Parquet should be used instead.",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Sections 1.2 and 7.1: scope, appropriate uses, and non-goals",
    },
    {
        "id": "P030",
        "prompt": "Retrieve Section 7.4 of ONTO.pdf and summarize the limitations concerning synthetic data, a single tokenizer, nesting depth, minification baselines, homogeneous records, and streaming.",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Section 7.4: evaluation and implementation limitations",
    },

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  CATEGORY 3: MCP ALL (all 3 tools: users + posts + weather)         ║
    # ║  Expected: All MCP tools fire in generation_node                    ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    {
        "id": "P031",
        "prompt": "Give me a full dashboard summary: list all users, their latest posts, and the current weather in Colombo.",
        "category": TaskCategory.MCP_ALL,
        "expected_tools": ["mcp_get_users", "mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Full MCP pipeline – all 3 tools needed for dashboard",
    },
    {
        "id": "P032",
        "prompt": "Create a report combining user profiles, all blog posts, and today's weather conditions for Sri Lanka.",
        "category": TaskCategory.MCP_ALL,
        "expected_tools": ["mcp_get_users", "mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Comprehensive report – triggers all MCP tools",
    },
    {
        "id": "P033",
        "prompt": "Show me everything you can fetch: users, posts, and weather data all at once.",
        "category": TaskCategory.MCP_ALL,
        "expected_tools": ["mcp_get_users", "mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Explicit all-data request – full MCP execution",
    },
    {
        "id": "P034",
        "prompt": "I need a daily briefing with team member info, content updates, and weather forecast for Colombo.",
        "category": TaskCategory.MCP_ALL,
        "expected_tools": ["mcp_get_users", "mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Daily briefing format – all tools required",
    },
    {
        "id": "P035",
        "prompt": "Generate a newsletter draft that includes user spotlights, recent articles, and a weather widget for Colombo, Sri Lanka.",
        "category": TaskCategory.MCP_ALL,
        "expected_tools": ["mcp_get_users", "mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Newsletter generation – all MCP data sources",
    },

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  CATEGORY 4: MCP USERS ONLY (mcp_get_users)                        ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    {
        "id": "P036",
        "prompt": "List all the users available in the system.",
        "category": TaskCategory.MCP_USERS_ONLY,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": False,
        "description": "Direct user listing – single MCP tool",
    },
    {
        "id": "P037",
        "prompt": "How many users are registered? Show their names and emails.",
        "category": TaskCategory.MCP_USERS_ONLY,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": False,
        "description": "User count and details – users tool only",
    },
    {
        "id": "P038",
        "prompt": "Show me the user who works at the company named 'Romaguera-Crona'.",
        "category": TaskCategory.MCP_USERS_ONLY,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": False,
        "description": "Filtered user lookup by company name",
    },
    {
        "id": "P039",
        "prompt": "What are the usernames and websites of all registered users?",
        "category": TaskCategory.MCP_USERS_ONLY,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": False,
        "description": "User metadata extraction – users tool only",
    },
    {
        "id": "P040",
        "prompt": "Find the user with email 'Sincere@april.biz' and show their full profile.",
        "category": TaskCategory.MCP_USERS_ONLY,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": False,
        "description": "Specific user lookup by email",
    },
    {
        "id": "P041",
        "prompt": "Which users have addresses in the city of 'Gwenborough'?",
        "category": TaskCategory.MCP_USERS_ONLY,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": False,
        "description": "User filtering by city – address data",
    },
    {
        "id": "P042",
        "prompt": "List all user phone numbers from the system.",
        "category": TaskCategory.MCP_USERS_ONLY,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": False,
        "description": "Phone number extraction – users tool",
    },
    {
        "id": "P043",
        "prompt": "What companies are the users associated with? Show company names and catch phrases.",
        "category": TaskCategory.MCP_USERS_ONLY,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": False,
        "description": "Company data from user profiles",
    },
    {
        "id": "P044",
        "prompt": "Show me a table of all users with their name, email, and city.",
        "category": TaskCategory.MCP_USERS_ONLY,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": False,
        "description": "Tabular user summary – formatting test",
    },
    {
        "id": "P045",
        "prompt": "How many unique companies do the registered users belong to?",
        "category": TaskCategory.MCP_USERS_ONLY,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": False,
        "description": "Aggregation query on user company data",
    },

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  CATEGORY 5: MCP POSTS ONLY (mcp_get_posts)                        ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    {
        "id": "P046",
        "prompt": "Show me all the blog posts available in the system.",
        "category": TaskCategory.MCP_POSTS_ONLY,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": False,
        "description": "Full posts listing – single MCP tool",
    },
    {
        "id": "P047",
        "prompt": "How many total posts are there? List the first 5 post titles.",
        "category": TaskCategory.MCP_POSTS_ONLY,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": False,
        "description": "Post count and sample – posts tool only",
    },
    {
        "id": "P048",
        "prompt": "Fetch the posts written by user ID 1 and summarize them.",
        "category": TaskCategory.MCP_POSTS_ONLY,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": False,
        "description": "Filtered posts by userId",
    },
    {
        "id": "P049",
        "prompt": "What is the title and body of post number 42?",
        "category": TaskCategory.MCP_POSTS_ONLY,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": False,
        "description": "Specific post lookup by ID",
    },
    {
        "id": "P050",
        "prompt": "Summarize the themes across all blog posts in the system.",
        "category": TaskCategory.MCP_POSTS_ONLY,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": False,
        "description": "Thematic analysis of post content",
    },
    {
        "id": "P051",
        "prompt": "Show me posts that contain the word 'qui' in their title.",
        "category": TaskCategory.MCP_POSTS_ONLY,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": False,
        "description": "Keyword filtering on post titles",
    },
    {
        "id": "P052",
        "prompt": "What are the longest posts by body text? Show top 3.",
        "category": TaskCategory.MCP_POSTS_ONLY,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": False,
        "description": "Post length analysis – content processing",
    },
    {
        "id": "P053",
        "prompt": "Group all posts by userId and tell me which user wrote the most.",
        "category": TaskCategory.MCP_POSTS_ONLY,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": False,
        "description": "Posts aggregation by author",
    },
    {
        "id": "P054",
        "prompt": "List the titles of the last 10 posts in the system.",
        "category": TaskCategory.MCP_POSTS_ONLY,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": False,
        "description": "Recent posts listing – tail retrieval",
    },
    {
        "id": "P055",
        "prompt": "Show me all posts from user ID 5 with their full body text.",
        "category": TaskCategory.MCP_POSTS_ONLY,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": False,
        "description": "Author-specific full content retrieval",
    },

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  CATEGORY 6: MCP WEATHER ONLY (mcp_get_weather)                     ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    {
        "id": "P056",
        "prompt": "What is the current weather in Colombo, Sri Lanka?",
        "category": TaskCategory.MCP_WEATHER_ONLY,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Colombo weather – default coordinates (6.9271, 79.8612)",
    },
    {
        "id": "P057",
        "prompt": "Tell me the current temperature and wind speed in Colombo.",
        "category": TaskCategory.MCP_WEATHER_ONLY,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Specific weather metrics – temperature + wind",
    },
    {
        "id": "P058",
        "prompt": "Is it raining right now in Colombo? Check the precipitation data.",
        "category": TaskCategory.MCP_WEATHER_ONLY,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Precipitation check – weather tool only",
    },
    {
        "id": "P059",
        "prompt": "What is the humidity level right now in Colombo, Sri Lanka?",
        "category": TaskCategory.MCP_WEATHER_ONLY,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Humidity data extraction – weather API",
    },
    {
        "id": "P060",
        "prompt": "Should I carry an umbrella today in Colombo? Check the weather conditions.",
        "category": TaskCategory.MCP_WEATHER_ONLY,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Practical weather advice – requires live data",
    },
    {
        "id": "P061",
        "prompt": "Give me a weather summary for the Colombo region including all available metrics.",
        "category": TaskCategory.MCP_WEATHER_ONLY,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Full weather summary – all metrics",
    },
    {
        "id": "P062",
        "prompt": "How windy is it in Colombo right now?",
        "category": TaskCategory.MCP_WEATHER_ONLY,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Wind speed check – single metric",
    },
    {
        "id": "P063",
        "prompt": "What does the current weather data say about conditions near latitude 6.93 longitude 79.86?",
        "category": TaskCategory.MCP_WEATHER_ONLY,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Coordinate-based weather query – explicit lat/lon",
    },
    {
        "id": "P064",
        "prompt": "Compare the current temperature with the wind speed in Colombo – is it pleasant outside?",
        "category": TaskCategory.MCP_WEATHER_ONLY,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Multi-metric weather analysis",
    },
    {
        "id": "P065",
        "prompt": "Fetch the latest weather reading for Sri Lanka and present it in a structured format.",
        "category": TaskCategory.MCP_WEATHER_ONLY,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Structured weather output formatting test",
    },

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  CATEGORY 7: MCP USERS + POSTS (2-tool combo)                       ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    {
        "id": "P066",
        "prompt": "Show me all users and their corresponding blog posts.",
        "category": TaskCategory.MCP_USERS_POSTS,
        "expected_tools": ["mcp_get_users", "mcp_get_posts"],
        "needs_retrieval": False,
        "description": "User-to-posts mapping – 2 MCP tools",
    },
    {
        "id": "P067",
        "prompt": "Which user has written the most posts? Show their profile and post titles.",
        "category": TaskCategory.MCP_USERS_POSTS,
        "expected_tools": ["mcp_get_users", "mcp_get_posts"],
        "needs_retrieval": False,
        "description": "Cross-reference users and posts for top author",
    },
    {
        "id": "P068",
        "prompt": "Create a directory listing each user's name, email, and number of posts they've written.",
        "category": TaskCategory.MCP_USERS_POSTS,
        "expected_tools": ["mcp_get_users", "mcp_get_posts"],
        "needs_retrieval": False,
        "description": "User directory with post counts – join operation",
    },
    {
        "id": "P069",
        "prompt": "Find the user named 'Leanne Graham' and show all their posts.",
        "category": TaskCategory.MCP_USERS_POSTS,
        "expected_tools": ["mcp_get_users", "mcp_get_posts"],
        "needs_retrieval": False,
        "description": "Named user lookup + post retrieval",
    },
    {
        "id": "P070",
        "prompt": "For each user, show their company name and the titles of their first two posts.",
        "category": TaskCategory.MCP_USERS_POSTS,
        "expected_tools": ["mcp_get_users", "mcp_get_posts"],
        "needs_retrieval": False,
        "description": "Compound user+posts data with company info",
    },

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  CATEGORY 8: MCP USERS + WEATHER (2-tool combo)                     ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    {
        "id": "P071",
        "prompt": "List all users and also tell me the current weather in Colombo.",
        "category": TaskCategory.MCP_USERS_WEATHER,
        "expected_tools": ["mcp_get_users", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "User list + weather – 2 independent MCP tools",
    },
    {
        "id": "P072",
        "prompt": "Show registered user profiles and include today's weather conditions for Sri Lanka.",
        "category": TaskCategory.MCP_USERS_WEATHER,
        "expected_tools": ["mcp_get_users", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "User profiles + weather data side by side",
    },
    {
        "id": "P073",
        "prompt": "How many users are there, and what's the temperature in Colombo right now?",
        "category": TaskCategory.MCP_USERS_WEATHER,
        "expected_tools": ["mcp_get_users", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Count users + weather metric – dual query",
    },
    {
        "id": "P074",
        "prompt": "Create a morning digest showing the team members and current weather conditions in Colombo.",
        "category": TaskCategory.MCP_USERS_WEATHER,
        "expected_tools": ["mcp_get_users", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Morning digest format – users + weather",
    },
    {
        "id": "P075",
        "prompt": "Show me the user directory along with a weather advisory for Colombo, Sri Lanka.",
        "category": TaskCategory.MCP_USERS_WEATHER,
        "expected_tools": ["mcp_get_users", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Directory + advisory combo",
    },

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  CATEGORY 9: MCP POSTS + WEATHER (2-tool combo)                     ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    {
        "id": "P076",
        "prompt": "Show me the latest blog posts and the current weather in Colombo.",
        "category": TaskCategory.MCP_POSTS_WEATHER,
        "expected_tools": ["mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Posts + weather – 2 independent data sources",
    },
    {
        "id": "P077",
        "prompt": "Give me a content update with all posts and also include the weather forecast for Colombo.",
        "category": TaskCategory.MCP_POSTS_WEATHER,
        "expected_tools": ["mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Content update + weather integration",
    },
    {
        "id": "P078",
        "prompt": "How many posts exist and is it hot in Colombo today?",
        "category": TaskCategory.MCP_POSTS_WEATHER,
        "expected_tools": ["mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Post count + temperature – dual query",
    },
    {
        "id": "P079",
        "prompt": "Create an afternoon update: summarize blog activity and current weather conditions in Sri Lanka.",
        "category": TaskCategory.MCP_POSTS_WEATHER,
        "expected_tools": ["mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Afternoon update format – posts + weather",
    },
    {
        "id": "P080",
        "prompt": "Fetch all articles and the current wind speed in Colombo, then present them together.",
        "category": TaskCategory.MCP_POSTS_WEATHER,
        "expected_tools": ["mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": False,
        "description": "Articles + specific weather metric combo",
    },

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  CATEGORY 10: VECTOR DB + MCP (retrieval AND MCP tools combined)    ║
    # ║  Expected: Pinecone retrieval + MCP tool execution in generation    ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    {
        "id": "P081",
        "prompt": "Using the indexed ONTO.pdf paper, explain the schema-once, data-many design in Section 3. Also show all registered users.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": True,
        "description": "Section 3: schema-once design plus user directory",
    },
    {
        "id": "P082",
        "prompt": "According to Section 3.1 of ONTO.pdf, how are nested location coordinates represented in the Telemetry example? Also fetch the current weather in Colombo.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Section 3.1: nested telemetry syntax plus weather",
    },
    {
        "id": "P083",
        "prompt": "Retrieve the three token-saving mechanisms described in Section 3.1 of ONTO.pdf, and list all blog posts from the system.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": True,
        "description": "Section 3.1: schema, delimiters, and hierarchy plus posts",
    },
    {
        "id": "P084",
        "prompt": "Based on Sections 1.2 and 7.1 of ONTO.pdf, explain the research scope and intended use cases. Also show user profiles and the current weather in Colombo.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users", "mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Sections 1.2 and 7.1: scope plus users and weather",
    },
    {
        "id": "P085",
        "prompt": "Retrieve ONTO.pdf Section 7.2 and explain its preprocessing integration pattern. Also fetch all blog posts and report how many posts each user ID has.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": True,
        "description": "Section 7.2: integration pattern plus post aggregation",
    },
    {
        "id": "P086",
        "prompt": "According to Section 4.3 of ONTO.pdf, which model, quantization, runtime, and run protocol were used for latency testing? Also fetch the current weather in Colombo, Sri Lanka.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Section 4.3: latency methodology plus weather",
    },
    {
        "id": "P087",
        "prompt": "Using Section 7.4 of ONTO.pdf, explain the homogeneous-record requirement and lack of a streaming parser. Also list all users and their posts.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users", "mcp_get_posts"],
        "needs_retrieval": True,
        "description": "Section 7.4: implementation limitations plus users and posts",
    },
    {
        "id": "P088",
        "prompt": "From Table 2 of ONTO.pdf, report the token reductions for all three datasets at 1,000 records. Also fetch the user directory, all posts, and current weather in Colombo.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users", "mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Table 2: token efficiency plus all three MCP tools",
    },
    {
        "id": "P089",
        "prompt": "According to Section 4.1 of ONTO.pdf, which datasets were evaluated and which include nested fields? Also report the number of registered users.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": True,
        "description": "Section 4.1: dataset structures plus user count",
    },
    {
        "id": "P090",
        "prompt": "Retrieve Section 3.2 of ONTO.pdf and explain the reasons for pipe delimiters and fixed indentation. Also fetch current Colombo weather data.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Section 3.2: delimiter and indentation rationale plus weather",
    },
    {
        "id": "P091",
        "prompt": "Using Section 3.3 of ONTO.pdf, explain inferred types and backtick-forced strings. Also retrieve all posts by user ID 3.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": True,
        "description": "Section 3.3: type inference plus filtered posts",
    },
    {
        "id": "P092",
        "prompt": "According to ONTO.pdf Section 3.3, how are null and empty strings distinguished? Also show the user who works at 'Deckow-Crist'.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": True,
        "description": "Section 3.3: null and empty-string semantics plus user lookup",
    },
    {
        "id": "P093",
        "prompt": "From Table 4 of ONTO.pdf, report JSON and ONTO latency for IoT Telemetry at 1,000 records. Also fetch all posts and current weather in Colombo.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Table 4: IoT latency comparison plus posts and weather",
    },
    {
        "id": "P094",
        "prompt": "Using the Telemetry example in Section 3.1 of ONTO.pdf, explain entity declarations, record counts, and field-value alignment. Then list the first five users with their emails.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": True,
        "description": "Section 3.1: entity syntax plus user emails",
    },
    {
        "id": "P095",
        "prompt": "According to Section 3.2 of ONTO.pdf, how does the caret delimiter represent arrays within a value? Also list all blog posts and fetch current weather in Colombo, Sri Lanka.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Section 3.2: array syntax plus posts and weather",
    },
    {
        "id": "P096",
        "prompt": "Retrieve Table 3 of ONTO.pdf and explain the contributions of key elimination, punctuation, and indentation to token savings. Also fetch all users, all posts, and current weather for Colombo.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users", "mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Table 3: token composition plus all three MCP tools",
    },
    {
        "id": "P097",
        "prompt": "According to ONTO.pdf Section 4.4, what are the four comprehension task types and their scoring rules? Also report the total user count and the number of posts per user.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users", "mcp_get_posts"],
        "needs_retrieval": True,
        "description": "Section 4.4: comprehension protocol plus user and post counts",
    },
    {
        "id": "P098",
        "prompt": "Using Sections 6.1 and 6.4 of ONTO.pdf, explain the reduction gap between flat and nested datasets and the limits of the nesting-depth evidence. Also fetch user phone numbers.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": True,
        "description": "Sections 6.1 and 6.4: nesting overhead plus user contacts",
    },
    {
        "id": "P099",
        "prompt": "According to ONTO.pdf Section 5.4 and Table 5, what counting and lookup limitations were observed across formats? Also fetch all system users and current weather in Colombo.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users", "mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Section 5.4 and Table 5: comprehension caveats plus users and weather",
    },
    {
        "id": "P100",
        "prompt": "Using Section 7.4 of the indexed ONTO.pdf paper, summarize the evaluation limitations and missing baselines. Also fetch the full user directory, all posts, and current weather for Colombo, Sri Lanka.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users", "mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Section 7.4: limitations plus all three MCP tools",
    },
]


# ── Helper Functions ────────────────────────────────────────────────────────

def get_prompts_by_category(category: TaskCategory) -> list[BenchmarkPrompt]:
    """Filter prompts by their task category."""
    return [p for p in BENCHMARK_PROMPTS if p["category"] == category]


def get_prompt_by_id(prompt_id: str) -> BenchmarkPrompt | None:
    """Look up a single prompt by its ID (e.g., 'P001')."""
    for p in BENCHMARK_PROMPTS:
        if p["id"] == prompt_id:
            return p
    return None


def get_category_distribution() -> dict[str, int]:
    """Return a count of prompts per category."""
    dist: dict[str, int] = {}
    for p in BENCHMARK_PROMPTS:
        cat = p["category"]
        dist[cat] = dist.get(cat, 0) + 1
    return dist


def get_retrieval_prompts() -> list[BenchmarkPrompt]:
    """Return only prompts that require Pinecone vector retrieval."""
    return [p for p in BENCHMARK_PROMPTS if p["needs_retrieval"]]


def get_mcp_prompts() -> list[BenchmarkPrompt]:
    """Return only prompts that require at least one MCP tool."""
    return [p for p in BENCHMARK_PROMPTS if p["expected_tools"]]


# ── Quick Validation ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Total benchmark prompts: {len(BENCHMARK_PROMPTS)}")
    print(f"\nCategory distribution:")
    for cat, count in sorted(get_category_distribution().items()):
        print(f"  {cat:25s} → {count} prompts")
    print(f"\nRetrieval prompts:  {len(get_retrieval_prompts())}")
    print(f"MCP prompts:        {len(get_mcp_prompts())}")
    print(f"Direct answer:      {len(get_prompts_by_category(TaskCategory.DIRECT_ANSWER))}")

