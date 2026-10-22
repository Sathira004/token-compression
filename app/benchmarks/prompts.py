"""
Benchmarking Prompts for Token Compression LangGraph System
============================================================

100 prompts organized by task category to test all execution paths:

CATEGORIES & EXPECTED ROUTING
─────────────────────────────
1. DIRECT_ANSWER       → route_query returns "generate_response" (no retrieval)
                         LLM answers from its own knowledge, no Pinecone hit.

2. VECTOR_DB_ONLY      → route_query returns "retrieve_documents"
                         Pinecone RAG retrieval → optimization → generation.
                         Tests the document knowledge base (ONTO.pdf content).

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
    # ║  Prompts target ONTO.pdf ontology/domain content                    ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    {
        "id": "P016",
        "prompt": "What are the main concepts defined in the ontology document?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Top-level ontology overview – requires ONTO.pdf retrieval",
    },
    {
        "id": "P017",
        "prompt": "Explain the class hierarchy described in the knowledge base.",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Ontology structure question – Pinecone retrieval",
    },
    {
        "id": "P018",
        "prompt": "What relationships exist between the entities in the ontology?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Entity relationship extraction from domain docs",
    },
    {
        "id": "P019",
        "prompt": "What properties are defined for the primary classes in the document?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Property-level detail from ONTO.pdf",
    },
    {
        "id": "P020",
        "prompt": "Summarize the key findings from the research document in the knowledge base.",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Document summary – requires chunk retrieval",
    },
    {
        "id": "P021",
        "prompt": "What methodology is described in the ingested documents?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Methodology extraction from domain knowledge",
    },
    {
        "id": "P022",
        "prompt": "Are there any constraints or rules mentioned in the ontology?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Constraint extraction – requires retrieval",
    },
    {
        "id": "P023",
        "prompt": "What are the subclasses defined under the main category in the document?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Hierarchical structure – Pinecone search",
    },
    {
        "id": "P024",
        "prompt": "Provide a detailed explanation of the domain model from the knowledge base.",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Domain model deep dive – multi-chunk retrieval",
    },
    {
        "id": "P025",
        "prompt": "What examples or use cases are described in the indexed documents?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Use case extraction from knowledge base",
    },
    {
        "id": "P026",
        "prompt": "What are the data types and their definitions in the ontology?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Data type definitions – targeted retrieval",
    },
    {
        "id": "P027",
        "prompt": "How does the document define the scope of the knowledge representation?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Scope definition retrieval from ONTO.pdf",
    },
    {
        "id": "P028",
        "prompt": "What taxonomic classifications are present in the indexed content?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Taxonomy extraction – semantic search",
    },
    {
        "id": "P029",
        "prompt": "List the main sections and their purposes from the document.",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Document structure overview – retrieval needed",
    },
    {
        "id": "P030",
        "prompt": "What conclusions or recommendations does the knowledge base contain?",
        "category": TaskCategory.VECTOR_DB_ONLY,
        "expected_tools": [],
        "needs_retrieval": True,
        "description": "Conclusion extraction from indexed docs",
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
        "prompt": "Using the knowledge base, explain the ontology structure and also show me all registered users.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": True,
        "description": "Ontology retrieval + user data – hybrid pipeline",
    },
    {
        "id": "P082",
        "prompt": "What does the document say about entity relationships? Also fetch the weather in Colombo.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Doc retrieval + weather – cross-source",
    },
    {
        "id": "P083",
        "prompt": "Summarize the ontology classes from the knowledge base and list all blog posts from the system.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": True,
        "description": "Ontology summary + posts – retrieval + MCP",
    },
    {
        "id": "P084",
        "prompt": "Based on the indexed documents, what is the research scope? Also show me user profiles and the weather in Colombo.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users", "mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Research scope + users + weather – full hybrid",
    },
    {
        "id": "P085",
        "prompt": "Retrieve the key concepts from the document and correlate with user activity from the posts.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": True,
        "description": "Concept retrieval + post correlation",
    },
    {
        "id": "P086",
        "prompt": "What methodology does the knowledge base describe? Include a weather check for Sri Lanka.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Methodology + weather – retrieval + single MCP",
    },
    {
        "id": "P087",
        "prompt": "Show me the ontology constraints from the document, all users, and their posts in one report.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users", "mcp_get_posts"],
        "needs_retrieval": True,
        "description": "Document constraints + users + posts – 3 sources",
    },
    {
        "id": "P088",
        "prompt": "Create a comprehensive overview: document findings, user directory, all posts, and current weather.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users", "mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Full system test – retrieval + all 3 MCP tools",
    },
    {
        "id": "P089",
        "prompt": "What are the taxonomic categories in the knowledge base? And how many registered users are there?",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": True,
        "description": "Taxonomy retrieval + user count – hybrid",
    },
    {
        "id": "P090",
        "prompt": "Explain the domain model from indexed documents while also fetching current Colombo weather data.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Domain model + weather – knowledge + live data",
    },
    {
        "id": "P091",
        "prompt": "Search the knowledge base for data type definitions and also retrieve all posts by user ID 3.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_posts"],
        "needs_retrieval": True,
        "description": "Data types + filtered posts – retrieval + MCP",
    },
    {
        "id": "P092",
        "prompt": "What properties are described in the ontology? Show that alongside the user who works at 'Deckow-Crist'.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": True,
        "description": "Ontology properties + specific user lookup",
    },
    {
        "id": "P093",
        "prompt": "Retrieve the document conclusions from the knowledge base. Also give me all posts and weather in Colombo.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Conclusions + posts + weather – 3-source hybrid",
    },
    {
        "id": "P094",
        "prompt": "Using the indexed documents, describe the class hierarchy. Then list the first 5 users with their emails.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": True,
        "description": "Class hierarchy + user email listing",
    },
    {
        "id": "P095",
        "prompt": "What examples are in the knowledge base? Also show the latest blog content and Sri Lanka weather.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Examples + posts + weather – multi-source",
    },
    {
        "id": "P096",
        "prompt": "Full system check: retrieve document overview from Pinecone, fetch all users, all posts, and weather for Colombo.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users", "mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Maximum pipeline stress test – all components",
    },
    {
        "id": "P097",
        "prompt": "What is the scope of the indexed research? Cross-reference with the number of blog posts and user count.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users", "mcp_get_posts"],
        "needs_retrieval": True,
        "description": "Research scope + cross-reference MCP data",
    },
    {
        "id": "P098",
        "prompt": "Show ontology subclasses from the document and also fetch user phone numbers.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users"],
        "needs_retrieval": True,
        "description": "Subclasses + user contacts – hybrid retrieval",
    },
    {
        "id": "P099",
        "prompt": "Give me the document's key findings and recommendations, plus all system users and the weather in Colombo.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users", "mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Findings + users + weather – 3-source integration",
    },
    {
        "id": "P100",
        "prompt": "Build a complete system diagnostic: ontology analysis from the knowledge base, full user directory, all posts content, and real-time weather data for Colombo, Sri Lanka.",
        "category": TaskCategory.VECTOR_DB_PLUS_MCP,
        "expected_tools": ["mcp_get_users", "mcp_get_posts", "mcp_get_weather"],
        "needs_retrieval": True,
        "description": "Ultimate benchmark – every pipeline path exercised",
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

