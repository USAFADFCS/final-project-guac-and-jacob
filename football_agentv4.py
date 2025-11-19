import asyncio
import os
import textwrap

from dotenv import load_dotenv
load_dotenv()

"""
Documentation:

THIS FILE DOESNT WORK

Earlier documentation for football_agent.py and football_agentv2.py:

Early development, figuring out API's:
https://chatgpt.com/share/68f94381-0474-8006-acd7-d1ed740df7c5

Heavy modification from the original code with help from GPT-5. Transcript link:

https://chatgpt.com/share/68f942d9-0f78-8006-a0ef-3680d03b3359

Later development. making a more robust version of this system and dropping JSON mode:
https://chatgpt.com/share/6907edc9-7e4c-8006-8af0-0cdc7474d42d

Heavy inspiration for this model is from from demo_multi_agent.py.

Attempting to implement a vegas and weather tool. Abandoned weather because it is too
Complicated:
https://chatgpt.com/share/691113fd-6b80-8006-a4cf-2943edc72527

"""

# fairlib core
from fairlib import (
    settings,
    OpenAIAdapter,
    ToolRegistry,
    SafeCalculatorTool,
    WebSearcherTool,
    ToolExecutor,
    WorkingMemory,
    ReActPlanner,
    SimpleAgent,
)

from vegas_tool import VegasOddsTool, RealOddsProvider

# =================== CONFIG: ALWAYS PROSE ===================
PROSE_FINAL_SPEC = textwrap.dedent("""
FINAL ANSWER FORMAT (PROSE, MANDATORY):
- Do NOT output 'Final Answer' or any wrapper words.
- Output concise prose using this structure:
  Line 1: VERDICT: <HOLD|DROP|START|SIT|FAIR|FAVORABLE|UNFAVORABLE>
  Line 2-4: Why — 2 to 4 short sentences with concrete numbers (PPG, last-3, trend, injury/schedule) or 'unknown'.
  Line 5 (optional if known): Quick stats — ppg_season=..., last3=..., trend=..., turnovers=...
  Line 6 (optional if available): Sources — ESPN, FantasyData, PFF (short names only).
- No JSON. No extra thoughts or tool calls.
""").strip()

# =================== ENV & API KEYS ===================
settings.api_keys.openai_api_key = os.getenv("OPENAI_API_KEY")
settings.api_keys.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
settings.search_engine.google_cse_search_api = os.getenv("GOOGLE_CSE_SEARCH_API")
settings.search_engine.google_cse_search_engine_id = os.getenv("GOOGLE_CSE_SEARCH_ENGINE_ID")

# =================== TASK CLASSIFICATION ===================
from typing import Literal

TaskType = Literal["trade", "start_sit", "hold_drop"]

def classify_task(user_query: str) -> TaskType:
    q = user_query.lower()
    if ("trade" in q) or ((" for " in q) and not any(x in q for x in ["start", "sit", "bench"])):
        return "trade"
    if ("start" in q) or ("sit" in q):
        return "start_sit"
    return "hold_drop"

# =================== AGENT FACTORY ===================
def create_worker_agent(llm, tools, role_description: str) -> SimpleAgent:
    tool_registry = ToolRegistry()
    for t in tools:
        tool_registry.register_tool(t)
    planner = ReActPlanner(llm, tool_registry)
    executor = ToolExecutor(tool_registry)
    memory = WorkingMemory()
    agent = SimpleAgent(llm, planner, executor, memory, stateless=True)
    agent.role_description = role_description
    return agent

def get_web_searcher_tool(cse_search_api, cse_engine_id):
    web_search_config = {
        "google_api_key": cse_search_api,
        "google_search_engine_id": cse_engine_id,
        "cache_ttl": settings.search_engine.web_search_cache_ttl,
        "cache_max_size": settings.search_engine.web_search_cache_max_size,
        "max_results": settings.search_engine.web_search_max_results,
    }
    return WebSearcherTool(config=web_search_config)

# =================== PROMPTS (WORKERS) ===================
RESEARCHER_PROMPT_BASE = textwrap.dedent("""
You are a fantasy football researcher. Use web_searcher when needed.
You can also call vegas_odds to fetch the real spread/total/moneyline for the next matchup
(e.g., 'Away @ Home' or JSON {{ "home": "...", "away": "..." }}).

USER QUESTION:
{user_query}

TASK:
1) Identify the relevant NFL player(s) and their next opponent.
2) Retrieve concise, fantasy-relevant stats: season PPG, last-3 PPG, role/snap, injury.
3) Fetch real lines via vegas_odds and mention spread/total (and movement if available).
4) Provide a one-sentence ROS outlook per player.
5) Cite 2–3 short sources (names only). Keep under ~12 lines, no tables, no JSON.
If a stat is unavailable, say 'unknown'.
""").strip()



ANALYST_PROMPT_BASE = textwrap.dedent("""
You are a fantasy football analyst. You can use the safe calculator for arithmetic if needed.

CONTEXT (from research):
{research_block}

USER QUESTION:
{user_query}

TASK:
- Extract per-player season PPG and last-3 PPG (or 'unknown').
- For TRADE: compute season-PPG delta (A - B) and last-3 delta (A - B) if both known; else 'unknown'.
- For START/SIT (single or A vs B): determine trend per player as up/flat/down based on last-3 vs season (tolerance ±0.5).
- For HOLD/DROP: estimate replacement baseline (QB=15.0, RB=9.5, WR=10.0, TE=7.5) and compute delta=season_ppg - baseline if known; label trend up/flat/down (±0.5 tolerance).

OUTPUT REQUIREMENTS:
- Short prose summary of the key computed deltas and trends.
- No JSON. Keep it under ~8 lines.
""").strip()

# =================== PROMPTS (FINAL SYNTHESIS) ===================
FINAL_SYNTHESIS_TRADE = textwrap.dedent("""
You are the team manager. Produce the final decision in the required prose format.

USER QUESTION:
{user_query}

RESEARCH SUMMARY:
{research_block}

ANALYST SUMMARY:
{analyst_block}

DECISION RULES (apply sensibly):
- Choose: FAIR | FAVORABLE | UNFAVORABLE (for the user trading away Player A for Player B).
- Consider PPG, last-3 trend, turnovers, injuries/OL notes, near-term schedule, ROS outlook.
- Tie-breaker: if season PPG delta within ±0.5, break ties using trend + schedule; if still tied → FAIR.

Now write ONLY the final answer using this format:
{prose_spec}
""").strip()

FINAL_SYNTHESIS_STARTSIT = textwrap.dedent("""
You are the team manager. Produce the final decision in the required prose format.

USER QUESTION:
{user_query}

RESEARCH SUMMARY:
{research_block}

ANALYST SUMMARY:
{analyst_block}

DECISION RULES:
- Output START if season PPG ≥ 12 or last-3 ≥ season and trend up; penalize material injury/snap loss or tough schedule next 2–3.
- Otherwise SIT.

Now write ONLY the final answer using this format:
{prose_spec}
""").strip()

FINAL_SYNTHESIS_HOLDDROP = textwrap.dedent("""
You are the team manager. Produce the final decision in the required prose format.

USER QUESTION:
{user_query}

RESEARCH SUMMARY:
{research_block}

ANALYST SUMMARY:
{analyst_block}

DECISION RULES:
- HOLD if replacement_delta ≥ +2.0 or trend up with last-3 ≥ season.
- DROP if replacement_delta ≤ 0 and trend down or multiweek OUT.
- Otherwise HOLD.

Now write ONLY the final answer using this format:
{prose_spec}
""").strip()

# =================== MAIN ORCHESTRATION (NO JSON) ===================
async def run_pipeline(user_query: str) -> str:
    # Core LLM for synthesis and to drive workers' planners
    llm = OpenAIAdapter(
        api_key=settings.api_keys.openai_api_key,
        model_name=settings.models["openai_gpt4"].model_name
    )

    # Build workers
    researcher = create_worker_agent(
        llm,
        [get_web_searcher_tool(settings.search_engine.google_cse_search_api,
                               settings.search_engine.google_cse_search_engine_id),
        VegasOddsTool(provider=RealOddsProvider(
            bookmakers=['draftkings','fanduel'],
            window_hours=10*24,
        )),
        ],
        "A research agent that uses a web search tool for player info and vegas_odds for real lines."
    )

    analyst = create_worker_agent(
        llm,
        [SafeCalculatorTool()],
        "An analyst agent that performs mathematical calculations and trend checks for NFL player comparisons."
    )

    # Step 1: classify task
    task_type = classify_task(user_query)

    # Step 2: Research
    research_prompt = RESEARCHER_PROMPT_BASE.format(user_query=user_query)
    research_block = await researcher.arun(research_prompt)

    # Step 3: Analyst
    analyst_prompt = ANALYST_PROMPT_BASE.format(user_query=user_query, research_block=research_block)
    analyst_block = await analyst.arun(analyst_prompt)

    # Step 4: Final synthesis in prose only
    # ...after you build research_block and analyst_block

    # 4) Final synthesis in prose only — use a tool-less SimpleAgent
    synthesizer = create_worker_agent(
        llm,
        [],  # no tools
    "You are the final-decider. Produce ONLY the final prose answer in the required format."
    )

    if task_type == "trade":
        synth_prompt = FINAL_SYNTHESIS_TRADE.format(
            user_query=user_query,
            research_block=research_block,
            analyst_block=analyst_block,
            prose_spec=PROSE_FINAL_SPEC
        )
    elif task_type == "start_sit":
        synth_prompt = FINAL_SYNTHESIS_STARTSIT.format(
            user_query=user_query,
            research_block=research_block,
            analyst_block=analyst_block,
            prose_spec=PROSE_FINAL_SPEC
        )
    else:
        synth_prompt = FINAL_SYNTHESIS_HOLDDROP.format(
            user_query=user_query,
            research_block=research_block,
            analyst_block=analyst_block,
            prose_spec=PROSE_FINAL_SPEC
        )

    final_answer = await synthesizer.arun(synth_prompt)  # ← instead of llm.arun(...)
    return str(final_answer).strip()

# =================== CLI ENTRY ===================
async def main():
    print("=" * 60)
    print("Football Manager (Prose Mode)")
    print("=" * 60)

    user_query = "Should I trade Drake London for Jahmyr Gibbs?"
    print("\nUSER QUERY:", user_query)

    answer = await run_pipeline(user_query)

    print("\n" + "-" * 60)
    print("FINAL PROSE ANSWER")
    print("-" * 60)
    print(answer)
    print("-" * 60)

if __name__ == "__main__":
    asyncio.run(main())

