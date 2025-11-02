import asyncio
import os

"""
This is our first attempt to create an agent to manage a fantasy football team. Just playing around
for now.Adapted from the multi-agent code for our purposes with the help of GPT-5:

Early development, figuring out API's:
https://chatgpt.com/share/68f94381-0474-8006-acd7-d1ed740df7c5

Later development, trying to refine outputs:
https://chatgpt.com/share/68f942d9-0f78-8006-a0ef-3680d03b3359

"""

# --- Step 1: Import all necessary components ---
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
    ManagerPlanner,
    HierarchicalAgentRunner
)

from demos.demo_tools.mock_web_searcher import MockWebSearcherTool

from dotenv import load_dotenv
load_dotenv()

from typing import Literal, List, Dict, Optional
from pydantic import BaseModel, Field, ValidationError
import json
import textwrap


# LOAD API KEYS AND SETTNGS FROM ENV VARS
settings.api_keys.openai_api_key = os.getenv("OPENAI_API_KEY")
settings.api_keys.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
settings.search_engine.google_cse_search_api = os.getenv("GOOGLE_CSE_SEARCH_API")
settings.search_engine.google_cse_search_engine_id = os.getenv("GOOGLE_CSE_SEARCH_ENGINE_ID")

def create_agent(llm, tools, role_description):
    """
    A helper factory function to simplify the creation of worker agents.
    Each agent gets its own tool registry, planner, executor, and memory.
    """
    tool_registry = ToolRegistry()
    for tool in tools:
        tool_registry.register_tool(tool)
    
    planner = ReActPlanner(llm, tool_registry)
    executor = ToolExecutor(tool_registry)
    memory = WorkingMemory()
    
    # create a stateless agent
    agent = SimpleAgent(llm, planner, executor, memory, stateless=True)

    # This custom attribute helps the manager understand the worker's purpose.
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
    
# --- NEW: strict output schemas ---
class TradeVerdict(BaseModel):
    players: Dict[str, str]
    verdict: Literal["FAIR","FAVORABLE","UNFAVORABLE"]
    why: str = Field(min_length=12, max_length=480)
    quick_stats: Dict[str, str]
    contextual_factors: List[str]
    trade_tip: str
    sources: List[Dict[str,str]]

class StartSitVerdict(BaseModel):
    players: Dict[str, str]  # {"start": "<Player A>", "sit": "<Player B>"} or {"candidate": "<Player>"}
    verdict: Literal["START","SIT"]
    why: str = Field(min_length=12, max_length=480)
    quick_stats: Dict[str, str]
    sources: List[Dict[str,str]]

class HoldDropVerdict(BaseModel):
    player: str
    verdict: Literal["HOLD","DROP"]
    why: str = Field(min_length=12, max_length=480)
    quick_stats: Dict[str, str]
    sources: List[Dict[str,str]]

# --- NEW: classify the user query into a task type ---
TaskType = Literal["trade", "start_sit", "hold_drop"]

def classify_task(user_query: str) -> TaskType:
    q = user_query.lower()
    if ("trade" in q) or ((" for " in q) and not any(x in q for x in ["start", "sit", "bench"])):
        return "trade"
    if ("start" in q) or ("sit" in q):
        return "start_sit"
    return "hold_drop"

TRADE_CONTRACT = textwrap.dedent("""
You are the Manager of a team with two workers:
- Researcher (has a web_searcher tool)
- Analyst (has a safe calculator tool)

TASK: Decide a SINGLE actionable verdict about a 1-for-1 fantasy football trade from the user's perspective (trading Player A for Player B).

PROCESS (exactly once each):
1) Extract the two NFL players being exchanged from the user's query.
2) Ask the Researcher once to fetch concise, fantasy-relevant data for both players:
   - Desired per player: season PPG, last-3 PPG, rushing TDs (if RB/QB), interceptions/turnovers (if QB), and one-sentence ROS outlook.
   - Include 2–3 credible links (FantasyData, ESPN, PFF, PlayerProfiler, FFToday, CBS).
   - If unavailable, return "unknown".
3) Ask the Analyst once to compute:
   - PPG difference: A_season_ppg - B_season_ppg
   - Last-3 difference: A_last3 - B_last3
   If any input is 'unknown', output 'unknown' for that diff.
4) Synthesize the decision without deferring.
5) You must call the Researcher exactly once and the Analyst exactly once before you produce the final decision JSON.

DECISION RULE:
- Output one of: FAIR, FAVORABLE, UNFAVORABLE (for the user trading away Player A for Player B).
- Consider PPG, last-3 trend, turnovers, injuries/OL notes, near-term schedule, and ROS outlook.
- Tie-breaker: if season PPG delta within ±0.5, break ties using trend + schedule; if still tied → FAIR.

OUTPUT: Return EXACTLY this JSON (no extra text):
{
  "players": {"from_user_team": "<Player A>", "offered": "<Player B>"},
  "verdict": "FAIR | FAVORABLE | UNFAVORABLE",
  "why": "2-4 sentences with concrete stats or 'unknown' and short source names (e.g., ESPN, FantasyData).",
  "quick_stats": {
    "<Player A>_ppg": "number or 'unknown'",
    "<Player B>_ppg": "number or 'unknown'",
    "last3_trend_<Player A>": "up | flat | down | unknown",
    "last3_trend_<Player B>": "up | flat | down | unknown"
  },
  "contextual_factors": [
    "schedule note",
    "injury/OL note",
    "rushing floor vs passing ceiling"
  ],
  "trade_tip": "one sentence of practical advice",
  "sources": [
    {"name":"FantasyData","url":"..."},
    {"name":"ESPN","url":"..."},
    {"name":"PFF","url":"..."}
  ]
}
""").strip()

STARTSIT_CONTRACT = textwrap.dedent("""
You are the Manager of a team with two workers:
- Researcher (web_searcher)
- Analyst (safe calculator)

TASK: Decide START or SIT for exactly one NFL player (or choose between two if the user clearly presents an A vs B start/sit).

PROCESS (exactly once each):
1) Identify the player(s) implicated (one or two). If two are named, evaluate START for the better and SIT for the other.
2) Ask Researcher once for season PPG, last-3 PPG, snap share/routes run (if available), injury status, and 2 credible links.
3) Ask Analyst once for deltas and a trend assessment (up/flat/down). Unknown inputs → 'unknown'.
4) You must call the Researcher exactly once and the Analyst exactly once before you produce the final decision JSON.
DECISION RULE:
- Output START if season PPG ≥ 12 or last-3 ≥ season and trend up; penalize material injury/snap loss or tough schedule next 2–3.
- Otherwise SIT.

OUTPUT EXACTLY (no extra text):
{
  "players": {"candidate": "<Player>"} OR {"start": "<Player A>", "sit": "<Player B>"},
  "verdict": "START" | "SIT",
  "why": "2-4 sentences citing numbers or 'unknown'.",
  "quick_stats": {...},
  "sources": [{"name":"...","url":"..."}, {"name":"...","url":"..."}]
}
""").strip()

HOLDDROP_CONTRACT = textwrap.dedent("""
You are the Manager of a team with two workers:
- Researcher (web_searcher)
- Analyst (safe calculator)

TASK: Decide HOLD or DROP on exactly one NFL player for a 12-team half-PPR redraft league.

PROCESS (exactly once each):
1) Identify the single player from the user's query.
2) Ask Researcher once for season PPG, last-3 PPG, injury status, role/snap share if available, and 2 credible links.
3) Ask Analyst once to assess trend (up/flat/down) and compare season PPG to an assumed replacement level of:
   - QB: 15.0, RB: 9.5, WR: 10.0, TE: 7.5 (use 'unknown' if position missing).
4) You must call the Researcher exactly once and the Analyst exactly once before you produce the final decision JSON.

DECISION RULE:
- HOLD if replacement_delta ≥ +2.0 or trend up with last-3 ≥ season.
- DROP if replacement_delta ≤ 0 and trend down or multiweek OUT.
- Otherwise HOLD.

OUTPUT EXACTLY (no extra text):
{
  "player": "<Name>",
  "verdict": "HOLD" | "DROP",
  "why": "2-4 short sentences citing PPG, last-3, replacement delta, schedule/injury if relevant.",
  "quick_stats": {"ppg_season": "...", "ppg_last3": "...", "trend": "up|flat|down|unknown", "replacement_delta": "..."},
  "sources": [{"name":"...","url":"..."}, {"name":"...","url":"..."}]
}
""").strip()

MANAGER_ACTION_API = textwrap.dedent("""
ACTION API (MANDATORY):
- To use a tool, output a SINGLE JSON object at TOP LEVEL with EXACTLY:
  {"tool_name": "delegate", "tool_input": {"worker_name": "<Researcher|Analyst>", "task": "<what to do>"}}
- No other keys. No 'Thought'. No 'Action'. No prose before/after.

FINAL ANSWER EMISSION (MANDATORY):
- When you are done, do NOT output 'Final Answer' or any wrapper.
- Output ONLY the final decision JSON specified by the contract. A single JSON object, nothing else.
""").strip()

def _with_action_api(contract: str) -> str:
    return MANAGER_ACTION_API + "\n\n" + contract



# --- NEW: build the manager role_description dynamically ---
def build_manager_prompt(user_query: str) -> str:
    task = classify_task(user_query)
    if task == "trade":
        return _with_action_api(TRADE_CONTRACT)
    if task == "start_sit":
        return _with_action_api(STARTSIT_CONTRACT)
    return _with_action_api(HOLDDROP_CONTRACT)  # hold_drop

def _validate(model_cls, payload):
    # pydantic v2 preferred path
    if hasattr(model_cls, "model_validate_json") and isinstance(payload, str):
        return model_cls.model_validate_json(payload)
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload if isinstance(payload, dict) else json.loads(payload))

    # pydantic v1 fallback
    if isinstance(payload, str) and hasattr(model_cls, "parse_raw"):
        return model_cls.parse_raw(payload)
    if hasattr(model_cls, "parse_obj"):
        return model_cls.parse_obj(payload if isinstance(payload, dict) else json.loads(payload))
    raise RuntimeError("No suitable Pydantic validation method found.")

def validate_manager_output(task: TaskType, raw):
    try:
        if task == "trade":
            return _validate(TradeVerdict, raw)
        if task == "start_sit":
            return _validate(StartSitVerdict, raw)
        return _validate(HoldDropVerdict, raw)
    except Exception as e:
        raise ValueError(f"Manager output failed schema validation: {e}")
FINALIZE_PROMPT_SUFFIX = textwrap.dedent("""
FINALIZATION MODE (MANDATORY):
You have already gathered the necessary information and delegated to workers.
Now output ONLY the final decision JSON specified by the contract for this task.
- No thoughts.
- No actions.
- No 'Final Answer' boilerplate.
- Output must be EXACTLY one JSON object starting with '{' and ending with '}'.
""").strip()

def build_finalize_prompt(contract: str) -> str:
    return contract + "\n\n" + FINALIZE_PROMPT_SUFFIX

def is_json_object(s: str) -> bool:
    s = s.strip()
    return s.startswith("{") and s.endswith("}")



async def main():
    """
    The main function to set up and run the multi-agent system.
    """
    
    print("=" * 60)
    print("Football Manager")
    print("=" * 60)
    
    # --- Step 2: Initialize Core Components ---
    print("\n📚 Initializing fairlib.core.components...")
    llm = OpenAIAdapter(
        api_key=settings.api_keys.openai_api_key,
        model_name=settings.models["openai_gpt4"].model_name
    )

    # --- Step 3: Create Specialized Worker Agents ---
    print("👥 Building the agent team...")
    
    # The get_web_searcher_tool function automatically chooses the right implementation
    search_tool = get_web_searcher_tool(settings.search_engine.google_cse_search_api, settings.search_engine.google_cse_search_engine_id)
    
    # The Researcher: Its only tool is the WebSearcher
    researcher = create_agent(
        llm, 
        [search_tool],
        "A research agent that uses a web search tool to find current, real time information on NFL Players."
    )
    print("   ✓ Researcher agent created")

    # The Analyst: Its only tool is the SafeCalculator
    analyst = create_agent(
        llm,
        [SafeCalculatorTool()],
        "An analyst agent that performs mathematical calculations using a safe calculator to compare NFL Players."
    )
    print("   ✓ Analyst agent created")

    # We organize the workers in a dictionary so the manager can find them by name.
    workers = {"Researcher": researcher, "Analyst": analyst}

    # --- Step 4: Create the Manager Agent ---
    manager_memory = WorkingMemory()
    manager_planner = ManagerPlanner(llm, workers)
    manager_agent = SimpleAgent(llm, manager_planner, None, manager_memory)
# After creating the manager agent:

    print("   ✓ Manager agent created")

    # --- Step 5: Initialize the Hierarchical Runner ---
    team_runner = HierarchicalAgentRunner(manager_agent, workers)
    print("\n🚀 Agent team ready!\n")
    
    # --- Step 6: Define a Complex User Query ---
    print("=" * 60)
    print("📋 USER QUERY:")
    user_query = "Who should I start, Gadsen or Fannin?"
    print(f"   '{user_query}'")
    print("=" * 60)

    task_type = classify_task(user_query)
    manager_agent.role_description = build_manager_prompt(user_query)
    
    # --- Step 7: Run the Agent Team ---
    print("\n🔄 Starting multi-agent collaboration...\n")
    print("-" * 40)
    
    final_answer = await team_runner.arun(user_query)
    
    try:
        _ = validate_manager_output(task_type, final_answer)
    except Exception:
        # One-shot finalize/repair attempt
        print("[WARN] Manager output failed validation. Retrying in FINALIZATION MODE...")
        # Force a contract that ONLY allows emitting the final JSON
        manager_agent.role_description = build_finalize_prompt(build_manager_prompt(user_query))
        # Ask the same question again; manager memory should contain prior context
        final_answer = await team_runner.arun(user_query)

        # If the model still returned prose, try to salvage a JSON substring
        if isinstance(final_answer, str) and not is_json_object(final_answer):
            first = final_answer.find("{")
            last = final_answer.rfind("}")
            if first != -1 and last != -1 and last > first:
                candidate = final_answer[first:last+1]
                final_answer = candidate

        # Validate again (raise if still bad)
        _ = validate_manager_output(task_type, final_answer)


    # --- Step 8: Display the Final Result ---
    print("-" * 40)
    print("\n✅ FINAL SYNTHESIZED ANSWER:")
    print("=" * 60)
    print(final_answer)
    print("=" * 60)


if __name__ == "__main__":
    # Run the asynchronous main function.
    asyncio.run(main())