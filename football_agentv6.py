import asyncio
import os
import textwrap

from dotenv import load_dotenv
load_dotenv()

"""
Documentation:

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

version 5 added in proper manager workflow and chatbot functionality:
https://chatgpt.com/share/69169c61-2130-8006-bad6-c13cb822ed4e

Version 6 attempted to fix json issue and improved calculator tool use:
https://chatgpt.com/share/69169ca2-0ba4-8006-938b-6052b89ae930

Finally, this thread evaluates the README summary and clarifies any last
minute questions on the code while commenting. Also generated a png of code structure:


"""

# fairlib core
# import packages
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

# =================== CONFIG: ALWAYS PROSE ===================
# Final output structure
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

# explicit fantasy scoring spec so the Analyst stops inventing math
SCORING_SPEC = textwrap.dedent("""
Use this approximate half-PPR fantasy scoring model:

- RB/WR/TE:
  fantasy_points = (rush_yds / 10) + (rec_yds / 10) + (receptions * 0.5) + (total_TDs * 6)

- QB:
  fantasy_points = (pass_yds / 25) + (rush_yds / 10) + (total_TDs * 4) - (interceptions * 2)

- Then:
  PPG = fantasy_points / games_played.

If some components (like receptions or interceptions) are unknown, compute fantasy_points
using only the known components and clearly note that the PPG is approximate.

You MUST use the safe calculator tool for all arithmetic (fantasy_points, PPG, and
replacement deltas). Do NOT do math in your head.
""").strip()

# =================== ENV & API KEYS ===================
settings.api_keys.openai_api_key = os.getenv("OPENAI_API_KEY")
settings.api_keys.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
settings.search_engine.google_cse_search_api = os.getenv("GOOGLE_CSE_SEARCH_API")
settings.search_engine.google_cse_search_engine_id = os.getenv("GOOGLE_CSE_SEARCH_ENGINE_ID")

# =================== TASK CLASSIFICATION ===================
from typing import Literal

TaskType = Literal["trade", "start_sit", "hold_drop"]

# manager helper function
def classify_task(user_query: str) -> TaskType:
    """
    Simple heuristic classifier for the fantasy question type.
    Acts as a deterministic 'manager helper' for routing.
    """
    q = user_query.lower()
    if ("trade" in q) or ((" for " in q) and not any(x in q for x in ["start", "sit", "bench"])):
        return "trade"
    if ("start" in q) or ("sit" in q):
        return "start_sit"
    return "hold_drop"

# =================== AGENT FACTORY ===================
# Creates worker agents given their role and tools they will need
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

# Builds web searcher agent specifically
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

# Researcher prompts
RESEARCHER_PROMPT_BASE = textwrap.dedent("""
You are a fantasy football researcher. Use your web_searcher tool when needed.

USER QUESTION:
{user_query}

TASK:
1) Identify the relevant NFL player(s) for this question.
2) For each player, gather concise, fantasy-relevant stats:
   - season PPG (half-PPR if available),
   - last-3 PPG,
   - role/snap share if available (or 'unknown'),
   - injury status (healthy/questionable/out/unknown),
   - turnovers (for QBs) or rushing TDs (for QBs/RBs) if available.
3) Provide a one-sentence rest-of-season outlook per player.
4) Give 2–3 short source attributions (names only: ESPN, FantasyData, PFF, CBS, etc.).
5) Output short, crisp prose. No tables. No JSON. Keep it under ~12 lines total.

If any stat is unavailable, state 'unknown'.
""").strip()

# AUDITOR PROMPT
AUDITOR_PROMPT_BASE = textwrap.dedent("""
You are a fantasy football research auditor.

USER QUESTION:
{user_query}

RESEARCH SUMMARY (from the researcher agent):
{research_block}

YOUR TASK:
- Check the research for:
  - Obvious inconsistencies in stats (e.g., conflicting PPG numbers for the same player).
  - Missing or vague source attributions.
  - Overly confident claims when stats are 'unknown' or clearly uncertain.
- In 3–6 short sentences:
  - Comment on the overall reliability of the research (e.g., HIGH, MEDIUM, or LOW CONFIDENCE).
  - Point out any specific issues or uncertainties.
  - If everything looks reasonable, say so explicitly.

Do NOT redo the web search. Do NOT introduce new stats. Just audit what you see.
Do NOT output JSON; respond in plain prose only.
""").strip()

# Analyst Prompt
ANALYST_PROMPT_BASE = textwrap.dedent("""
You are a fantasy football analyst. You can use the safe calculator for arithmetic if needed.

CONTEXT (from research):
{research_block}

AUDITOR NOTES (on research quality):
{audit_block}

USER QUESTION:
{user_query}

SCORING MODEL (for your calculations):
{scoring_spec}

TASK:
- Use the scoring model above for fantasy-point and PPG calculations.
- You MUST use the safe calculator tool for all non-trivial arithmetic.
- Extract per-player season PPG and last-3 PPG (or 'unknown'). If the research block only
  gives raw stats (yards, TDs, games), compute approximate PPG using the scoring model.
- For TRADE: compute season-PPG delta (A - B) and last-3 delta (A - B) if both known;
  otherwise mark the relevant delta as 'unknown'.
- For START/SIT (single or A vs B): determine trend per player as up/flat/down based on
  last-3 vs season (tolerance ±0.5).
- For HOLD/DROP: estimate replacement baseline (QB=15.0, RB=9.5, WR=10.0, TE=7.5)
  and compute delta = season_ppg - baseline if known; label trend up/flat/down
  using ±0.5 tolerance.

OUTPUT REQUIREMENTS:
- Short prose summary of the key computed deltas and trends.
- Mention if the auditor labeled the research as LOW CONFIDENCE.
- No JSON. Do NOT output dictionaries or key/value code blocks.
- Keep it under ~8 lines.
""").strip()

# =================== PROMPTS (MANAGER & FINAL SYNTHESIS) ===================

# MANAGER PROMPT
MANAGER_PROMPT_BASE = textwrap.dedent("""
You are the manager agent for a fantasy football assistant.

You have the following worker agents:
- Researcher: can use web tools to gather current stats, roles, injuries, and short ROS outlooks.
- Auditor: checks the research for consistency, missing sources, and flags LOW/MEDIUM/HIGH CONFIDENCE.
- Analyst: uses the research (and audit notes) to compute deltas, trends, and replacement-level comparisons.
- Synthesizer: produces the final decision in a strict prose format for the user.

USER QUESTION:
{user_query}

CLASSIFIED TASK TYPE:
{task_type}

TASK:
- In 3–5 short bullet points, describe:
  - Which workers should be called,
  - In what order,
  - And what each should focus on for this specific question.
- Do NOT call tools. Do NOT output JSON. Just give the plan as plain text bullets.
""").strip()

# Outpute for trade decisions
FINAL_SYNTHESIS_TRADE = textwrap.dedent("""
You are the team manager. Produce the final decision in the required prose format.

USER QUESTION:
{user_query}

MANAGER PLAN:
{manager_plan}

RESEARCH SUMMARY:
{research_block}

AUDITOR SUMMARY:
{audit_block}

ANALYST SUMMARY:
{analyst_block}

DECISION RULES (apply sensibly):
- Choose: FAIR | FAVORABLE | UNFAVORABLE (for the user trading away Player A for Player B).
- Consider PPG, last-3 trend, turnovers, injuries/OL notes, near-term schedule, ROS outlook.
- Take into account if the auditor flagged LOW CONFIDENCE and soften the verdict if needed.
- Tie-breaker: if season PPG delta within ±0.5, break ties using trend + schedule; if still tied → FAIR.

Now write ONLY the final answer using this format:
{prose_spec}
""").strip()

# Output for start sit decisions
FINAL_SYNTHESIS_STARTSIT = textwrap.dedent("""
You are the team manager. Produce the final decision in the required prose format.

USER QUESTION:
{user_query}

MANAGER PLAN:
{manager_plan}

RESEARCH SUMMARY:
{research_block}

AUDITOR SUMMARY:
{audit_block}

ANALYST SUMMARY:
{analyst_block}

DECISION RULES:
- Output START if season PPG ≥ 12 or last-3 ≥ season and trend up; penalize material injury/snap loss or tough schedule next 2–3.
- Otherwise SIT.
- If auditor labeled the research as LOW CONFIDENCE, be more conservative and mention uncertainty in the prose.

Now write ONLY the final answer using this format:
{prose_spec}
""").strip()

# Output for hold-drop decisions
FINAL_SYNTHESIS_HOLDDROP = textwrap.dedent("""
You are the team manager. Produce the final decision in the required prose format.

USER QUESTION:
{user_query}

MANAGER PLAN:
{manager_plan}

RESEARCH SUMMARY:
{research_block}

AUDITOR SUMMARY:
{audit_block}

ANALYST SUMMARY:
{analyst_block}

DECISION RULES:
- HOLD if replacement_delta ≥ +2.0 or trend up with last-3 ≥ season.
- DROP if replacement_delta ≤ 0 and trend down or multiweek OUT.
- Otherwise HOLD.
- If auditor labeled the research as LOW CONFIDENCE, lean toward HOLD unless the player is clearly unstartable.

Now write ONLY the final answer using this format:
{prose_spec}
""").strip()

# =================== MAIN ORCHESTRATION (MANAGER + WORKERS ===================
# Runs pipeline
async def run_pipeline(user_query: str, return_debug: bool = False):
    """
    Top-level 'manager' pipeline that:
      1) Classifies the task.
      2) Uses a Manager agent to plan worker usage.
      3) Calls Researcher → Auditor → Analyst → Synthesizer.

    If return_debug=False (default): returns the final prose verdict (str).
    If return_debug=True: returns a dict with intermediate artifacts:
        {
            "task_type": ...,
            "manager_plan": ...,
            "research_block": ...,
            "audit_block": ...,
            "analyst_block": ...,
            "final_answer": ...
        }
    """
    # Core LLM for synthesis and to drive workers' planners
    llm = OpenAIAdapter(
        api_key=settings.api_keys.openai_api_key,
        model_name=settings.models["openai_gpt4"].model_name
    )

    # === Build worker agents ===
    researcher = create_worker_agent(
        llm,
        [get_web_searcher_tool(settings.search_engine.google_cse_search_api,
                               settings.search_engine.google_cse_search_engine_id)],
        "A research agent that uses a web search tool to find current, real-time information on NFL players."
    )

    analyst = create_worker_agent(
        llm,
        [SafeCalculatorTool()],
        "An analyst agent that performs mathematical calculations and trend checks for NFL player comparisons."
    )

    auditor = create_worker_agent(
        llm,
        [],
        "A verification agent that audits research for consistency, missing sources, and overall confidence level."
    )

    synthesizer = create_worker_agent(
        llm,
        [],  # no tools
        "You are the final-decider. Produce ONLY the final prose answer in the required format. Never output JSON."
    )

    manager = create_worker_agent(
        llm,
        [],  # no tools
        "You are the manager agent. You plan how to use the worker agents given the user's question."
    )

    # Step 1: classify task (deterministic helper)
    task_type = classify_task(user_query)

    # Step 2: Manager plans the workflow
    manager_prompt = MANAGER_PROMPT_BASE.format(
        user_query=user_query,
        task_type=task_type
    )
    manager_plan = await manager.arun(manager_prompt)

    # Step 3: Research
    research_prompt = RESEARCHER_PROMPT_BASE.format(user_query=user_query)
    research_block = await researcher.arun(research_prompt)

    # Step 4: Audit the research
    auditor_prompt = AUDITOR_PROMPT_BASE.format(
        user_query=user_query,
        research_block=research_block
    )
    audit_block = await auditor.arun(auditor_prompt)

    # Step 5: Analyst
    analyst_prompt = ANALYST_PROMPT_BASE.format(
        user_query=user_query,
        research_block=research_block,
        audit_block=audit_block,
        scoring_spec=SCORING_SPEC, # Scoring rules defined earlier to not make up math
    )
    analyst_block = await analyst.arun(analyst_prompt)

    # Step 6: Final synthesis in prose only
    # Output for trade decisions
    if task_type == "trade":
        synth_prompt = FINAL_SYNTHESIS_TRADE.format(
            user_query=user_query,
            manager_plan=manager_plan,
            research_block=research_block,
            audit_block=audit_block,
            analyst_block=analyst_block,
            prose_spec=PROSE_FINAL_SPEC
        )
    # Output for start_sit decisions
    elif task_type == "start_sit":
        synth_prompt = FINAL_SYNTHESIS_STARTSIT.format(
            user_query=user_query,
            manager_plan=manager_plan,
            research_block=research_block,
            audit_block=audit_block,
            analyst_block=analyst_block,
            prose_spec=PROSE_FINAL_SPEC
        )
    # Output for hold_drop decisions
    else:
        synth_prompt = FINAL_SYNTHESIS_HOLDDROP.format(
            user_query=user_query,
            manager_plan=manager_plan,
            research_block=research_block,
            audit_block=audit_block,
            analyst_block=analyst_block,
            prose_spec=PROSE_FINAL_SPEC
        )

    final_answer = await synthesizer.arun(synth_prompt)
    final_answer = str(final_answer).strip()

    # Returns debug output when needed. Will keep on by default to see thoughts.
    if return_debug:
        return {
            "task_type": task_type,
            "manager_plan": str(manager_plan).strip(),
            "research_block": str(research_block).strip(),
            "audit_block": str(audit_block).strip(),
            "analyst_block": str(analyst_block).strip(),
            "final_answer": final_answer,
        }
    else:
        return final_answer

# =================== CLI ENTRY: INTERACTIVE LOOP ===================
async def main():
    # User Interface design
    print("=" * 70)
    print("Football Manager (Interactive, showing manager plan + worker outputs)")
    print("Type 'quit' or 'exit' to end.")
    print("=" * 70)

    while True:
        user_query = input("\nYou: ").strip()
        # To exit
        if user_query.lower() in {"quit", "exit"}:
            print("Goodbye- may your waiver claims always clear.")
            break
        # Stay open while no query entered
        if not user_query:
            continue

        # Run full pipeline with debug info so we can show the 'thoughts/actions'
        result = await run_pipeline(user_query, return_debug=True)

        print("\n" + "-" * 70)
        print(f"TASK TYPE: {result['task_type']}")
        print("-" * 70)

        print("\n[MANAGER PLAN]")
        print(result["manager_plan"])

        print("\n[RESEARCHER OUTPUT]")
        print(result["research_block"])

        print("\n[AUDITOR OUTPUT]")
        print(result["audit_block"])

        print("\n[ANALYST OUTPUT]")
        print(result["analyst_block"])

        print("\n[FINAL ANSWER]")
        print(result["final_answer"])
        print("-" * 70)

# Runs the whole program
if __name__ == "__main__":
    asyncio.run(main())
