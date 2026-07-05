import json
import re

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

from app.agent.state import AgentState
from app.agent.tools import AGENT_TOOLS
from app.core.config import settings

SYSTEM_PROMPT = (
    "You are an expert clinical trial document classifier. You must classify the document "
    "into exactly one of the following categories: Protocol, SAP, ICF, CSR, IB, Combined. "
    "You have been provided with evidence gathered by a retrieval pipeline. This evidence "
    "may be strong or limited depending on how many pages were readable. You must always "
    "return a final classification — do not refuse, do not ask for more information, and do "
    "not say the evidence is insufficient. If confidence is low, make your best judgment from "
    "the available evidence and reflect that uncertainty in the final_confidence score. The "
    "deep reader fallback has already been used if applicable — no further retrieval is "
    "possible. Base your decision entirely on what is in front of you. "
    "Always respond with valid JSON only."
)

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        model = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=settings.anthropic_api_key,
            temperature=0,
        )
        _agent = create_agent(model=model, tools=AGENT_TOOLS, system_prompt=SYSTEM_PROMPT)
    return _agent


def _build_user_prompt(state: AgentState) -> str:
    top_chunks = (state.get("retrieved_chunks") or [])[:3]
    top_chunks_text = "\n".join(
        f"- [{c.get('category', 'unknown')}] {c.get('content', '')[:300]}" for c in top_chunks
    ) or "(none retrieved)"

    if state.get("fallback_triggered"):
        fallback_note = f"Note: Deep reader fallback was triggered. Document was read up to {settings.max_pages_fallback} pages."
    else:
        fallback_note = f"Note: Classification based on first {settings.max_pages_initial} pages only."

    return f"""## Voting Results from Similar Documents
{state.get('vote_counts', {})}
Confidence: {state.get('confidence', 0.0):.2%}
Preliminary category: {state.get('predicted_category', '')}
{fallback_note}

## Document Content (first pages)
{state.get('extracted_text', '')[:4000]}

## Top Retrieved Similar Chunks
{top_chunks_text}

## Task
Based on all evidence, provide final classification.

Respond ONLY with this JSON:
{{
  "final_category": "<category>",
  "final_confidence": <0.0-1.0>,
  "reasoning": "<2-3 sentence explanation>",
  "evidence_chunks": ["<key phrase 1>", "<key phrase 2>"]
}}"""


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text!r}")
    return json.loads(match.group(0))


def finalizer_node(state: AgentState) -> dict:
    try:
        user_prompt = _build_user_prompt(state)
        agent = _get_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        final_message = result["messages"][-1].content
        parsed = _extract_json(final_message)

        return {
            "final_category": parsed["final_category"],
            "final_confidence": float(parsed["final_confidence"]),
            "reasoning": parsed["reasoning"],
            "evidence_chunks": parsed.get("evidence_chunks", []),
        }
    except Exception as exc:
        return {
            "error": f"finalizer_node failed: {exc}",
            "final_category": state.get("predicted_category", ""),
            "final_confidence": state.get("confidence", 0.0),
            "reasoning": "LLM finalization failed; falling back to preliminary classification.",
            "evidence_chunks": [],
        }
