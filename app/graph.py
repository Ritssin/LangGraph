import os
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class GraphState(TypedDict):
    prompt: str
    route: str
    response: str


class RouteDecision(BaseModel):
    target: Literal["calculation", "news", "other"] = Field(
        description=(
            "calculation: math, arithmetic, compute, solve equations, percentages, numbers; "
            "news: latest news, headlines, current events, breaking news, what happened today; "
            "other: everything else"
        )
    )


def _llm(model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )


def router_node(state: GraphState) -> dict[str, str]:
    llm = _llm().with_structured_output(RouteDecision)
    text = (
        "Classify the user message into exactly one category.\n\n"
        "- calculation: anything primarily about math or numeric computation.\n"
        "- news: requests for recent or current news, headlines, or world events.\n"
        "- other: all other requests.\n\n"
        f"User message:\n{state['prompt']}"
    )
    decision = llm.invoke(text)
    return {"route": decision.target}


def route_to_agent(state: GraphState) -> Literal["agent1", "agent2", "agent3"]:
    return {"calculation": "agent1", "news": "agent2", "other": "agent3"}[state["route"]]


def agent1_calculation(state: GraphState) -> dict[str, str]:
    llm = _llm()
    msgs = [
        SystemMessage(
            content=(
                "You are a precise calculator assistant. Solve the user's math request. "
                "Show brief working if helpful, then give the final answer clearly."
            )
        ),
        HumanMessage(content=state["prompt"]),
    ]
    out = llm.invoke(msgs)
    return {"response": out.content}


def agent2_news(state: GraphState) -> dict[str, str]:
    search = DuckDuckGoSearchRun()
    try:
        raw = search.invoke(state["prompt"][:300])
    except Exception as exc:  # noqa: BLE001
        raw = f"(Search unavailable: {exc})"
    if not raw or not str(raw).strip():
        raw = "No search results returned."

    llm = _llm()
    msgs = [
        SystemMessage(
            content=(
                "You summarize web search snippets for the user. "
                "Be factual, cite uncertainty if results are thin, and keep it concise."
            )
        ),
        HumanMessage(
            content=f"User request:\n{state['prompt']}\n\nSearch results:\n{raw}"
        ),
    ]
    out = llm.invoke(msgs)
    return {"response": out.content}


def agent3_joke(state: GraphState) -> dict[str, str]:
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.9,
    )
    msgs = [
        SystemMessage(
            content=(
                "You respond with a short, friendly joke. "
                "The joke can loosely relate to the user's message, or be a general one-liner if it does not fit."
            )
        ),
        HumanMessage(content=state["prompt"]),
    ]
    out = llm.invoke(msgs)
    return {"response": out.content}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("router", router_node)
    g.add_node("agent1", agent1_calculation)
    g.add_node("agent2", agent2_news)
    g.add_node("agent3", agent3_joke)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_to_agent)
    g.add_edge("agent1", END)
    g.add_edge("agent2", END)
    g.add_edge("agent3", END)
    return g.compile()


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
