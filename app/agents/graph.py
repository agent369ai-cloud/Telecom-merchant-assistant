from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.state import MerchantAgentState
from app.tools.rag import query_hybrid_rag
from app.tools.rms_api import update_rms_inventory
from app.tools.guardrails import check_guardrails
from app.config import settings

# ---------------------------------------------------------------------------
# LLM — shared across nodes
# ---------------------------------------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=settings.GROQ_API_KEY,
)
llm_with_tools = llm.bind_tools([query_hybrid_rag, update_rms_inventory])


# ---------------------------------------------------------------------------
# Node 1 — Guardrail: runs BEFORE the LLM, blocks unsafe messages
# ---------------------------------------------------------------------------
def guardrail_node(state: MerchantAgentState):
    last_message = state["messages"][-1].content
    is_safe, reason = check_guardrails(last_message)

    if not is_safe:
        return {
            "messages": [AIMessage(content=f"Security Alert: Your request was blocked. Reason: {reason}")],
            "next_node": "end",
        }

    return {"next_node": "router"}


# ---------------------------------------------------------------------------
# Node 2 — Router: LLM decides to call a tool or answer directly
# ---------------------------------------------------------------------------
def router_planner_node(state: MerchantAgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        return {
            "messages": [response],
            "next_node": response.tool_calls[0]["name"],
        }

    return {"messages": [response], "next_node": "end"}


# ---------------------------------------------------------------------------
# Node 3 — RAG: retrieves policy chunks and stores context in state
# ---------------------------------------------------------------------------
def rag_node(state: MerchantAgentState):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    context = query_hybrid_rag.invoke(tool_call["args"])
    # Store raw context separately so validate_node can access it later
    return {
        "rag_context": context,
        "next_node": "synthesize",
    }


# ---------------------------------------------------------------------------
# Node 4 — Synthesize: LLM generates a grounded answer from retrieved context
# ---------------------------------------------------------------------------
def synthesize_node(state: MerchantAgentState):
    # Recover original question from the first HumanMessage
    original_question = next(
        m.content for m in state["messages"] if isinstance(m, HumanMessage)
    )
    tier = state.get("current_tier", "Standard")
    context = state.get("rag_context", "")

    prompt = (
        f"You are a Telecom Merchant Support assistant. "
        f"The merchant's tier is: {tier}.\n\n"
        f"Answer the question using ONLY the context below. "
        f"If the context does not contain enough information, say so clearly — do not invent details.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {original_question}\n\n"
        f"Answer:"
    )

    response = llm.invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}


# ---------------------------------------------------------------------------
# Node 5 — Validate: LLM-as-judge faithfulness check
# Verifies the synthesized answer is grounded in the retrieved context.
# ---------------------------------------------------------------------------
def validate_node(state: MerchantAgentState):
    context = state.get("rag_context", "")
    answer = state["messages"][-1].content

    prompt = (
        "You are a strict fact-checker. Given a context and an answer, "
        "verify that every factual claim in the answer is supported by the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Reply in exactly this format (two lines, nothing else):\n"
        "FAITHFUL: yes\n"
        "ISSUES: none\n\n"
        "Or if there are unsupported claims:\n"
        "FAITHFUL: no\n"
        "ISSUES: <describe each unsupported claim>\n\n"
        "Rules: if ISSUES is 'none', FAITHFUL must be 'yes'. Never write FAITHFUL: no with ISSUES: none."
    )

    result = llm.invoke(prompt).content.strip()

    issues_line = next(
        (line for line in result.splitlines() if line.upper().startswith("ISSUES:")),
        "ISSUES: none",
    )
    issues = issues_line.split(":", 1)[-1].strip()

    # Primary signal: if no issues were found, the answer is faithful.
    # Secondary signal: explicit FAITHFUL: yes as a fallback.
    faithful = issues.lower() == "none" or "FAITHFUL: yes" in result

    if faithful:
        return {"faithfulness_ok": True}

    # Append a transparency disclaimer so the user sees the caveat
    disclaimer = f"\n\n⚠️ Validation Notice: This answer may contain unverified claims. Issues detected: {issues}"
    patched_answer = answer + disclaimer
    return {
        "messages": [AIMessage(content=patched_answer)],
        "faithfulness_ok": False,
    }


# ---------------------------------------------------------------------------
# Node 6 — RMS: inventory / merchant system write operations
# ---------------------------------------------------------------------------
def rms_node(state: MerchantAgentState):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    result = update_rms_inventory.invoke(tool_call["args"])
    return {"messages": [AIMessage(content=f"[System Confirmation]: {result}")]}


# ---------------------------------------------------------------------------
# Build the StateGraph pipeline
# ---------------------------------------------------------------------------
workflow = StateGraph(MerchantAgentState)

workflow.add_node("guardrail", guardrail_node)
workflow.add_node("router", router_planner_node)
workflow.add_node("query_hybrid_rag", rag_node)
workflow.add_node("synthesize", synthesize_node)
workflow.add_node("validate", validate_node)
workflow.add_node("update_rms_inventory", rms_node)

workflow.set_entry_point("guardrail")


def after_guardrail(state: MerchantAgentState):
    return state.get("next_node", "end")

workflow.add_conditional_edges(
    "guardrail",
    after_guardrail,
    {"router": "router", "end": END},
)


def after_router(state: MerchantAgentState):
    dest = state.get("next_node")
    if dest in ["query_hybrid_rag", "update_rms_inventory"]:
        return dest
    return END

workflow.add_conditional_edges(
    "router",
    after_router,
    {
        "query_hybrid_rag": "query_hybrid_rag",
        "update_rms_inventory": "update_rms_inventory",
        END: END,
    },
)

# RAG → synthesize → validate → END
workflow.add_edge("query_hybrid_rag", "synthesize")
workflow.add_edge("synthesize", "validate")
workflow.add_edge("validate", END)

workflow.add_edge("update_rms_inventory", END)

agent_runtime = workflow.compile()
