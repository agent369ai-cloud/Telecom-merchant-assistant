from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.state import MerchantAgentState
from app.tools.rag import query_hybrid_rag
from app.tools.rms_api import update_rms_inventory
from app.config import settings

# Initialize Model Engine
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=settings.OPENAI_API_KEY)
llm_with_tools = llm.bind_tools([query_hybrid_rag, update_rms_inventory])

def router_planner_node(state: MerchantAgentState):
    """Evaluates business rules, runs guardrails, and schedules work nodes."""
    messages = state["messages"]
    last_user_message = messages[-1].content
    
    # 🛑 Structural Input Guardrail
    if "override_system_admin" in last_user_message:
        return {
            "messages": [AIMessage(content="Security Alert: Injection attempt intercepted.")],
            "next_node": "end"
        }
        
    response = llm_with_tools.invoke(messages)
    
    # Conditional logic determination
    if response.tool_calls:
        return {
            "messages": [response],
            "next_node": response.tool_calls[0]["name"]
        }
    
    return {"messages": [response], "next_node": "end"}

def rag_node(state: MerchantAgentState):
    """Executes RAG retrieval and feeds background contexts back to the agent state."""
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    
    result = query_hybrid_rag.invoke(tool_call["args"])
    return {"messages": [AIMessage(content=f"[Verified Resource]: {result}")]}

def rms_node(state: MerchantAgentState):
    """Executes transaction systems configuration writes."""
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    
    result = update_rms_inventory.invoke(tool_call["args"])
    return {"messages": [AIMessage(content=f"[System Confirmation]: {result}")]}

# --- Build State Pipeline ---
workflow = StateGraph(MerchantAgentState)

workflow.add_node("router", router_planner_node)
workflow.add_node("query_hybrid_rag", rag_node)
workflow.add_node("update_rms_inventory", rms_node)

workflow.set_entry_point("router")

def execution_router_edge(state: MerchantAgentState):
    dest = state.get("next_node")
    if dest in ["query_hybrid_rag", "update_rms_inventory"]:
        return dest
    return END

workflow.add_conditional_edges(
    "router",
    execution_router_edge,
    {
        "query_hybrid_rag": "query_hybrid_rag",
        "update_rms_inventory": "update_rms_inventory",
        END: END
    }
)

workflow.add_edge("query_hybrid_rag", END)
workflow.add_edge("update_rms_inventory", END)

# Compiled Executable Graph Object
agent_runtime = workflow.compile()