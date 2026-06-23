from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.agents.graph import agent_runtime

api = FastAPI(title="Rakuten Merchant Support Engine API", version="1.0.0")

class ChatPayload(BaseModel):
    merchant_id: str
    tier: str
    message: str

@api.get("/health")
async def health_check():
    return {"status": "healthy", "service": "rakuten-agent-core"}

@api.post("/v1/agent/chat")
async def process_merchant_intent(payload: ChatPayload):
    try:
        # Build clean running configuration state context
        runtime_context = {
            "messages": [HumanMessage(content=payload.message)],
            "merchant_id": payload.merchant_id,
            "current_tier": payload.tier,
            "next_node": "router",
            "is_compliant": True
        }
        
        # Invoke Graph execution flow
        result_state = agent_runtime.invoke(runtime_context)
        final_system_message = result_state["messages"][-1].content
        
        return {
            "merchant_id": payload.merchant_id,
            "agent_response": final_system_message
        }
        
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Runtime Exception: {str(ex)}")