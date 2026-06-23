rakuten-merchant-assistant/
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerization for AKS
├── README.md                   # Setup and execution guide
│
└── app/
    ├── __init__.py
    ├── main.py                 # FastAPI microservice gateway
    ├── config.py               # Configuration & environment variables
    │
    ├── agents/
    │   ├── __init__.py
    │   ├── state.py            # LangGraph state definition
    │   └── graph.py            # LangGraph agent orchestration & routing
    │
    └── tools/
        ├── __init__.py
        ├── rag.py              # Mocked Hybrid RAG (Azure + Pinecone) interface
        └── rms_api.py          # Mocked Rakuten Merchant System API tool calls


#pip install pydantic-settings


# Clone or navigate into your repository base folder
cd rakuten-merchant-assistant

# Create the virtual isolation folder
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Clean install packages
pip install -r requirements.txt

export OPENAI_API_KEY="your-real-openai-api-key-here"

python -m uvicorn app.main:api --reload --port 8000



curl -X POST "http://localhost:8000/v1/agent/chat" \
     -H "Content-Type: application/json" \
     -d '{"merchant_id": "shop-9921", "tier": "Gold", "message": "What are the rules for the upcoming super sale?"}'