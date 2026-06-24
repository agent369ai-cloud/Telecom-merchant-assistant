


import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool
from app.config import settings

# ---------------------------------------------------------------------------
# Knowledge base — telecom merchant policy documents (replace with real PDFs)
# ---------------------------------------------------------------------------
_POLICY_DOCS = [
    {
        "id": "doc_001",
        "text": "Super Sale Event Policy 2026: All merchants in Super Sale must maintain a minimum 2x margin matching on point-back architectures. Gold tier merchants receive priority placement. Sales period runs for 72 hours maximum. Non-compliant merchants will be removed from the campaign.",
        "meta": {"source": "super_sale_policy.pdf", "category": "promotions"},
    },
    {
        "id": "doc_002",
        "text": "Return and Refund Policy: Merchants must handle all consumer refund claims within 5 business days per METI guidelines. Gold tier merchants must process within 3 business days. Refund disputes must be escalated to merchant support within 48 hours of customer complaint.",
        "meta": {"source": "return_policy.pdf", "category": "returns"},
    },
    {
        "id": "doc_003",
        "text": "Gold Tier Merchant Benefits: Gold tier merchants receive dedicated account manager support, priority customer service queue, reduced commission rate of 8% versus standard 12%, and early access to promotional campaigns including Super Sale and seasonal events.",
        "meta": {"source": "tier_benefits.pdf", "category": "tier_info"},
    },
    {
        "id": "doc_004",
        "text": "Inventory Management Guidelines: Merchants must maintain accurate stock levels at all times. Out-of-stock items must be updated within 24 hours. Persistent stock discrepancies over 72 hours may result in tier downgrade. RMS API updates are processed in real-time.",
        "meta": {"source": "inventory_policy.pdf", "category": "inventory"},
    },
    {
        "id": "doc_005",
        "text": "Product Listing Compliance: All product listings must include accurate category metadata, pricing in JPY, and a minimum of 3 product images. Non-compliant listings will be suspended within 48 hours of violation notice. Repeat violations result in account review.",
        "meta": {"source": "listing_compliance.pdf", "category": "compliance"},
    },
    {
        "id": "doc_006",
        "text": "Commission Structure: Standard tier pays 12% commission. Silver tier pays 10%. Gold tier pays 8%. Platinum tier pays 6%. Commission is calculated on net sale price excluding taxes and shipping. Monthly settlement is processed on the 15th of each month.",
        "meta": {"source": "commission_policy.pdf", "category": "billing"},
    },
    {
        "id": "doc_007",
        "text": "Telecom Product Guidelines: All telecom products including SIM cards, mobile plans, and accessories must comply with telecommunications regulatory requirements. Data plan listings must clearly state network speed, data caps, and contract terms in the product description.",
        "meta": {"source": "telecom_guidelines.pdf", "category": "telecom"},
    },
    {
        "id": "doc_008",
        "text": "Dispute Resolution Process: Merchant disputes must be filed within 30 days of the disputed event. Required documentation includes transaction IDs, customer communication logs, and shipping proof. Standard resolution timeline is 10 to 15 business days.",
        "meta": {"source": "dispute_resolution.pdf", "category": "disputes"},
    },
    {
        "id": "doc_009",
        "text": "Silver Tier Requirements: Merchants qualify for Silver tier with minimum monthly sales of 500,000 JPY and a customer satisfaction score above 4.2. Silver tier merchants receive a dedicated support email line and 10% commission rate.",
        "meta": {"source": "tier_requirements.pdf", "category": "tier_info"},
    },
    {
        "id": "doc_010",
        "text": "Shipping and Delivery Standards: All orders must be shipped within 2 business days of confirmation. Gold and Platinum tier merchants must ship within 1 business day. Delayed shipments must be communicated to customers proactively via the RMS messaging system.",
        "meta": {"source": "shipping_policy.pdf", "category": "shipping"},
    },
]

# ---------------------------------------------------------------------------
# ChromaDB — in-memory vector store with ONNX-based embeddings (no API key)
# ---------------------------------------------------------------------------
_chroma_client = chromadb.Client()
_embed_fn = embedding_functions.DefaultEmbeddingFunction()  # all-MiniLM-L6-v2 via ONNX

_collection = _chroma_client.create_collection(
    name="telecom_merchant_policies",
    embedding_function=_embed_fn,
    metadata={"hnsw:space": "cosine"},
)

_collection.add(
    ids=[d["id"] for d in _POLICY_DOCS],
    documents=[d["text"] for d in _POLICY_DOCS],
    metadatas=[d["meta"] for d in _POLICY_DOCS],
)


# ---------------------------------------------------------------------------
# Cohere reranker — improves result quality; falls back gracefully if no key
# ---------------------------------------------------------------------------
def _rerank(query: str, documents: list[str], top_n: int = 3) -> list[int]:
    if not settings.COHERE_API_KEY:
        return list(range(min(top_n, len(documents))))
    try:
        import cohere
        co = cohere.Client(settings.COHERE_API_KEY)
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=top_n,
        )
        return [r.index for r in response.results]
    except Exception as e:
        print(f"[Reranker] Cohere rerank failed, using vector results: {e}")
        return list(range(min(top_n, len(documents))))


# ---------------------------------------------------------------------------
# RAG tool exposed to the LangGraph agent
# ---------------------------------------------------------------------------
@tool
def query_hybrid_rag(query: str, merchant_tier: str) -> str:
    """Queries the telecom merchant policy knowledge base using semantic search and reranking.
    Use this for questions about policies, returns, promotions, compliance, tier benefits, commissions."""

    # Step 1 — Vector search: retrieve top 6 semantically similar chunks
    augmented_query = f"{query} (merchant tier: {merchant_tier})" if merchant_tier else query
    results = _collection.query(query_texts=[augmented_query], n_results=6)
    candidate_docs = results["documents"][0]
    candidate_meta = results["metadatas"][0]

    if not candidate_docs:
        return "No relevant policy found for your query."

    # Step 2 — Rerank: pick the best 3 from the 6 candidates
    reranked_indices = _rerank(query, candidate_docs, top_n=3)

    # Step 3 — Build context string for the LLM
    context_blocks = [
        f"[Source: {candidate_meta[i]['source']}]\n{candidate_docs[i]}"
        for i in reranked_indices
    ]
    return "Relevant Policy Information:\n\n" + "\n\n---\n\n".join(context_blocks)
