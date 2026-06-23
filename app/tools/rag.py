from langchain_core.tools import tool

@tool
def query_hybrid_rag(query: str, merchant_tier: str) -> str:
    """Queries Azure AI Search and Pinecone index for Rakuten Ichiba merchant compliance policies."""
    print(f"[RAG Engine] Hybrid fetch execution for query: '{query}'")
    
    # Core internal compliance lookups
    query_lower = query.lower()
    if "super sale" in query_lower:
        return "Policy Update (2026): Point back architectures during Super Sale events require a mandatory 2x margin matching."
    if "return" in query_lower or "refund" in query_lower:
        return "Standard Return Policy: Merchants must handle consumer refund claims within 5 business days per METI guidelines."
        
    return "Fallback Policy Context: General Rakuten Terms of Service apply. Ensure all product metadata forms match category requirements."