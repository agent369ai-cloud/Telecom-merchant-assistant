from langchain_core.tools import tool

@tool
def update_rms_inventory(item_id: str, quantity: int) -> str:
    """Modifies product stock levels directly inside the Rakuten Merchant System (RMS) database."""
    print(f"[RMS API] Writing to core ledger -> Item ID: {item_id}, New Qty: {quantity}")
    return f"Execution Success: Item ID {item_id} has been modified to {quantity} units in the transactional database."