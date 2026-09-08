import json
from pathlib import Path

from langchain_core.tools import tool

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

MODEL = "gemini-3.1-flash-lite"

def _read(file_name: str):
    with open(DATA_DIR / file_name, encoding="utf-8") as f:
        return json.load(f)

_CUSTOMERS = _read("customers.json")
_EQUIPMENT = _read("equipment.json")
_PARTS = _read("parts.json")
_INVENTORY = _read("inventory.json")
_PRICING = _read("pricing.json")


@tool
def get_customer(email: str) -> dict:
    """
    Look up a customer by their email address (the sender of the received message) and return
    customer_id, name, and price level. If the address does not match any known customer,
    return a dict with the key "error".
    """
    normalized_email = email.strip().lower()
    for customer in _CUSTOMERS:
        if customer["email"].lower() == normalized_email:
            return customer
    return {"error": f"There is no customer recorded with the email address '{email}'."}


@tool
def find_equipment(model: str | None = None, serial_number: str | None = None) -> list[dict]:
    """
    Search for equipment by model and/or serial number. At least one of the two parameters
    must be provided. If only the model is given and there are multiple pieces of equipment
    with that model (different serial numbers), return all candidate results -- the caller
    (agent or operator) decides how to resolve the ambiguity, not the tool. If the serial number is also given, the search is exact.
    """
    equipments = []
    for equipment in _EQUIPMENT:
        if model is not None and equipment["model"].lower() != model.strip().lower():
            continue
        if serial_number is not None and equipment["serial_number"].lower() != serial_number.strip().lower():
            continue
        equipments.append(equipment)
    return equipments


@tool
def search_parts(query: str, equipment_id: str | None = None) -> list[dict]:
    """
    Search for parts by simple keywords (in the part name and in its aliases -- the informal terms
    that clients use to describe the part, e.g. "oil filter"). The search is literal on words, 
    it does NOT understand synonyms outside of the aliases already listed. If equipment_id is given, 
    the results are filtered to parts compatible with that equipment's model. 
    It may return 0, 1, or multiple candidate parts -- an empty list means the part does not exist in the catalog for that query/model, 
    multiple results mean real ambiguity (e.g. two types of oil filter), not a search error.
    """
    equipment_model = None
    if equipment_id is not None:
        equipment = next((e for e in _EQUIPMENT if e["equipment_id"] == equipment_id), None)
        if equipment is None:
            return []
        equipment_model = equipment["model"]

    tokens = query.lower().split()
    parts = []
    for part in _PARTS:
        if equipment_model is not None and equipment_model not in part["compatible_models"]:
            continue
        field_to_search = " ".join([part["name"], *part["alias"]]).lower()
        if tokens and not all(token in field_to_search for token in tokens):
            continue
        parts.append(part)
    return parts


@tool
def check_stock(part_number: str, quantity: int) -> dict:
    """
    Check if the available stock for a part number covers the requested quantity.
    Returns available (bool), quantity_available, and quantity_requested. If the part_number
    does not exist, returns a dict with the key "error".
    """
    if part_number not in _INVENTORY:
        return {"error": f"Part number '{part_number}' doesn't exist."}
    quantity_available = _INVENTORY[part_number]
    return {
        "part_number": part_number,
        "quantity_available": quantity_available,
        "quantity_requested": quantity,
        "available": quantity_available >= quantity,
    }


@tool
def get_price(part_number: str, customer_id: str, quantity: int) -> dict:
    """
    Calculate the total price for a quantity of a part, for a specific customer (the customer's price level determines the discount applied). 
    Prices come EXCLUSIVELY from pricing.json -- this tool is the only source of truth for price, 
    no other node or agent is allowed to calculate or assume a price. If the part_number or customer_id does not exist, 
    return a dict with the key "error".
    """
    list_price = _PRICING["list_prices"].get(part_number)
    if list_price is None:
        return {"error": f"Part number '{part_number}' doesn't have listed price."}
    customer = next((c for c in _CUSTOMERS if c["customer_id"] == customer_id), None)
    if customer is None:
        return {"error": f"Customer '{customer_id}' doesn't exist."}
    discount = _PRICING["discount_price_level"].get(customer["price_level"], 0.0)
    total_price = list_price * quantity * (1 - discount)
    return {
        "part_number": part_number,
        "unit_price": list_price,
        "discount": discount,
        "quantity": quantity,
        "total_price": round(total_price, 2),
    }


