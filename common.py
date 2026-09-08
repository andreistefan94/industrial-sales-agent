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
    normalized_email = email.strip().lower()
    for customer in _CUSTOMERS:
        if customer["email"].lower() == normalized_email:
            return customer
    return {"error": f"There is no customer recorded with the email address '{email}'."}


@tool
def find_equipment(model: str | None = None, serial_number: str | None = None) -> list[dict]:
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


