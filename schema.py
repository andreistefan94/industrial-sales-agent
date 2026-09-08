from typing import Literal, TypedDict

from pydantic import BaseModel


class PartRequested(BaseModel):
    customer_description: str
    part_number: str | None = None
    quantity: int | None = None


class RequestParts(BaseModel):
    equipment_model: str | None
    equipment_serial_number: str | None
    parts_requested: list[PartRequested]
    missing_information: list[str]


class RequestState(TypedDict, total=False):
    request_id: str
    email_text: str
    request: dict
    equipment_id: str
    parts: list[dict]
    offer: list[dict]
    missing_information: list[str]
    draft_reply: str
    status: str


class CheckAnswer(BaseModel):
    valid_parts: bool
    valid_prices: bool
    valid_availability: bool
    halucinating: bool
    observations: list[str]


HumanDecision = Literal["part_selected", "ask_clarification"]
