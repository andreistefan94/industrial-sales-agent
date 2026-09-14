"""Exercise 4 -- A resumable industrial sales workflow.

The graph pauses at the two points that require a person: resolving an
ambiguous identification and supplying information requested from the customer.
It also validates the generated draft and allows one regeneration when the
validation fails.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import BASE_DIR, MODEL, check_stock, get_customer, get_price
from exercices.ex1_extraction import extract_request
from exercices.ex2_agent_react import IdentificationResult, build_agent, indentify_part
from schema import CheckAnswer, PartRequested, RequestParts, RequestState

load_dotenv()

_identifying_agent = None
MAX_DRAFT_ATTEMPTS = 2


def _text(message) -> str:
	content = message.content
	if isinstance(content, str):
		return content
	return "\n".join(
		part["text"]
		for part in content
		if isinstance(part, dict) and part.get("type") == "text"
	)


def _agent():
	global _identifying_agent
	if _identifying_agent is None:
		_identifying_agent = build_agent()
	return _identifying_agent


def node_message_received(state: RequestState) -> dict:
	return {"status": "message_received"}


def node_extract_request(state: RequestState) -> dict:
	request = extract_request(state["email_text"])
	customer = get_customer.func(state["sender_email"])
	missing = list(request.missing_information)
	customer_id = customer.get("customer_id")
	if customer_id is None:
		missing.append(f"Customer unknown: {customer.get('error')}")
	return {
		"request": request.model_dump(),
		"customer_id": customer_id,
		"missing_information": missing,
		"status": "request_extracted",
	}


def node_identify_parts(state: RequestState) -> dict:
	request = RequestParts(
		equipment_model=state["request"].get("equipment_model"),
		equipment_serial_number=state["request"].get("equipment_serial_number"),
		parts_requested=[
			PartRequested(**part) for part in state["request"].get("parts_requested", [])
		],
		missing_information=state["request"].get("missing_information", []),
	)
	result: IdentificationResult = indentify_part(_agent(), request)
	missing = list(state.get("missing_information", []))
	if result.equipment_ambiguity:
		missing.append("equipment_serial_number (more than one equipment matched)")
	missing.extend(result.observations)
	return {
		"equipment_id": result.equipment_id,
		"parts": [part.model_dump() for part in result.parts],
		"missing_information": missing,
		"status": "parts_identified",
	}


def _part_is_ambiguous(part: dict) -> bool:
	return bool(part.get("candidates") or part.get("canidates") or part.get("not_found"))


def route_after_identification(state: RequestState) -> str:
	if not state.get("equipment_id"):
		return "generate_clarification"
	if any(_part_is_ambiguous(part) for part in state.get("parts", [])):
		return "human_intervention"
	return "check_stock_price"


def node_human_intervention(state: RequestState) -> dict:
	parts = [
		{
			"customer_description": part.get("customer_description"),
			"candidates": part.get("candidates") or part.get("canidates", []),
			"not_found": part.get("not_found", False),
		}
		for part in state.get("parts", [])
		if _part_is_ambiguous(part)
	]
	decision = interrupt(
		{
			"type": "human_intervention",
			"equipment_id": state.get("equipment_id"),
			"parts": parts,
			"allowed_decisions": ["part_selected", "ask_clarification"],
		}
	)
	if decision.get("decision") in {"part_selected", "piesa_selectata"}:
		part_number = decision["part_number"]
		updated_parts = []
		for part in state.get("parts", []):
			candidates = part.get("candidates") or part.get("canidates", [])
			if any(candidate.get("part_number") == part_number for candidate in candidates):
				selected = next(candidate for candidate in candidates if candidate["part_number"] == part_number)
				updated_parts.append({**part, "part_number": part_number, "name": selected.get("name"), "candidates": []})
			else:
				updated_parts.append(part)
		return {"parts": updated_parts, "status": "human_decision_applied"}
	if decision.get("decision") == "ask_clarification":
		return {"status": "clarification_needed"}
	raise ValueError("Unsupported human intervention decision")


def route_after_human_intervention(state: RequestState) -> str:
	if state.get("status") == "clarification_needed":
		return "generate_clarification"
	return "check_stock_price"


CLARIFICATION_PROMPT = """Scrie un email scurt si politicos in limba romana catre client. Cere exact informatiile din lista Missing information si nu inventa alte detalii. Returneaza doar corpul emailului."""


def node_generate_clarification(state: RequestState) -> dict:
	prompt = ChatPromptTemplate.from_messages(
		[("system", CLARIFICATION_PROMPT), ("human", "Missing information:\n{missing}")]
	)
	message = (prompt | ChatGoogleGenerativeAI(model=MODEL, temperature=0)).invoke(
		{"missing": "\n".join(f"- {item}" for item in state.get("missing_information", []))}
	)
	return {"draft_reply": _text(message), "status": "waiting_for_customer"}


def node_wait_for_clarification(state: RequestState) -> dict:
	answer = interrupt(
		{"type": "customer_clarification", "draft_reply": state.get("draft_reply", "")}
	)
	return {
		"email_text": f"{state['email_text']}\n\nCustomer reply:\n{answer}",
		"missing_information": [],
		"status": "customer_reply_received",
	}


def node_check_stock_price(state: RequestState) -> dict:
	offer = []
	for part in state.get("parts", []):
		quantity = part.get("quantity") or 1
		stock = check_stock.func(part["part_number"], quantity)
		price = get_price.func(part["part_number"], state["customer_id"], quantity)
		offer.append(
			{
				"part_number": part["part_number"],
				"name": part.get("name"),
				"quantity": quantity,
				"available": stock.get("available", False),
				"total_price": price.get("total_price"),
			}
		)
	return {"offer": offer, "status": "offer_ready"}


DRAFT_PROMPT = """Redacteaza un email in limba romana folosind exclusiv contextul JSON. Nu inventa coduri, cantitati, preturi sau conditii comerciale. Mentioneaza disponibilitatea fiecarei piese si totalul din oferta. Returneaza doar corpul emailului."""


def node_generate_draft(state: RequestState) -> dict:
	prompt = ChatPromptTemplate.from_messages(
		[("system", DRAFT_PROMPT), ("human", "Context:\n{context}")]
	)
	context = json.dumps(
		{
			"equipment_model": state["request"].get("equipment_model"),
			"equipment_serial_number": state["request"].get("equipment_serial_number"),
			"offer": state.get("offer", []),
		},
		ensure_ascii=False,
	)
	message = (prompt | ChatGoogleGenerativeAI(model=MODEL, temperature=0)).invoke(
		{"context": context}
	)
	return {
		"draft_reply": _text(message),
		"draft_attempts": state.get("draft_attempts", 0) + 1,
		"status": "draft_generated",
	}


def _deterministic_draft_check(state: RequestState) -> CheckAnswer:
	draft = state.get("draft_reply", "").lower()
	offer = state.get("offer", [])
	valid_parts = all(item["part_number"].lower() in draft for item in offer)
	valid_quantities = all(str(item["quantity"]) in draft for item in offer)
	valid_availability = all(
		("disponibil" in draft or "stoc" in draft) for item in offer
	)
	total = round(sum(item.get("total_price") or 0 for item in offer), 2)
	total_values = {str(total), str(int(total)) if total.is_integer() else ""}
	total_values.update(value.replace(".", ",") for value in tuple(total_values) if value)
	valid_prices = any(value and value in draft for value in total_values)
	return CheckAnswer(
		valid_parts=valid_parts and valid_quantities,
		valid_prices=valid_prices,
		valid_availability=valid_availability,
		halucinating=False,
		observations=[] if valid_parts and valid_prices else ["Draft does not contain all factual offer data."],
	)


def node_check_draft(state: RequestState) -> dict:
	validation = _deterministic_draft_check(state)
	valid = validation.valid_parts and validation.valid_prices and validation.valid_availability
	return {
		"draft_validation": validation.model_dump(),
		"status": "draft_valid" if valid else "draft_invalid",
	}


def route_after_draft_check(state: RequestState) -> str:
	if state.get("status") == "draft_valid":
		return "end"
	if state.get("draft_attempts", 0) < MAX_DRAFT_ATTEMPTS:
		return "regenerate_draft"
	return "end"


def build_graph():
	graph = StateGraph(RequestState)
	graph.add_node("message_received", node_message_received)
	graph.add_node("extract_request", node_extract_request)
	graph.add_node("identify_parts", node_identify_parts)
	graph.add_node("human_intervention", node_human_intervention)
	graph.add_node("generate_clarification", node_generate_clarification)
	graph.add_node("wait_for_clarification", node_wait_for_clarification)
	graph.add_node("check_stock_price", node_check_stock_price)
	graph.add_node("generate_draft", node_generate_draft)
	graph.add_node("check_draft", node_check_draft)

	graph.add_edge(START, "message_received")
	graph.add_edge("message_received", "extract_request")
	graph.add_edge("extract_request", "identify_parts")
	graph.add_conditional_edges(
		"identify_parts",
		route_after_identification,
		{
			"generate_clarification": "generate_clarification",
			"human_intervention": "human_intervention",
			"check_stock_price": "check_stock_price",
		},
	)
	graph.add_conditional_edges(
		"human_intervention",
		route_after_human_intervention,
		{"generate_clarification": "generate_clarification", "check_stock_price": "check_stock_price"},
	)
	graph.add_edge("generate_clarification", "wait_for_clarification")
	graph.add_edge("wait_for_clarification", "extract_request")
	graph.add_edge("check_stock_price", "generate_draft")
	graph.add_edge("generate_draft", "check_draft")
	graph.add_conditional_edges(
		"check_draft",
		route_after_draft_check,
		{"regenerate_draft": "generate_draft", "end": END},
	)
	return graph.compile(checkpointer=InMemorySaver())


def run_with_resume(initial_state: RequestState, resume_values: list) -> dict:
	graph = build_graph()
	config = {"configurable": {"thread_id": initial_state["request_id"]}}
	state = graph.invoke(initial_state, config=config)
	for value in resume_values:
		state = graph.invoke(Command(resume=value), config=config)
	return state


def _run_test_scenarios() -> None:
	with open(BASE_DIR / "test_cases.jsonl", encoding="utf-8") as file:
		scenarios = [json.loads(line) for line in file if line.strip()]
	for scenario in scenarios:
		if scenario["type"] != "email":
			continue
		state: RequestState = {
			"request_id": scenario["request_id"],
			"email_text": scenario["email_text"],
			"sender_email": scenario["sender_email"],
		}
		print(f"\n=== {scenario['request_id']} ===")
		print(run_with_resume(state, [retry["resume"] for retry in scenario.get("retries", [])]))


if __name__ == "__main__":
	_run_test_scenarios()
