"""Exercise 3 -- Building the Workflow with LangGraph.

The business flow is not left to the discretion of the agent: who calls which tool and in what order, inside the part identification, remains the job of the ReAct agent from Ex2.
But the path BETWEEN steps (extraction -> identification -> clarification / human intervention / offer -> draft) is deterministic code. The rules explicitly guaranteed in the graph (not left to the model):
    * the price comes EXCLUSIVELY from get_price (check_stock_price_node) -- no LLM node calculates or assumes a price;
    * availability is checked with check_stock before it reaches the offer or the draft;
    * an ambiguous identification (equipment or part) CANNOT reach check_stock_price_node -- route_after_identification explicitly blocks this path, regardless of what the agent from Ex2 "thinks         
"""

import json

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import BASE_DIR, MODEL, check_stock, get_customer, get_price
from ex1_extractie import extract_request
from ex2_agent_react import IdentificationResult, build_agent, indentify_part
from schema import PartRequested, RequestState, RequestParts

load_dotenv()

_indentifying_agent = None


def _extract_text(msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    return "\n".join(
        piece["text"] for piece in content if isinstance(piece, dict) and piece.get("type") == "text"
    )


def _get_indentifying_agent():
    global _indentifying_agent
    if _indentifying_agent is None:
        _indentifying_agent = build_agent()
    return _indentifying_agent


def extract_request_node(state: RequestState) -> dict:
    request = extract_request(state["email_text"])
    customer = get_customer.func(state["sender_email"])

    missing_information = list(request.missing_information)
    customer_id = customer.get("customer_id")
    if customer_id is None:
        missing_information.append(f"Customer unknown: {customer.get('error')}")

    return {
        "request": request.model_dump(),
        "customer_id": customer_id,
        "missing_information": missing_information,
        "status": "request_extracted",
    }


def indentifying_parts_node(state: RequestState) -> dict:
    request = RequestParts(
        equipment_model=state["request"].get("equipment_model"),
        equipment_serial_number=state["request"].get("equipment_serial_number"),
        parts_requested=[
            PartRequested(**p) for p in state["request"].get("parts_requested", [])
        ],
        missing_information=state["request"].get("missing_information", []),
    )
    agent = _get_indentifying_agent()
    res: IdentificationResult = indentify_part(agent, request)

    missing_information = list(state.get("missing_information", []))
    if res.equipment_ambiguity:
        missing_information.append("equipment_serial_number (more than one equipment matched)")
    missing_information.extend(res.observations)

    return {
        "equipment_id": res.equipment_id,
        "parts": [p.model_dump() for p in res.parts],
        "missing_information": missing_information,
        "status": "parts_identified"
    }


def route_after_identification(state: RequestState) -> str:
    if not state.get("equipment_id"):
        return "ask_clarification"
    parts = state.get("parts", [])
    ambiguity = any(p.get("candidates") or p.get("not_found") for p in parts)
    if ambiguity:
        return "human_intervention"
    return "check_stock_price"


CLARIFICATION_PROMPT = """Esti un asistent de vanzari care scrie un raspuns scurt si politicos \
catre un client, cerand EXACT informatiile lipsa listate mai jos -- nimic in plus, nu \
inventa alte intrebari sau detalii despre comanda. Raspunde doar cu textul emailului, in \
limba romana."""


def ask_clarification_node(state: RequestState) -> dict:
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", CLARIFICATION_PROMPT), ("human", "Missing information:\n{missing}")]
    )
    msg = (prompt | llm).invoke(
        {"missing": "\n".join(f"- {info}" for info in state.get("missing_information", []))}
    )
    return {"draft_reply": _extract_text(msg), "status": "waiting_for_clarification"}


def human_intervention_node(state: RequestState) -> dict:
    return {"status": "human_intervention_needed"}


def check_stock_price_node(state: RequestState) -> dict:
    offer = []
    for part in state.get("parts", []):
        part_number = part["part_number"]
        quantity = part.get("quantity") or 1

        stock = check_stock.func(part_number, quantity)
        price = get_price.func(part_number, state["customer_id"], quantity)

        offer.append(
            {
                "part_number": part_number,
                "name": part.get("nume"),
                "quantity": quantity,
                "available": stock.get("available", False),
                "total_price": price.get("total_price"),
            }
        )

    return {"offer": offer, "status": "offer_ready"}


DRAFT_PROMPT = """Esti un asistent de vanzari care redacteaza un raspuns catre un client, pe \
baza EXCLUSIVA a datelor din contextul de mai jos (echipament si oferta). Nu inventa piese, \
preturi, cantitati sau conditii comerciale care nu apar in context. Daca o piesa nu e \
disponibila, mentioneaza asta clar in loc sa o omiti. Raspunde doar cu textul emailului, in \
limba romana, in stilul exemplului din enunt (lista de piese, disponibilitate, total)."""


def generate_draft_node(state: RequestState) -> dict:
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", DRAFT_PROMPT), ("human", "Context:\n{context}")]
    )
    context = json.dumps(
        {
            "equipment_model": state["request"].get("equipment_model"),
            "equipment_serial_number": state["request"].get("equipment_serial_number"),
            "equipment_id": state.get("equipment_id"),
            "offer": state.get("offer", []),
        },
        ensure_ascii=False,
    )
    msg = (prompt | llm).invoke({"context": context})
    return {"draft_reply": _extract_text(msg), "status": "ready_to_send"}


def build_graph():
    graph = StateGraph(RequestState)
    graph.add_node("extract_request", extract_request_node)
    graph.add_node("indentifying_parts", indentifying_parts_node)
    graph.add_node("ask_clarification", ask_clarification_node)
    graph.add_node("human_intervention", human_intervention_node)
    graph.add_node("check_stock_price", check_stock_price_node)
    graph.add_node("generate_draft", generate_draft_node)

    graph.add_edge(START, "extract_request")
    graph.add_edge("extract_request", "indentifying_parts")
    graph.add_conditional_edges(
        "indentifying_parts",
        route_after_identification,
        {
            "ask_clarification": "ask_clarification",
            "human_intervention": "human_intervention",
            "check_stock_price": "check_stock_price",
        },
    )
    graph.add_edge("ask_clarification", END)
    graph.add_edge("human_intervention", END)
    graph.add_edge("check_stock_price", "generate_draft")
    graph.add_edge("generate_draft", END)

    return graph.compile()


def _run_test_scenarios() -> None:
    with open(BASE_DIR / "test_cases.jsonl", encoding="utf-8") as f:
        scenarios = [json.loads(line) for line in f if line.strip()]

    graph = build_graph()

    for scenario in scenarios:
        if scenario["type"] != "email":
            continue
        print(f"\n=== {scenario['request_id']} -- {scenario['expected_result']} ===")
        initial_state: RequestState = {
            "request_id": scenario["request_id"],
            "email_text": scenario["email_text"],
            "sender_email": scenario["sender_email"],
        }
        final_state = graph.invoke(initial_state, config={"recursion_limit": 25})
        print(f"final_state: {final_state.get('status')}")
        print(f"equipment_id: {final_state.get('equipment_id')}")
        print(f"missing_information: {final_state.get('missing_information')}")
        if final_state.get("offer"):
            print(f"offer: {json.dumps(final_state['offer'], ensure_ascii=False, indent=2)}")
        if final_state.get("draft_reply"):
            print(f"draft_reply:\n{final_state['draft_reply']}")


if __name__ == "__main__":
    _run_test_scenarios()
