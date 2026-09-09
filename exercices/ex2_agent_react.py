"""
Exercise 2 -- Part Identification.
A ReAct agent (create_agent from langchain 1.x) receives the structured request (Ex1) and decides on its own, at runtime, 
in what order and how many times to call find_equipment / search_parts to arrive at an equipment_id and part_numbers. 
Here, the number and order of calls vary depending on what each tool returns -- hence the autonomy of ReAct, unlike Ex1 (a single LLM call) or the deterministic steps in Ex3 (stock/price check).
Central rule from the statement: the agent does not choose arbitrarily when the result is ambiguous (multiple equipment or multiple candidate parts) -- it reports the ambiguity, it does not guess.
"""

import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import BASE_DIR, MODEL, find_equipment, search_parts
from ex1_extractie import extract_request
from schema import RequestParts

load_dotenv()


class CandidatePart(BaseModel):
    part_number: str
    name: str


class PartIdentified(BaseModel):
    customer_description: str
    quantity: int | None = None
    part_number: str | None = None
    name: str | None = None
    canidates: list[CandidatePart] = []
    not_found: bool = False


class IdentificationResult(BaseModel):
    equipment_id: str | None
    equipment_ambiguity: bool
    parts: list[PartIdentified]
    intervention_needed: bool
    observations: list[str]


SYSTEM_PROMPT = """Esti un agent care identifica echipamentul si piesele de schimb pornind \
de la o solicitare deja structurata (extrasa dintr-un email de client). Ai la dispozitie \
doar doua tool-uri: find_equipment si search_parts.

Pasi:
1. Daca solicitarea are model_echipament si/sau serie_echipament, foloseste find_equipment \
ca sa gasesti equipment_id. Daca ai si model si serie, cauta cu ambele -- cautarea devine \
exacta. Daca find_equipment intoarce mai multe rezultate (de obicei pentru ca lipseste \
seria si modelul respectiv are mai multe echipamente inregistrate), NU alege unul arbitrar: \
lasa equipment_id necompletat, seteaza echipament_ambiguu=true si adauga o observatie clara.
2. Pentru fiecare intrare din piese_solicitate, foloseste search_parts cu descrierea \
clientului (descriere_client) ca query. Daca ai deja un equipment_id sigur, foloseste-l ca \
sa restrangi cautarea la piese compatibile cu acel model. Daca nu ai inca equipment_id \
(pentru ca e ambiguu sau lipseste), poti cauta si fara equipment_id.
3. search_parts poate intoarce:
   - 0 rezultate -> incearca O SINGURA reformulare a query-ului (ex. forma de singular in \
loc de plural, fara cuvinte de legatura precum "de"/"pentru"), pentru ca de multe ori \
alias-urile din catalog nu contin toate formele gramaticale. Daca si a doua incercare da \
0 rezultate, piesa nu exista in catalog pentru acel query/model: seteaza negasita=true \
pentru acea piesa si adauga o observatie. NU incerca mai mult de doua query-uri diferite \
pentru aceeasi piesa.
   - exact 1 rezultat -> identificare certa: completeaza part_number si nume.
   - 2 sau mai multe rezultate -> ambiguitate reala (ex. mai multe tipuri de filtru de \
ulei). NU alege una arbitrar. Pune toate rezultatele in campul candidati (part_number + \
nume pentru fiecare) si lasa part_number/nume necompletate pentru acea piesa. NU mai \
repeta acelasi apel sau un apel echivalent pentru aceasta piesa -- ambiguitatea e deja \
stabilita, treci mai departe.
4. Daca oricare piesa e negasita, are candidati multipli, sau echipamentul e ambiguu, \
seteaza necesita_interventie=true la nivel global.
5. Nu inventa niciodata un equipment_id, part_number sau nume care nu a fost intors chiar \
de un tool -- foloseste exact valorile primite ca raspuns la apelurile de tool-uri.
6. Fii eficient: nu apela un tool cu argumente identice sau echivalente cu un apel deja \
facut in aceasta conversatie -- refoloseste rezultatul deja obtinut. De indata ce ai un \
rezultat (cert, ambiguu sau negasit) pentru fiecare piesa din piese_solicitate si ai \
stabilit equipment_id (sau ambiguitatea lui), opreste-te din apelat tool-uri si raspunde \
direct in formatul structurat cerut.

Raspunde in final in formatul structurat cerut."""

# Safety limit for the ReAct loop (model -> tool -> model -> ...). A normal request
# needs at most 4-6 tool calls (one find_equipment + one search_parts per part). Without a "forced" structured response (see ToolStrategy below), with
# a small model (gemini flash-lite) the agent could never converge -- it would repeat the same search_parts with equivalent reformulations even after 
# it had already found a certain result 
RECURSION_LIMIT = 12


def build_agent():
    return create_agent(
        model=ChatGoogleGenerativeAI(model=MODEL, temperature=0),
        tools=[find_equipment, search_parts],
        system_prompt=SYSTEM_PROMPT,
        # ToolStrategy obliga modelul sa apeleze explicit un tool "de finalizare" cu
        # schema ceruta, in loc sa lase raspunsul structurat la latitudinea modelului
        # (AutoStrategy) -- asta a rezolvat in practica bucla in care modelul verifica
        # la nesfarsit piese deja identificate, in loc sa se opreasca.
        response_format=ToolStrategy(schema=IdentificationResult),
    )


def _recursion_error_result() -> IdentificationResult:
    return IdentificationResult(
        equipment_id=None,
        equipment_ambiguity=False,
        parts=[],
        intervention_needed=True,
        observations=[
            f"The agent did not manage to converge to a response within the limit of "
            f"{RECURSION_LIMIT} iterations -- manual intervention is needed."
        ],
    )


def indentify_part(agent, request: RequestParts) -> IdentificationResult:
    message = request.model_dump_json(indent=2)
    try:
        res = agent.invoke(
            {"messages": [("human", message)]}, config={"recursion_limit": RECURSION_LIMIT}
        )
    except GraphRecursionError:
        return _recursion_error_result()
    return res["structured_response"]


def _display_trace(agent, request: RequestParts) -> IdentificationResult:
    message = request.model_dump_json(indent=2)
    print("Trace (tool calls):")
    last_step = None
    try:
        for step in agent.stream(
            {"messages": [("human", message)]},
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode="values",
        ):
            last_step = step
            msg = step["messages"][-1]
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    print(f"  -> {tc['name']}({tc['args']})")
            if msg.__class__.__name__ == "ToolMessage":
                print(f"     <- {msg.content}")
    except GraphRecursionError:
        print(f"  !! A limit of {RECURSION_LIMIT} iterations was reached without a final structured response -- escalation to operator.")
        return _recursion_error_result()
    return last_step["structured_response"]


def _run_test_scenarios() -> None:
    with open(BASE_DIR / "test_cases.jsonl", encoding="utf-8") as f:
        scenarios = [json.loads(line) for line in f if line.strip()]

    agent = build_agent()

    for scenario in scenarios:
        if scenario["type"] != "email":
            continue
        print(f"\n=== {scenario['request_id']} -- {scenario['expected_result']} ===")
        request = extract_request(scenario["email_text"])
        print(f"Request extracted: {request.model_dump_json()}")
        res = _display_trace(agent, request)
        print("Identification result:")
        print(res.model_dump_json(indent=2))


if __name__ == "__main__":
    _run_test_scenarios()
