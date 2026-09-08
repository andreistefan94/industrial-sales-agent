"""Exercise1 - Email extraction to structured output

This module contains the function `extract_request` which takes an email text as input and extracts the structured information about the requested parts for industrial equipment. 
The extraction is performed using a language model with a structured output schema defined by the `RequestParts` Pydantic model.
"""
import json

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import BASE_DIR, MODEL
from schema import RequestParts

load_dotenv()

SYSTEM_PROMPT = """Esti un asistent care extrage dintr-un email de la un client structura \
solicitarii de piese de schimb pentru echipamente industriale.

Reguli stricte:
- Extrage DOAR informatiile care apar explicit sau clar implicit in mesaj.
- Nu inventa niciodata coduri de piesa, serii de echipament sau cantitati care nu sunt \
scrise in mesaj. Daca o cantitate nu e mentionata, las-o necompletata (None), nu presupune 1.
- Daca clientul descrie o piesa informal (ex. "filtrul acela mare de aer"), pune acea \
descriere in descriere_client exact cum a scris-o clientul -- identificarea codului real \
al piesei se face ulterior, de catre alt sistem, nu de tine.
- Daca modelul echipamentului, seria, sau alte informatii necesare identificarii precise \
lipsesc din mesaj, adauga cate o intrare descriptiva in informatii_lipsa (ex. \
"serie_echipament").
- Nu incerca sa ghicesti sau sa completezi informatii lipsa -- raporteaza-le ca lipsa."""


def extract_request(email_text: str) -> RequestParts:
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
    structured_llm = llm.with_structured_output(RequestParts)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{email_text}")]
    )
    chain = prompt | structured_llm
    return chain.invoke({"email_text": email_text})


def _run_test_scenarios() -> None:
    with open(BASE_DIR / "test_cases.jsonl", encoding="utf-8") as f:
        scenarios = [json.loads(line) for line in f if line.strip()]

    for scenario in scenarios:
        if scenario["type"] != "email":
            continue
        print(f"\n=== {scenario['request_id']} -- {scenario['expected_result']} ===")
        print(f"Email:\n{scenario['email_text']}\n")
        res = extract_request(scenario["email_text"])
        print("Body:")
        print(res.model_dump_json(indent=2, exclude_none=False))


if __name__ == "__main__":
    _run_test_scenarios()
