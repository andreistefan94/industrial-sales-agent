# industrial-sales-agent

## Project purpose

This project builds an AI agent that automates the processing of commercial inquiries received by email from an industrial equipment supplier.

For example, a customer may send:

```text
Hello,

We have an ACX-200 compressor, serial number SN-48392.
We urgently need two oil filters and the gasket for the separator cover.

Could you confirm the part numbers, availability, and price?
```

In practice, messages may contain incomplete or informal information. The customer may provide the equipment model and serial number, describe a part in everyday language, or omit information that is required. The agent must be able to:

- interpret the incoming email;
- identify the customer, equipment, and requested parts;
- confirm the part numbers;
- check stock and pricing for the requested quantity;
- request clarification or operator assistance when information is insufficient;
- generate a draft response for the customer;
- continue processing when a customer replies to a clarification request later.

The project applies concepts from previous courses to make concrete implementation decisions and combine LLM reasoning with deterministic operations on the supplier's data.

## Workflow architecture

The agent does not access the dataset files directly. The LLM decides what information needs to be looked up, while tools perform the deterministic operations:

```text
get_customer(email)
find_equipment(model=None, serial_number=None)
search_parts(query, equipment_id=None)
check_stock(part_number, quantity)
get_price(part_number, customer_id, quantity)
```

Customer identity is important in two operations: `get_customer` identifies the sender, while `get_price` calculates the price associated with the customer. Equipment lookup should also take `customer_id` into account because industrial equipment may be specially configured for each customer. This prevents the same model or serial number from automatically being treated as belonging to any customer.

## Local data

The supplier's systems are simulated using the following files:

```text
data/
├── customers.json
├── equipment.json
├── parts.json
├── inventory.json
└── pricing.json
```

The code skeleton may include:

- `common.py` - tools that read data from the dataset;
- `schema.py` - the main Pydantic types used by the agent;
- `test_cases.jsonl` - test cases covering multiple scenarios;
- `requirements.txt` - project dependencies.
