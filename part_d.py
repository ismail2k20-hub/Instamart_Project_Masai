
"""
Part D implementation for the Instamart AI/ML capstone.
Run from the repository root after Parts A-C and generate_data.py have been run.
Uses only the supplied offline mock_llm.py; no API key or network access.
"""
from pathlib import Path
import csv
import json
import sqlite3
import time
import uuid

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(".")
POLICY_DIR = ROOT / "data" / "policies"
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

import mock_llm


def ensure_order_database():
    """Create the Part A SQLite database from the supplied synthetic CSV if absent.

    If your Part A code has already created data/instamart.db, this function leaves
    that database untouched. The fallback makes this Part D submission runnable
    with the supplied generate_data.py alone.
    """
    if DB_PATH.exists():
        return
    csv_path = RAW_DIR / "orders.csv"
    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    # Keep one canonical record per order. Prefer a non-pending record, then the
    # latest timestamp; this avoids the planted earlier pending duplicates.
    canonical = {}
    for row in rows:
        order_id = row.get("order_id", "")
        if not order_id:
            continue
        rank = (row.get("status") != "pending", row.get("order_ts", ""))
        if order_id not in canonical or rank > canonical[order_id][0]:
            canonical[order_id] = (rank, row)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""CREATE TABLE orders (
            order_id TEXT PRIMARY KEY, status TEXT, order_amount REAL,
            delivery_minutes INTEGER
        )""")
        for _, row in canonical.values():
            try:
                amount = float(row["order_amount"])
            except (TypeError, ValueError):
                amount = None
            try:
                minutes = int(row["delivery_minutes"]) if row["delivery_minutes"] else None
            except (TypeError, ValueError):
                minutes = None
            connection.execute(
                "INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?)",
                (row["order_id"], row["status"], amount, minutes),
            )

ensure_order_database()


# ---------------------------------------------------------------------
# TASKS 1-3: chunking, TF-IDF retrieval, and policy-grounded answering
# ---------------------------------------------------------------------
def read_policy_documents():
    docs = {}
    for path in sorted(POLICY_DIR.glob("*.txt")):
        docs[path.name] = path.read_text(encoding="utf-8").split()
    if len(docs) != 5:
        raise FileNotFoundError(
            f"Expected five policy .txt files in {POLICY_DIR}; found {len(docs)}. "
            "Run python generate_data.py from the repository root first."
        )
    return docs


def make_chunks(overlap=10, chunk_size=40):
    """Return chunks with source names; overlap=10 means next start is 30."""
    documents = read_policy_documents()
    chunks = []
    step = chunk_size - overlap
    for source, words in documents.items():
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append({"source": source, "text": " ".join(words[start:end])})
            if end >= len(words):
                break
            start += step
    return chunks


def build_retriever(overlap=10):
    chunks = make_chunks(overlap=overlap)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    return chunks, vectorizer, matrix


CHUNKS, VECTORIZER, CHUNK_MATRIX = build_retriever(overlap=10)


def retrieve(query, k=3, chunks=None, vectorizer=None, matrix=None):
    """Return the k best chunks as dictionaries with source, score and text."""
    if chunks is None:
        chunks, vectorizer, matrix = CHUNKS, VECTORIZER, CHUNK_MATRIX
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    # Stable sorting makes ties reproducible.
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    return [
        {"source": chunks[i]["source"], "score": float(scores[i]), "text": chunks[i]["text"]}
        for i in ranked
    ]


POLICY_SYSTEM = (
    "You are a POLICY ASSISTANT. Answer only from the supplied policy context. "
    "If the answer is not present in that context, say exactly: "
    "I could not find that information in the policy documents."
)


def answer(question):
    results = retrieve(question, k=3)
    context = "\n".join(f"[{item['source']}]\n{item['text']}" for item in results)
    user_content = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    response = mock_llm.chat([
        {"role": "system", "content": POLICY_SYSTEM},
        {"role": "user", "content": user_content},
    ])
    return {"answer": response.choices[0].message.content, "sources": results}


def evaluate_retrieval(overlap):
    chunks, vectorizer, matrix = build_retriever(overlap=overlap)
    evaluation = pd.read_csv(RAW_DIR / "policy_eval.csv", keep_default_na=False)
    answerable_hits = 0
    fact_hits = 0
    fallback_hits = 0
    failures = []
    for _, row in evaluation.iterrows():
        question = row["question"]
        results = retrieve(question, 3, chunks, vectorizer, matrix)
        result_sources = [item["source"] for item in results]
        context = "\n".join(f"[{x['source']}]\n{x['text']}" for x in results)
        response = mock_llm.chat([
            {"role": "system", "content": POLICY_SYSTEM},
            {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"},
        ])
        answer_text = response.choices[0].message.content or ""
        if row["expected_doc"] != "NONE":
            if row["expected_doc"] in result_sources:
                answerable_hits += 1
            if row["expected_fact"] in answer_text:
                fact_hits += 1
            if row["expected_doc"] not in result_sources or row["expected_fact"] not in answer_text:
                failures.append({
                    "question": question,
                    "expected_doc": row["expected_doc"],
                    "top_sources": result_sources,
                    "answer": answer_text,
                })
        elif answer_text.startswith("I could not find"):
            fallback_hits += 1
        else:
            failures.append({"question": question, "answer": answer_text, "expected": "fallback"})
    return {
        "overlap": overlap,
        "chunks": len(chunks),
        "expected_doc_hits": f"{answerable_hits}/12",
        "expected_fact_hits": f"{fact_hits}/12",
        "unanswerable_fallbacks": f"{fallback_hits}/2",
        "failures": failures,
    }


# ---------------------------------------------------------------------
# TASK 5: refund tools and tool specifications
# ---------------------------------------------------------------------
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order's status, amount and delivery time in the SQLite database.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    {
        "name": "search_policy",
        "description": "Retrieve the three policy chunks most relevant to a policy query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "calculate_refund",
        "description": "Calculate a full refund for completed orders delivered after 45 minutes, otherwise zero.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": ["number", "null"]},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an automatic refund of INR 1000 or less in the simulated ledger.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund above INR 1000 for human review without issuing it automatically.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
            "required": ["order_id", "refund_inr"],
        },
    },
]


def get_order(order_id):
    if not DB_PATH.exists():
        return {"error": "database not found"}
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            "SELECT order_id, status, order_amount, delivery_minutes "
            "FROM orders WHERE order_id = ? LIMIT 1",
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]) if row[2] is not None else None,
        "delivery_minutes": int(row[3]) if row[3] is not None else None,
    }


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and float(delivery_minutes) > 45:
        amount = round(float(order_amount), 2)
    else:
        amount = 0.0
    return {"refund_inr": amount}


def _record_ledger(tool_name, order_id, refund_inr):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }
    with (OUTPUT_DIR / "refund_ledger.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")
    return {"refund_inr": round(float(refund_inr), 2)}


def issue_refund(order_id, refund_inr):
    if float(refund_inr) > 1000:
        return {"error": "refund above INR 1000 must be escalated"}
    return _record_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _record_ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


# ---------------------------------------------------------------------
# TASKS 6-7: ReAct agent, guardrails, limits, logging and checkpoints
# ---------------------------------------------------------------------
AGENT_SYSTEM = (
    "You are an AGENT that follows the refund policy. Use tools to look up the order, "
    "search the policy, calculate the refund, then issue a refund of INR 1000 or less "
    "or escalate a larger refund. Only completed orders can be refunded. Never issue "
    "a refund when the calculated amount is zero. Give a concise final status."
)


def _json_default(value):
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _tool_calls_from_message(message):
    calls = []
    for call in (getattr(message, "tool_calls", None) or []):
        function = getattr(call, "function", None)
        if function is not None:
            name = function.name
            arguments = function.arguments
            call_id = call.id
        else:
            name = call.name
            arguments = call.arguments
            call_id = call.id
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        calls.append({"id": call_id, "name": name, "arguments": arguments})
    return calls


def _checkpoint_path(run_id):
    return STATE_DIR / f"{run_id}.json"


def _save_checkpoint(run_id, scenario_id, messages, tokens_used, steps, tools_executed, llm_calls):
    data = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "steps": steps,
        "tools_executed": tools_executed,
        "llm_calls": llm_calls,
    }
    _checkpoint_path(run_id).write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def _agent_loop(run_id, scenario_id, messages, tokens_used, steps, tools_executed,
                llm_calls, max_iterations, token_limit, time_limit_seconds,
                started_at=None, crash_after_tool_count=None):
    if started_at is None:
        started_at = time.monotonic()
    while llm_calls < max_iterations:
        if time.monotonic() - started_at >= time_limit_seconds:
            return {
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK_TIMEOUT: handing this request to a human; time limit reached.",
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "tools_executed": tools_executed, "run_id": run_id,
            }

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += int(response.usage.total_tokens)

        if tokens_used > token_limit:
            return {
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK_TOKEN_LIMIT: handing this request to a human; token limit exceeded.",
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "tools_executed": tools_executed, "run_id": run_id,
            }

        message = response.choices[0].message
        tool_calls = _tool_calls_from_message(message)
        if not tool_calls:
            final_answer = message.content or "FALLBACK_MAX_ITERATIONS: handing this request to a human."
            status = final_answer.split(":", 1)[0]
            return {
                "status": status, "final_answer": final_answer, "steps": steps,
                "llm_calls": llm_calls, "tokens_used": tokens_used,
                "tools_executed": tools_executed, "run_id": run_id,
            }

        messages.append({
            "role": "assistant", "content": None,
            "tool_calls": tool_calls,
        })
        for call in tool_calls:
            tool_name = call["name"]
            arguments = call["arguments"]
            if tool_name not in TOOL_MAP:
                observation = {"error": f"unknown tool: {tool_name}"}
            else:
                observation = TOOL_MAP[tool_name](**arguments)
            tools_executed.append(tool_name)
            steps.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": tool_name,
                "content": json.dumps(observation, default=_json_default),
            })
            # Save only after the tool observation is safely appended.
            _save_checkpoint(run_id, scenario_id, messages, tokens_used, steps, tools_executed, llm_calls)
            if crash_after_tool_count is not None and len(tools_executed) == crash_after_tool_count:
                raise RuntimeError("SIMULATED_CRASH_AFTER_CHECKPOINT")
        # Continue to next model call.

    return {
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK_MAX_ITERATIONS: handing this request to a human; maximum model calls reached.",
        "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
        "tools_executed": tools_executed, "run_id": run_id,
    }


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tool_count=None):
    run_id = run_id or f"{scenario_id}_{uuid.uuid4().hex[:8]}"
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": user_message},
    ]
    return _agent_loop(
        run_id, scenario_id, messages, 0, [], [], 0,
        max_iterations, token_limit, time_limit_seconds,
        started_at=time.monotonic(),
        crash_after_tool_count=crash_after_tool_count,
    )


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    path = _checkpoint_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found for run_id={run_id}")
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    return _agent_loop(
        run_id=checkpoint["run_id"],
        scenario_id=checkpoint["scenario_id"],
        messages=checkpoint["messages"],
        tokens_used=checkpoint["tokens_used"],
        steps=checkpoint["steps"],
        tools_executed=checkpoint["tools_executed"],
        llm_calls=checkpoint["llm_calls"],
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        started_at=time.monotonic(),
    )


def save_agent_run(result, scenario_id):
    line = {
        "scenario_id": scenario_id,
        "status": result["status"],
        "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "steps": result["steps"],
    }
    with (OUTPUT_DIR / "agent_runs.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(line, default=_json_default) + "\n")


def _refund_from_result(result):
    for step in result["steps"]:
        observation = step["observation"]
        if "refund_inr" in observation:
            return observation["refund_inr"]
    # For issued/escalated actions the amount is present in their arguments.
    for step in result["steps"]:
        if step["tool_name"] in ("issue_refund", "escalate_to_human"):
            return step["arguments"].get("refund_inr", 0)
    return 0


def run_all_scenarios():
    scenarios = pd.read_csv(RAW_DIR / "agent_scenarios.csv", keep_default_na=False)
    results = []
    for _, row in scenarios.iterrows():
        result = run_agent(row["message"], row["scenario_id"])
        save_agent_run(result, row["scenario_id"])
        actual_refund = _refund_from_result(result)
        results.append({
            "scenario_id": row["scenario_id"],
            "status": result["status"],
            "refund_amount_inr": round(float(actual_refund), 2),
            "expected_outcome": row["expected_outcome"],
            "expected_refund_inr": float(row["expected_refund_inr"]),
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
        })
    report = pd.DataFrame(results)
    report.to_csv(OUTPUT_DIR / "agent_scenario_results.csv", index=False)
    return report


def demonstrate_fallbacks():
    scenarios = pd.read_csv(RAW_DIR / "agent_scenarios.csv", keep_default_na=False)
    s1 = scenarios.loc[scenarios["scenario_id"] == "S1"].iloc[0]
    token_demo = run_agent(s1["message"], "S1_TOKEN_DEMO", token_limit=200)
    timeout_demo = run_agent(s1["message"], "S1_TIMEOUT_DEMO", time_limit_seconds=0)
    return token_demo, timeout_demo


def demonstrate_crash_resume():
    scenarios = pd.read_csv(RAW_DIR / "agent_scenarios.csv", keep_default_na=False)
    s1 = scenarios.loc[scenarios["scenario_id"] == "S1"].iloc[0]
    run_id = "S1_crash_resume_demo"
    try:
        run_agent(s1["message"], "S1", run_id=run_id, crash_after_tool_count=2)
    except RuntimeError as exc:
        if str(exc) != "SIMULATED_CRASH_AFTER_CHECKPOINT":
            raise
    result = resume_agent(run_id)
    return result


# ---------------------------------------------------------------------
# TASK 4 and final report: run policy evaluation and all agent demos
# ---------------------------------------------------------------------
def main():
    # Start fresh so each execution produces exactly seven scenario lines and a
    # ledger for the current demonstration run rather than duplicating old runs.
    for output_file in (OUTPUT_DIR / "agent_runs.jsonl", OUTPUT_DIR / "refund_ledger.jsonl"):
        if output_file.exists():
            output_file.unlink()
    print("TASK 1: Chunk count by policy")
    chunk_df = pd.DataFrame(CHUNKS)
    print(chunk_df.groupby("source").size().to_string())
    print("Total chunks:", len(CHUNKS))
    assert len(CHUNKS) == 15, f"Expected 15 chunks, got {len(CHUNKS)}"

    print("\nTASK 2: Example retrieval for 'refund policy late delivery'")
    for item in retrieve("refund policy late delivery", 3):
        print(item["source"], round(item["score"], 4), item["text"])

    print("\nTASK 3: Example grounded answer")
    print(answer("What delivery time makes a completed order eligible for a full refund?"))

    print("\nTASK 4: Policy evaluation (overlap=10)")
    eval_overlap = evaluate_retrieval(overlap=10)
    print({k: v for k, v in eval_overlap.items() if k != "failures"})
    print("Failures:", eval_overlap["failures"])

    print("\nTASK 4: Policy evaluation (overlap=0)")
    eval_no_overlap = evaluate_retrieval(overlap=0)
    print({k: v for k, v in eval_no_overlap.items() if k != "failures"})
    print("Failures:", eval_no_overlap["failures"])

    print("\nTASKS 5-6: Run all seven refund scenarios")
    scenario_report = run_all_scenarios()
    print(scenario_report.to_string(index=False))

    print("\nTASK 7: Fallback demonstrations")
    token_demo, timeout_demo = demonstrate_fallbacks()
    print("Token demo:", token_demo["status"], token_demo["llm_calls"], token_demo["tokens_used"])
    print("Timeout demo:", timeout_demo["status"], timeout_demo["llm_calls"], timeout_demo["tokens_used"])

    print("\nTASK 8: Crash-and-resume demonstration")
    resumed = demonstrate_crash_resume()
    print("Status:", resumed["status"])
    print("Final answer:", resumed["final_answer"])
    print("Tools executed:", resumed["tools_executed"])
    print("Tool steps:", len(resumed["steps"]))

    print("\nFinished. Inspect outputs/ and state/ for generated files.")


if __name__ == "__main__":
    main()


