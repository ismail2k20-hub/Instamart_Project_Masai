
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import mock_llm


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
TICKETS_PATH = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_PATH = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

TOKEN_BUDGET = 6000
FINAL_TOKEN_BUDGET = 100000
BASE_DELAY = 0.05

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_ID_RE = re.compile(r"ORD-\d{6}")

RETRY_LOG = []


# ------------------------------------------------------------
# Prompts
# Both prompts must include CLASSIFY.
# Zero-shot must not contain EXAMPLES:.
# Few-shot must contain EXAMPLES: and a fraud example.
# ------------------------------------------------------------

ZERO_SHOT_PROMPT = """
CLASSIFY customer support tickets.

Choose exactly one category:
billing, delivery, technical, fraud.

Return JSON only in this exact structure:
{"category": "billing", "reason": "brief reason"}

The category must be one of the four allowed categories.
Do not include Markdown or text outside the JSON.
""".strip()


FEW_SHOT_PROMPT = """
CLASSIFY customer support tickets.

Choose exactly one category:
billing, delivery, technical, fraud.

EXAMPLES:
Ticket: I was charged twice for the same order.
Output: {"category": "billing", "reason": "duplicate charge"}

Ticket: My parcel never arrived and the rider is missing.
Output: {"category": "delivery", "reason": "parcel not delivered"}

Ticket: Someone hacked my account and made an unauthorized purchase.
Output: {"category": "fraud", "reason": "unauthorized account activity"}

Ticket: The app crashes every time I try to log in.
Output: {"category": "technical", "reason": "app login problem"}

Now classify the supplied ticket.
Return JSON only in this exact structure:
{"category": "billing", "reason": "brief reason"}

Use only billing, delivery, technical, or fraud.
Do not include Markdown or text outside the JSON.
""".strip()


assert "CLASSIFY" in ZERO_SHOT_PROMPT
assert "EXAMPLES:" not in ZERO_SHOT_PROMPT
assert "CLASSIFY" in FEW_SHOT_PROMPT
assert "EXAMPLES:" in FEW_SHOT_PROMPT
assert '"category": "fraud"' in FEW_SHOT_PROMPT


# ------------------------------------------------------------
# PII redaction
# Apply patterns in the required order.
# ------------------------------------------------------------

def redact_text(text):
    text = text or ""

    text, email_count = EMAIL_RE.subn("[EMAIL]", text)
    text, phone_count = PHONE_RE.subn("[PHONE]", text)
    text, order_count = ORDER_ID_RE.subn("[ORDER_ID]", text)

    counts = {
        "emails": email_count,
        "phones": phone_count,
        "order_ids": order_count,
    }

    # Ensure no protected values remain after redaction.
    assert EMAIL_RE.search(text) is None, "Email remains after redaction"
    assert PHONE_RE.search(text) is None, "Phone number remains after redaction"
    assert ORDER_ID_RE.search(text) is None, "Order ID remains after redaction"

    return text, counts


def prepare_tickets(tickets):
    prepared = []
    totals = {"emails": 0, "phones": 0, "order_ids": 0}

    for _, row in tickets.iterrows():
        original = "" if pd.isna(row["text"]) else str(row["text"])
        redacted, counts = redact_text(original)

        for key, value in counts.items():
            totals[key] += value

        # Empty and long tickets are rejected before any model call.
        if not original.strip():
            initial_status = "skipped_empty"
        elif mock_llm.est_tokens(original) > 400:
            initial_status = "flagged_long"
        else:
            initial_status = None

        prepared.append({
            "ticket_id": str(row["ticket_id"]),
            "customer_id": str(row["customer_id"]),
            "order_id": (
                "" if pd.isna(row.get("order_id", ""))
                else str(row.get("order_id", ""))
            ),
            "text": redacted,
            "original_text": original,
            "true_category": (
                "" if pd.isna(row.get("true_category", ""))
                else str(row.get("true_category", "")).strip().lower()
            ),
            "initial_status": initial_status,
        })

    return prepared, totals


# ------------------------------------------------------------
# Retry wrapper
# At most 3 retries after the initial attempt.
# Delay for retry n: BASE_DELAY * 2**n * uniform(0.9, 1.15).
# ------------------------------------------------------------

def chat_with_retry(messages, ticket_id, token_total, token_budget):
    for attempt in range(4):
        # Check the budget immediately before every model call attempt.
        if token_total[0] >= token_budget:
            return None, "budget_exhausted"

        try:
            response = mock_llm.chat(messages)
            return response, None

        except mock_llm.TransientLLMError:
            if attempt >= 3:
                raise

            wait = (
                BASE_DELAY
                * (2 ** attempt)
                * random.uniform(0.9, 1.15)
            )

            line = (
                f"RETRY ticket_id={ticket_id} "
                f"retry_number={attempt} wait_seconds={wait:.4f}"
            )
            print(line)
            RETRY_LOG.append(line)
            time.sleep(wait)

    raise RuntimeError(f"Unexpected retry exit for ticket {ticket_id}")


# ------------------------------------------------------------
# Reply parsing
# JSON with an unsupported category is invalid output.
# Keyword fallback is used only when JSON parsing fails.
# ------------------------------------------------------------

def parse_reply(content):
    content = content or ""

    try:
        parsed = json.loads(content)

    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", content.lower()))
        matches = words & CATEGORIES

        if len(matches) == 1:
            return next(iter(matches)), "classified_keyword"

        return "", "flagged_invalid_output"

    if not isinstance(parsed, dict):
        return "", "flagged_invalid_output"

    category = parsed.get("category")

    if isinstance(category, str) and category.strip().lower() in CATEGORIES:
        return category.strip().lower(), "classified_json"

    return "", "flagged_invalid_output"


# ------------------------------------------------------------
# Customer segments lookup
# Missing customer IDs are treated as non-premium.
# ------------------------------------------------------------

def load_segment_lookup():
    segments = pd.read_csv(SEGMENTS_PATH, dtype={"customer_id": str})

    required = {"customer_id", "segment_name", "is_premium"}
    missing = required - set(segments.columns)

    if missing:
        raise ValueError(
            f"{SEGMENTS_PATH} is missing required columns: {sorted(missing)}"
        )

    lookup = {}

    for _, row in segments.iterrows():
        customer_id = str(row["customer_id"])

        premium_value = row["is_premium"]
        is_premium = (
            str(premium_value).strip().lower() in {"true", "1", "yes"}
            if not pd.isna(premium_value)
            else False
        )

        segment_name = (
            "" if pd.isna(row["segment_name"])
            else str(row["segment_name"])
        )

        lookup[customer_id] = {
            "segment_name": segment_name,
            "is_premium": is_premium,
        }

    return lookup


# ------------------------------------------------------------
# Routing
# ------------------------------------------------------------

HUMAN_REVIEW_STATUSES = {
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted",
}


def route_ticket(status, category, is_premium):
    if status in HUMAN_REVIEW_STATUSES:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if is_premium:
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


# ------------------------------------------------------------
# One independent pipeline run
# Process tickets in their original file order.
# User message content is exactly the redacted ticket text.
# ------------------------------------------------------------

def run_pipeline(prepared, segment_lookup, system_prompt, token_budget):
    rows = []
    token_total = [0]
    model_call_attempts = 0
    successful_model_responses = 0

    for ticket in prepared:
        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]
        segment = segment_lookup.get(
            customer_id,
            {"segment_name": "", "is_premium": False},
        )

        status = ticket["initial_status"]
        category = ""
        tokens_used = 0

        # These tickets must never reach the model.
        if status is None:
            # Recheck redaction immediately before constructing the prompt.
            user_content = ticket["text"]
            assert EMAIL_RE.search(user_content) is None
            assert PHONE_RE.search(user_content) is None
            assert ORDER_ID_RE.search(user_content) is None

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            # Assert there are exactly two messages and that the user
            # message is the redacted ticket text only.
            assert len(messages) == 2
            assert messages[-1]["content"] == ticket["text"]

            if token_total[0] >= token_budget:
                status = "budget_exhausted"
            else:
                # Count each attempt, including failed attempts.
                for attempt in range(4):
                    if token_total[0] >= token_budget:
                        status = "budget_exhausted"
                        break

                    model_call_attempts += 1

                    try:
                        response = mock_llm.chat(messages)
                        successful_model_responses += 1
                        tokens_used = int(response.usage.total_tokens)
                        token_total[0] += tokens_used

                        category, status = parse_reply(
                            response.choices[0].message.content
                        )
                        break

                    except mock_llm.TransientLLMError:
                        if attempt >= 3:
                            raise

                        wait = (
                            BASE_DELAY
                            * (2 ** attempt)
                            * random.uniform(0.9, 1.15)
                        )

                        line = (
                            f"RETRY ticket_id={ticket_id} "
                            f"retry_number={attempt} wait_seconds={wait:.4f}"
                        )
                        print(line)
                        RETRY_LOG.append(line)
                        time.sleep(wait)

        route, priority = route_ticket(
            status,
            category,
            segment["is_premium"],
        )

        rows.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment["segment_name"],
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used,
            "true_category": ticket["true_category"],
        })

    return {
        "rows": rows,
        "total_tokens": token_total[0],
        "model_call_attempts": model_call_attempts,
        "successful_model_responses": successful_model_responses,
    }


# ------------------------------------------------------------
# Reporting and evaluation
# ------------------------------------------------------------

def classified_rows(rows):
    return [
        row for row in rows
        if row["status"] in {"classified_json", "classified_keyword"}
    ]


def report_run(name, result):
    rows = result["rows"]
    classified = classified_rows(rows)

    correct = sum(
        row["category"] == row["true_category"]
        for row in classified
    )

    accuracy = correct / len(classified) if classified else 0.0
    tokens_per_classified = (
        result["total_tokens"] / len(classified)
        if classified else 0.0
    )

    print(f"\n{'=' * 65}")
    print(name)
    print(f"{'=' * 65}")
    print(f"Tickets in input:             {len(rows)}")
    print(f"Classified tickets:            {len(classified)}")
    print(f"Correct classifications:       {correct}")
    print(f"Accuracy on classified:        {accuracy:.4%}")
    print(f"Total response tokens:         {result['total_tokens']}")
    print(f"Tokens per classified ticket:   {tokens_per_classified:.2f}")
    print(f"Model call attempts:            {result['model_call_attempts']}")
    print(f"Successful responses:           {result['successful_model_responses']}")

    counts = pd.Series([row["status"] for row in rows]).value_counts()
    print("\nStatus counts:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")

    if name == "ZERO-SHOT UNBUDGETED":
        actual = [row["true_category"] for row in classified]
        predicted = [row["category"] for row in classified]

        confusion = pd.crosstab(
            pd.Categorical(actual, categories=sorted(CATEGORIES)),
            pd.Categorical(predicted, categories=sorted(CATEGORIES)),
            rownames=["Actual"],
            colnames=["Predicted"],
            dropna=False,
        )

        print("\nConfusion table:")
        print(confusion.to_string())

        print("\nMisclassifications:")
        mistakes = [
            row for row in classified
            if row["category"] != row["true_category"]
        ]
        if mistakes:
            error_table = pd.crosstab(
                pd.Series(
                    [row["true_category"] for row in mistakes],
                    name="Actual",
                ),
                pd.Series(
                    [row["category"] for row in mistakes],
                    name="Predicted",
                ),
            )
            print(error_table.to_string())
        else:
            print("None")

    return {
        "classified": len(classified),
        "correct": correct,
        "accuracy": accuracy,
        "total_tokens": result["total_tokens"],
        "tokens_per_classified": tokens_per_classified,
    }


def save_final_csv(rows, path):
    columns = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]

    df = pd.DataFrame(rows)

    # Drop evaluation-only fields from the required output.
    df = df[columns]
    df.to_csv(path, index=False)

    # Read back and verify the required schema and row count.
    check = pd.read_csv(path, dtype={"ticket_id": str, "customer_id": str})
    assert list(check.columns) == columns
    assert len(check) == 164

    return check


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TICKETS_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {TICKETS_PATH}")

    if not SEGMENTS_PATH.exists():
        raise FileNotFoundError(f"Missing Part B output: {SEGMENTS_PATH}")

    tickets = pd.read_csv(
        TICKETS_PATH,
        dtype={"ticket_id": str, "customer_id": str, "order_id": str},
        keep_default_na=True,
    )

    required_columns = {
        "ticket_id", "customer_id", "order_id", "text", "true_category"
    }
    missing = required_columns - set(tickets.columns)

    if missing:
        raise ValueError(f"tickets.csv is missing columns: {sorted(missing)}")

    assert len(tickets) == 164, f"Expected 164 tickets, found {len(tickets)}"

    prepared, pii_counts = prepare_tickets(tickets)
    segment_lookup = load_segment_lookup()

    status_before_calls = pd.Series(
        [ticket["initial_status"] for ticket in prepared]
    ).value_counts(dropna=False)

    empty_count = sum(
        ticket["initial_status"] == "skipped_empty"
        for ticket in prepared
    )
    long_count = sum(
        ticket["initial_status"] == "flagged_long"
        for ticket in prepared
    )

    print("PII redaction counts:")
    for key, count in pii_counts.items():
        print(f"  {key}: {count}")
    print(f"  total: {sum(pii_counts.values())}")

    print("\nPre-call checks:")
    print(f"  Empty tickets: {empty_count}")
    print(f"  Long tickets:  {long_count}")

    assert empty_count == 4, f"Expected 4 empty tickets, found {empty_count}"
    assert long_count == 4, f"Expected 4 long tickets, found {long_count}"
    assert pii_counts == {
        "emails": 55,
        "phones": 91,
        "order_ids": 132,
    }, f"Unexpected redaction counts: {pii_counts}"

    # --------------------------------------------------------
    # Run 1: Zero-shot, no budget restriction.
    # --------------------------------------------------------

    print("\nStarting zero-shot unbudgeted run...")
    random.seed(42)
    mock_llm.reset()
    RETRY_LOG.clear()

    zero_result = run_pipeline(
        prepared,
        segment_lookup,
        ZERO_SHOT_PROMPT,
        FINAL_TOKEN_BUDGET,
    )

    zero_metrics = report_run("ZERO-SHOT UNBUDGETED", zero_result)

    assert zero_metrics["classified"] == 147
    assert abs(zero_metrics["accuracy"] - 0.857) <= 0.005

    # --------------------------------------------------------
    # Run 2: Few-shot, budget 100000.
    # This is the final triage_results.csv output.
    # --------------------------------------------------------

    print("\nStarting few-shot run with budget 100000...")
    random.seed(42)
    mock_llm.reset()
    RETRY_LOG.clear()

    few_result = run_pipeline(
        prepared,
        segment_lookup,
        FEW_SHOT_PROMPT,
        FINAL_TOKEN_BUDGET,
    )

    few_metrics = report_run("FEW-SHOT UNBUDGETED", few_result)

    assert few_metrics["classified"] == 147
    assert abs(few_metrics["accuracy"] - 1.0) <= 0.005

    final_path = OUTPUT_DIR / "triage_results.csv"
    final_df = save_final_csv(few_result["rows"], final_path)

    # Expected few-shot status counts.
    few_status_counts = pd.Series(
        [row["status"] for row in few_result["rows"]]
    ).value_counts().to_dict()

    assert few_status_counts.get("classified_json", 0) == 135
    assert few_status_counts.get("classified_keyword", 0) == 12
    assert few_status_counts.get("flagged_invalid_output", 0) == 9
    assert few_status_counts.get("skipped_empty", 0) == 4
    assert few_status_counts.get("flagged_long", 0) == 4

    # --------------------------------------------------------
    # Run 3: Zero-shot, token budget demonstration.
    # --------------------------------------------------------

    print("\nStarting zero-shot budget demonstration with budget 6000...")
    random.seed(42)
    mock_llm.reset()
    RETRY_LOG.clear()

    budget_result = run_pipeline(
        prepared,
        segment_lookup,
        ZERO_SHOT_PROMPT,
        TOKEN_BUDGET,
    )

    report_run("ZERO-SHOT BUDGET DEMO", budget_result)

    budget_path = OUTPUT_DIR / "triage_budget_demo.csv"
    budget_df = save_final_csv(budget_result["rows"], budget_path)

    exhausted_count = int(
        (budget_df["status"] == "budget_exhausted").sum()
    )

    # The budget demo must exhaust the budget at least once.
    assert exhausted_count > 0, (
        "No tickets exhausted the 6000-token budget. "
        "Try lowering TOKEN_BUDGET to 5000 and rerun."
    )

    # --------------------------------------------------------
    # Routing and safety assertions
    # --------------------------------------------------------

    for row in few_result["rows"]:
        if row["status"] in HUMAN_REVIEW_STATUSES:
            assert row["route"] == "human_review"
            assert row["priority"] == "NORMAL"
        elif row["category"] == "fraud":
            assert row["route"] == "security_team"
            assert row["priority"] == "HIGH"

    assert len(final_df) == 164
    assert len(budget_df) == 164

    print("\n" + "=" * 65)
    print("PART C ACCEPTANCE CHECKS")
    print("=" * 65)
    print(f"Final output:       {final_path}")
    print(f"Budget output:      {budget_path}")
    print(f"Final CSV rows:     {len(final_df)}")
    print(f"Budget CSV rows:    {len(budget_df)}")
    print(f"Budget exhausted:   {exhausted_count}")
    print(f"Zero-shot accuracy: {zero_metrics['accuracy']:.2%}")
    print(f"Few-shot accuracy:  {few_metrics['accuracy']:.2%}")
    print(f"Zero-shot tokens:   {zero_metrics['total_tokens']}")
    print(f"Few-shot tokens:    {few_metrics['total_tokens']}")
    print(
        "Extra few-shot tokens: "
        f"{few_metrics['total_tokens'] - zero_metrics['total_tokens']}"
    )
    print("ALL PART C ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
