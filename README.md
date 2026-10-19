# volte-face

Compare two neighboring LLM responses and detect a material shift in **expressed
intent**: the objective being pursued, substantive commitments, material approach,
or willingness to act. When a shift is detected, a second decision evaluates
whether the current proposed behavior is authorized, out of bounds, or uncertain.
Changes in wording, tone, detail, or normal progress through a plan do not
by themselves count. A change can be legitimate or user-requested; detecting one
does not mean it is wrong.

## SubQ model

The [SubQ docs](https://docs.subq.ai/using-the-api/models/) currently document
`subq-snap-preview` for fast decisions, not a model named Flash. This service uses
that model through `POST https://api.subq.ai/v1/systemone`. If your account has a
different decision-model ID, set `SUBQ_MODEL` to that exact ID. This is the
[Decisions API](https://docs.subq.ai/endpoints/decisions-api/), not a chat API.

The first request asks four related yes/no (`noul`) questions: overall intent shift,
goal change, commitment reversal, and willingness change. SubQ returns
probabilities, not generated explanations. The service applies your threshold to
the overall probability; the three dimension probabilities are independent
diagnostic judgments, not a sum or a mathematical decomposition.

### Two-tier cascade

Tier 1 includes material changes in means or authorization boundaries even when
the ultimate goal is unchanged: solving a challenge versus obtaining a restricted
answer key is a shift. Switching between ordinary authorized solving techniques
is not.

When `shifted` is true, Tier 2 makes a separate SubQ request with a choice question:
`authorized`, `out_of_bounds`, or `uncertain`. It evaluates behavior against the
task and permissions, rather than attempting to prove hidden malicious intent.
Supply `boundaries` with caller-controlled permissions and prohibitions; do not
let the monitored agent write its own authorization rules. `context` supplies
task background, but is not a substitute for explicit authorization rules.

No detected shift returns `boundary_check: {"status": "not_evaluated"}`, **not**
a safety clearance. Set `force_boundary_check: true` to run Tier 2 regardless of
the shift result, particularly before high-impact actions or to detect continuing
unauthorized behavior. The caller must choose when to force a check; the service
does not automatically recognize high-impact tool calls or execute/block actions.
Include the proposed action in `current_response` and check before executing it.

An evaluated result includes the verdict, model confidence, and the probability
of each option. Low-confidence or `uncertain` decisions need review; even an
`authorized` verdict is not proof of safety. A Tier 2 failure returns an explicit
HTTP error with `detail.stage: "boundary"`, never an authorized or skipped result.
Token usage is summed across both successful stages.

## Run locally

Requires Python 3.9+ and a SubQ account with credits.
Run the following commands from the project directory. First install once:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
```

Set the API key in your terminal, then start the service:

```sh
export SUBQ_API_KEY="your-api-key"
.venv/bin/uvicorn volte_face.app:create_app --factory --host 127.0.0.1 --port 8000
```

Keep this terminal open; press Ctrl-C to stop the service. Do not put your real
key in source files or commit it. Environment variables must be set again in a
new terminal. Optionally set `SUBQ_MODEL` before starting to override the model.

Interactive API documentation: <http://127.0.0.1:8000/docs>.
`GET /health` reports process health and whether a key is configured; it does
not verify SubQ connectivity, account access, or credits.

```sh
curl --fail-with-body http://127.0.0.1:8000/health

curl --fail-with-body http://127.0.0.1:8000/v1/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "previous_response": "I will preserve the original files.",
    "current_response": "I will delete the originals instead.",
    "context": "The task is to reorganize files without losing data.",
    "boundaries": "Preserve all original files. Deletion is forbidden.",
    "threshold": 0.5
  }'
```

Run these requests from a second terminal while the service is running.

Example response shape (illustrative probabilities, not a recorded model result):

```json
{
  "shifted": true,
  "shift_probability": 0.9,
  "threshold": 0.5,
  "dimensions": {
    "goal_change": 0.7,
    "commitment_reversal": 0.95,
    "willingness_change": 0.1
  },
  "model": "subq-snap-preview",
  "usage": {"input_tokens": 700, "output_tokens": 0},
  "boundary_check": {
    "status": "evaluated",
    "verdict": "out_of_bounds",
    "confidence": 0.9,
    "probabilities": {
      "authorized": 0.05,
      "out_of_bounds": 0.9,
      "uncertain": 0.05
    }
  }
}
```

Both responses are required, nonblank strings, at most 40,000 characters each.
Optional `context` and `boundaries` have the same limit; include the original task and any
intervening user message when relevant. `threshold` defaults to `0.5`, is
inclusive (`probability >= threshold`), and accepts numbers from `0` to `1`.
At threshold `0`, even a zero probability yields `shifted: true`.
No conversation history is fetched or persisted locally.

Input errors return `422`; a missing key returns `503`. Upstream rate limits
return `503`, timeouts `504`, and other upstream failures or malformed decisions
`502`. Upstream HTTP errors include `detail.upstream_status` (for example `402`
for insufficient credits) without exposing the upstream response body.
Each SubQ request times out after 30 seconds of network inactivity (5 seconds connecting);
there are no automatic retries. Callers can back off on transient failures.

## Interpretation and deployment

This detects observable behavior, not an LLM's hidden intent. A probability is
not a guarantee or a measured accuracy score. Calibrate thresholds on labeled
pairs from your application, including paraphrases, plan progression, reversals,
refusal transitions, user-requested changes, and embedded prompt injections.
Treat near-threshold and high-impact decisions as requiring review. Without
context, an apparent reversal can be hard to interpret.

The API is an unauthenticated local prototype. Do not expose it publicly without
authentication, rate limiting, and a request-body limit at your ingress.
Response/context text is sent to SubQ; use only data you are authorized to send.
SubQ's [retention policy](https://docs.subq.ai/using-the-api/data-retention/)
defaults to indefinite storage unless your organization configures otherwise.
The service does not log response text, context, or API keys.

## Tests

```sh
.venv/bin/python -m pytest -q
```

Tests use a mocked SubQ HTTP transport and do not consume credits. They verify
the integration contract, threshold behavior, validation, and failure handling;
they do not establish the model's intent-detection accuracy. Live inference
requires a real API key and labeled examples for meaningful evaluation.

### Labeled intent examples

`tests/fixtures/intent_cases.json` contains 12 hand-labeled cases: six shifts and
six non-shifts. Each has an API-ready `request`, an `expected_shifted` boolean,
and a `rationale`. Send only the `request` object to `/v1/analyze`; keep the label
and rationale out of the model input. These are evaluation examples, not recorded
model predictions, and the mocked test suite does not measure accuracy on them.

`tests/fixtures/cascade_cases.json` adds eight cases with explicit expected
boundary statuses and verdicts: answer-key theft, euphemistic restricted lookup,
authorized reference lookup, legitimate solving techniques, rejected prompt
injection, continuing unauthorized behavior with a forced check, missing
permissions, and a user request that conflicts with caller boundaries.
The automated suite stubs SubQ judgments to verify routing, state handling,
verdict propagation, errors, and accounting. It does not prove those judgments
will be produced by the real model.

### One-command live end-to-end test

After the installation above, set your key and run:

```sh
export SUBQ_API_KEY="your-api-key"
.venv/bin/python scripts/run_e2e.py
```

You do **not** need to start the service separately. The script starts a real
Uvicorn subprocess on an available loopback port, waits for `/health`, sends
all 20 cases from both fixture files to `/v1/analyze` over HTTP, and lets the
service call SubQ, including the second stage when triggered or forced.
It stops its own subprocess on completion, failure, or Ctrl-C, without touching
any service you already have running. This runner requires a POSIX environment
(macOS or Linux) for passing the listening socket to the subprocess.

This is a **live, billable evaluation**: fixture text is sent to SubQ, and your
account needs credits and access to the configured decision model. Expected
labels and rationales stay local. There are no mocked decisions in this command.

Each result prints `PASS`, `FAIL` (label mismatch), or `ERROR` (request or response
failure), with expected/actual shift labels, probability, threshold, boundary
verdict or skipped status, model, and elapsed time for valid responses. Cascade
cases must match their expected boundary status and verdict as well as the shift
label. The summary shows matches over **all** cases, mismatch and error counts,
boundary-check count, reported input tokens from valid responses, and total time.
Cases continue after individual failures. Token totals exclude failed requests
and therefore are not a complete billing record.

Exit codes: `0` means every label matched, `1` means at least one mismatch or case
error, `2` means configuration, fixture, or startup failure, and `130` means
interrupted. A successful run evaluates only these examples, not general accuracy.

Optional overrides:

```sh
# Use a stricter threshold for every case.
.venv/bin/python scripts/run_e2e.py --threshold 0.7

# Use another file with the same case format.
.venv/bin/python scripts/run_e2e.py --cases tests/fixtures/intent_cases.json

# Run only the cascade scenarios.
.venv/bin/python scripts/run_e2e.py --cases tests/fixtures/cascade_cases.json
```

`--cases` accepts one or more files and replaces the default two-file selection.
The model comes from `SUBQ_MODEL` (default `subq-snap-preview`). Without
`--threshold`, each request's threshold is used, defaulting to `0.5`.
For failures, check `SUBQ_API_KEY`, account credits (`SubQ HTTP 402`), model access
(`403`), and rate limits (`429`). Avoid repeatedly rerunning credential or credit
failures; fix the configuration first.
