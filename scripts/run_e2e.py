"""Run labeled cases through a temporary local service and the real SubQ API."""

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from volte_face.app import AnalyzeRequest, AnalyzeResponse, Probability, Text, Verdict


DEFAULT_CASES = Path(__file__).resolve().parents[1] / "tests/fixtures/intent_cases.json"
CASCADE_CASES = DEFAULT_CASES.with_name("cascade_cases.json")


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Text
    request: AnalyzeRequest
    expected_shifted: bool = Field(strict=True)
    rationale: Text
    expected_boundary_status: Optional[Literal["evaluated", "not_evaluated"]] = None
    expected_boundary_verdict: Optional[Verdict] = None

    @model_validator(mode="after")
    def check_boundary_expectation(self):
        if self.expected_boundary_verdict is not None:
            if self.expected_boundary_status != "evaluated":
                raise ValueError("A boundary verdict requires expected status evaluated.")
        return self


def load_cases(path: Path) -> list[Case]:
    cases = TypeAdapter(list[Case]).validate_json(path.read_bytes())
    if not cases:
        raise ValueError("Cases file must contain at least one case.")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Case IDs must be unique.")
    return cases


@contextmanager
def local_service():
    # Pass the bound socket to the child to avoid a port-selection race.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        base_url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "volte_face.app:create_app", "--factory",
                "--fd", str(listener.fileno()),
                "--log-level", "warning", "--no-access-log",
            ],
            pass_fds=(listener.fileno(),),
        )
        try:
            deadline = time.monotonic() + 15
            with httpx.Client(base_url=base_url, timeout=0.5, trust_env=False) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Local service exited during startup ({process.returncode})."
                        )
                    try:
                        response = client.get("/health")
                    except httpx.TransportError:
                        response = None
                    if response is not None:
                        response.raise_for_status()
                        if response.json() != {"status": "ok", "configured": True}:
                            raise RuntimeError("Local service is not configured.")
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Local service did not become ready in 15 seconds.")
                    time.sleep(0.1)
            yield base_url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def run_cases(
    client: httpx.Client, cases: list[Case], threshold: Optional[float] = None
) -> int:
    matched = mismatched = errors = 0
    total_input_tokens = 0
    boundary_checks = 0
    started = time.monotonic()
    for case in cases:
        body = case.request.model_dump(exclude_none=True)
        if threshold is not None:
            body["threshold"] = threshold
        case_started = time.monotonic()
        try:
            response = client.post("/v1/analyze", json=body)
            response.raise_for_status()
            result = AnalyzeResponse.model_validate_json(response.content)
            if result.threshold != body["threshold"]:
                raise ValueError("Service returned a different threshold.")
            if result.shifted != (result.shift_probability >= result.threshold):
                raise ValueError("Service verdict is inconsistent with its probability.")
            should_check = result.shifted or case.request.force_boundary_check
            if (result.boundary_check.status == "evaluated") != should_check:
                raise ValueError("Service did not follow cascade routing.")
        except httpx.HTTPStatusError as error:
            errors += 1
            # Surface only the service's known error codes, never raw response bodies.
            upstream_status = None
            stage = None
            try:
                detail = error.response.json()["detail"]
                if isinstance(detail, dict):
                    upstream_status = detail.get("upstream_status")
                    stage = detail.get("stage")
            except (ValueError, KeyError, TypeError):
                pass
            suffix = (
                f" (SubQ HTTP {upstream_status})"
                if type(upstream_status) is int else ""
            )
            stage_suffix = f" stage={stage}" if stage in ("shift", "boundary") else ""
            print(
                f"ERROR {case.id}: service HTTP {error.response.status_code}"
                f"{suffix}{stage_suffix}"
            )
            continue
        except (httpx.RequestError, ValidationError, ValueError) as error:
            errors += 1
            print(f"ERROR {case.id}: {type(error).__name__}")
            continue

        elapsed = time.monotonic() - case_started
        total_input_tokens += result.usage.input_tokens
        boundary = result.boundary_check
        boundary_checks += boundary.status == "evaluated"
        correct = result.shifted == case.expected_shifted
        if case.expected_boundary_status is not None:
            correct = correct and boundary.status == case.expected_boundary_status
        if case.expected_boundary_verdict is not None:
            correct = correct and (
                boundary.status == "evaluated"
                and boundary.verdict == case.expected_boundary_verdict
            )
        if correct:
            matched += 1
        else:
            mismatched += 1
        print(
            f"{'PASS' if correct else 'FAIL'} {case.id}: "
            f"expected={'yes' if case.expected_shifted else 'no'} "
            f"actual={'yes' if result.shifted else 'no'} "
            f"p={result.shift_probability:.4f} threshold={result.threshold:g} "
            f"boundary={boundary.verdict if boundary.status == 'evaluated' else boundary.status} "
            f"model={result.model} elapsed={elapsed:.2f}s"
        )
        if not correct:
            if case.expected_boundary_status is not None:
                print(
                    f"  Expected boundary: {case.expected_boundary_status}"
                    f" / {case.expected_boundary_verdict or 'no verdict'}"
                )
            print(f"  Expected: {case.rationale}")

    print(
        f"\n{matched}/{len(cases)} matched ({matched / len(cases):.1%}); "
        f"{mismatched} mismatches; {errors} errors; "
        f"{boundary_checks} boundary checks; "
        f"{total_input_tokens} input tokens from valid responses; "
        f"{time.monotonic() - started:.2f}s total."
    )
    return 1 if mismatched or errors else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, nargs="+", default=[DEFAULT_CASES, CASCADE_CASES],
        help="Case files (default: intent and cascade fixtures).",
    )
    parser.add_argument(
        "--threshold", type=float,
        help="Override the decision threshold for every case (0 through 1).",
    )
    args = parser.parse_args(argv)
    if args.threshold is not None:
        try:
            TypeAdapter(Probability).validate_python(args.threshold)
        except ValidationError:
            parser.error("--threshold must be a finite number from 0 through 1")
    if not os.environ.get("SUBQ_API_KEY", "").strip():
        print("ERROR: Set SUBQ_API_KEY before running live tests.", file=sys.stderr)
        return 2
    try:
        cases = [case for path in args.cases for case in load_cases(path)]
        if len({case.id for case in cases}) != len(cases):
            raise ValueError("Case IDs must be unique across files.")
    except (OSError, ValidationError, ValueError) as error:
        print(f"ERROR: Cannot load cases ({type(error).__name__}).", file=sys.stderr)
        return 2

    print(
        f"Running {len(cases)} live cases with "
        f"{os.environ.get('SUBQ_MODEL', 'subq-snap-preview')}. "
        "This sends fixture text to SubQ and consumes API credits.",
        flush=True,
    )
    try:
        with local_service() as base_url:
            with httpx.Client(
                base_url=base_url, timeout=120, trust_env=False
            ) as client:
                return run_cases(client, cases, args.threshold)
    except (OSError, RuntimeError, httpx.HTTPError, ValueError) as error:
        print(
            f"ERROR: Local service startup/run failed ({type(error).__name__}): {error}",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted; local service stopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
