import json

import httpx
import pytest
from fastapi.testclient import TestClient

from scripts.run_e2e import (
    CASCADE_CASES, DEFAULT_CASES, load_cases, local_service, main, run_cases,
)
from volte_face.app import create_app


def test_all_fixtures_through_service_without_sending_labels(capsys):
    cases = load_cases(DEFAULT_CASES) + load_cases(CASCADE_CASES)
    expected = {
        case.request.current_response: case for case in cases
    }
    seen = []
    boundary_seen = []

    def subq(request):
        payload = json.loads(request.content)
        state = payload["state"]
        assert set(state) <= {
            "previous_response", "current_response", "context", "boundaries",
        }
        case = expected[state["current_response"]]
        if "boundary" in payload["questions"]:
            boundary_seen.append(case.id)
            verdict = case.expected_boundary_verdict or "uncertain"
            return httpx.Response(200, json={
                "model": payload["model"],
                "answers": {"boundary": {
                    "type": "choice", "choice": verdict, "confidence": 0.9,
                    "probabilities": {
                        key: 0.9 if key == verdict else 0.05
                        for key in ("authorized", "out_of_bounds", "uncertain")
                    },
                }},
                "usage": {"input_tokens": 50, "output_tokens": 0},
            })
        seen.append(state["previous_response"])
        probability = 0.9 if case.expected_shifted else 0.1
        return httpx.Response(200, json={
            "model": payload["model"],
            "answers": {
                key: {"type": "noul", "noul": probability}
                for key in payload["questions"]
            },
            "usage": {"input_tokens": 100, "output_tokens": 0},
        })

    with TestClient(create_app(
        api_key="fake-key", transport=httpx.MockTransport(subq)
    )) as client:
        assert run_cases(client, cases) == 0
    assert len(seen) == len(cases) == 20
    assert len(boundary_seen) == 13
    output = capsys.readouterr().out
    assert "20/20 matched (100.0%)" in output
    assert "13 boundary checks" in output
    assert "2650 input tokens" in output


def test_mismatches_and_errors_continue_and_fail_run(capsys):
    cases = load_cases(DEFAULT_CASES)[:4]
    calls = []

    def service(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(200, json={
                "shifted": False, "shift_probability": 0.1, "threshold": 0.6,
                "dimensions": {
                    "goal_change": 0.1,
                    "commitment_reversal": 0.1,
                    "willingness_change": 0.1,
                },
                "model": "test", "usage": {"input_tokens": 100, "output_tokens": 0},
                "boundary_check": {"status": "not_evaluated"},
            })
        if len(calls) == 2:
            return httpx.Response(502, json={
                "detail": {"upstream_status": 402, "message": "private data"}
            })
        if len(calls) == 3:
            return httpx.Response(200, text="private malformed response")
        raise httpx.ReadTimeout("private connection detail", request=request)

    with httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(service)
    ) as client:
        assert run_cases(client, cases, threshold=0.6) == 1
    assert len(calls) == 4
    assert all(call["threshold"] == 0.6 for call in calls)
    output = capsys.readouterr().out
    assert "1 mismatches; 3 errors" in output
    assert "SubQ HTTP 402" in output
    assert "private" not in output


def test_real_local_service_startup_and_shutdown(monkeypatch):
    monkeypatch.setenv("SUBQ_API_KEY", "fake-key")
    with local_service() as url:
        with httpx.Client(base_url=url, trust_env=False) as client:
            assert client.get("/health").json()["configured"] is True
    with httpx.Client(timeout=0.5, trust_env=False) as client:
        with pytest.raises(httpx.ConnectError):
            client.get(url + "/health")


def test_service_stops_when_run_raises(monkeypatch):
    monkeypatch.setenv("SUBQ_API_KEY", "fake-key")
    with pytest.raises(RuntimeError, match="run failed"):
        with local_service() as url:
            raise RuntimeError("run failed")
    with httpx.Client(timeout=0.5, trust_env=False) as client:
        with pytest.raises(httpx.ConnectError):
            client.get(url + "/health")


def test_missing_key_fails_before_starting_service(monkeypatch, capsys):
    monkeypatch.delenv("SUBQ_API_KEY", raising=False)
    assert main([]) == 2
    assert "Set SUBQ_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("threshold", ["nan", "inf", "-0.1", "1.1"])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(SystemExit) as error:
        main([f"--threshold={threshold}"])
    assert error.value.code == 2


@pytest.mark.parametrize("content", ["[]", "[{}]"])
def test_invalid_cases_rejected(tmp_path, content):
    path = tmp_path / "cases.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_cases(path)


def test_duplicate_case_ids_rejected(tmp_path):
    case = json.loads(DEFAULT_CASES.read_text())[0]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case, case]))
    with pytest.raises(ValueError, match="unique"):
        load_cases(path)


def test_runner_fails_when_boundary_verdict_is_wrong(capsys):
    case = load_cases(CASCADE_CASES)[0]
    result = {
        "shifted": True, "shift_probability": 0.9, "threshold": 0.5,
        "dimensions": {
            "goal_change": 0.1, "commitment_reversal": 0.9, "willingness_change": 0.1,
        },
        "model": "test", "usage": {"input_tokens": 100, "output_tokens": 0},
        "boundary_check": {
            "status": "evaluated", "verdict": "authorized", "confidence": 0.9,
            "probabilities": {"authorized": 0.9, "out_of_bounds": 0.05, "uncertain": 0.05},
        },
    }
    with httpx.Client(
        base_url="http://test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=result)),
    ) as client:
        assert run_cases(client, [case]) == 1
    output = capsys.readouterr().out
    assert "FAIL solving-to-answer-key-theft" in output
    assert "1 mismatches; 0 errors" in output


def test_runner_rejects_skipped_cascade_when_shifted(capsys):
    case = load_cases(CASCADE_CASES)[0]
    result = {
        "shifted": True, "shift_probability": 0.9, "threshold": 0.5,
        "dimensions": {
            "goal_change": 0.1, "commitment_reversal": 0.9, "willingness_change": 0.1,
        },
        "model": "test", "usage": {"input_tokens": 100, "output_tokens": 0},
        "boundary_check": {"status": "not_evaluated"},
    }
    with httpx.Client(
        base_url="http://test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=result)),
    ) as client:
        assert run_cases(client, [case]) == 1
    assert "0 mismatches; 1 errors" in capsys.readouterr().out
