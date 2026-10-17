"""POST /api/v1/jobs/{job_id}/feedback, GET /api/v1/config и таблица feedback.

Сбор отзывов выключен по умолчанию (FEEDBACK_ENABLED): раздел 10 ТЗ запрещает использовать
тестовые видео жюри для обучения. Отзывы — только для анализа ошибок на собственном датасете.
"""

from __future__ import annotations

import sys
from contextlib import closing
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from shared import db  # noqa: E402
from shared.schemas import AnalysisResult, EvidenceFrame, JobStatus, RuleRef, Verdict, Violation  # noqa: E402

KEY = "test-key-0123456789"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["1"])
def client(request, tmp_path, monkeypatch):
    """TestClient с задачами aaaa… (done, 1 нарушение) и bbbb… (processing).

    FEEDBACK_ENABLED по умолчанию "1"; тесты выключенного режима параметризуют фикстуру косвенно.
    """
    if request.param is None:
        monkeypatch.delenv("FEEDBACK_ENABLED", raising=False)
    else:
        monkeypatch.setenv("FEEDBACK_ENABLED", request.param)
    url = f"sqlite:///{tmp_path / 'app.db'}"
    monkeypatch.setenv("API_KEY", KEY)
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.syspath_prepend(str(ROOT / "backend"))
    sys.modules.pop("app.main", None)
    from app.main import app

    with TestClient(app) as c:
        with closing(db.connect(url)) as conn:
            db.create_job(conn, "aaaaaaaaaaaa", "aaaaaaaaaaaa/input.mov")
            db.create_job(conn, "bbbbbbbbbbbb", "bbbbbbbbbbbb/input.mov")
            v = Violation(type="Тест", timecode_sec=5.0, description="d",
                          evidence_frames=[EvidenceFrame(timestamp_sec=5.0, image_base64="AA==")],
                          rule=RuleRef(id="R02", doc="d", url="u", clause="c"))
            res = AnalysisResult(job_id="aaaaaaaaaaaa", status=JobStatus.DONE,
                                 verdict=Verdict.VIOLATION, violations=[v])
            conn.execute("UPDATE jobs SET started_at = 1 WHERE job_id = 'aaaaaaaaaaaa'")
            assert db.save_result(conn, res)
        c.db_url = url  # type: ignore[attr-defined]
        yield c


def _post(c, job_id, body, key=KEY):
    return c.post(f"/api/v1/jobs/{job_id}/feedback", json=body, headers={"X-API-Key": key} if key else {})


def test_feedback_saved(client):
    r = _post(client, "aaaaaaaaaaaa", {"violation_index": 0, "label": "rejected", "comment": "это клиент"})
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    with closing(db.connect(client.db_url)) as conn:
        rows = db.list_feedback(conn)
    assert len(rows) == 1
    assert (rows[0].rule_id, rows[0].timecode_sec, rows[0].label, rows[0].comment) == ("R02", 5.0, "rejected", "это клиент")


def test_feedback_errors(client):
    assert _post(client, "aaaaaaaaaaaa", {"violation_index": 0, "label": "confirmed"}, key=None).status_code == 401
    assert _post(client, "cccccccccccc", {"violation_index": 0, "label": "confirmed"}).status_code == 404
    assert _post(client, "bbbbbbbbbbbb", {"violation_index": 0, "label": "confirmed"}).status_code == 409
    assert _post(client, "aaaaaaaaaaaa", {"violation_index": 3, "label": "confirmed"}).status_code == 409
    assert _post(client, "aaaaaaaaaaaa", {"violation_index": 0, "label": "maybe"}).status_code == 422


def test_job_response_unchanged(client):
    _post(client, "aaaaaaaaaaaa", {"violation_index": 0, "label": "confirmed"})
    body = client.get("/api/v1/jobs/aaaaaaaaaaaa", headers={"X-API-Key": KEY}).json()
    assert set(body) == {"job_id", "status", "verdict", "violations"}


def _config(c):
    return c.get("/api/v1/config", headers={"X-API-Key": KEY})


def test_config_enabled(client):
    assert _config(client).json() == {"feedback_enabled": True}
    assert client.get("/api/v1/config").status_code == 401


@pytest.mark.parametrize("client", [None, "0", "false", "no", ""], indirect=True)
def test_feedback_disabled_by_default(client):
    """Не задано или не «1/true/yes» -> 404 и в таблицу ничего не пишется."""
    assert _config(client).json() == {"feedback_enabled": False}
    r = _post(client, "aaaaaaaaaaaa", {"violation_index": 0, "label": "confirmed"})
    assert r.status_code == 404 and "FEEDBACK_ENABLED" in r.json()["detail"]
    with closing(db.connect(client.db_url)) as conn:
        assert db.list_feedback(conn) == []
    body = client.get("/api/v1/jobs/aaaaaaaaaaaa", headers={"X-API-Key": KEY}).json()
    assert set(body) == {"job_id", "status", "verdict", "violations"}


@pytest.mark.parametrize("client", ["true", "YES", " 1 "], indirect=True)
def test_feedback_enabled_values(client):
    assert _config(client).json() == {"feedback_enabled": True}
    assert _post(client, "aaaaaaaaaaaa", {"violation_index": 0, "label": "confirmed"}).status_code == 200
