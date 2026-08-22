"""Public SDK contract: the surface third parties integrate against.

These tests are deliberately strict about shape and serialisation — the point of
an SDK is that the surface stops moving.
"""

from __future__ import annotations

import json

import pytest

from formulagate.sdk import (
    ClaimResult,
    Formulagate,
    Source,
    VerifyResult,
    check,
    verify,
)

SOURCES = [
    {
        "id": "rel-1",
        "text": "energy mass equivalence relativity rest frame",
        "formula": r"$E = m c^2$ with $\gamma = 1$",
        "domain": "physics",
    },
    {
        "id": "rel-2",
        "text": "momentum mass velocity relativity frame",
        "formula": r"$p = m v$ with $\gamma = 1$",
        "domain": "physics",
    },
]


# ─── verify: the deterministic layer, no corpus involved ──────────────────────


def test_verify_accepts_valid_physics() -> None:
    result = verify("E = m c^2")
    assert isinstance(result, VerifyResult)
    assert result.ok
    assert result.dimensions == "consistent"
    assert set(result.symbols) == {"E", "m", "c"}


def test_verify_rejects_dimensionally_impossible_physics() -> None:
    result = verify("E = m c^3")
    assert not result.ok
    assert result.dimensions == "inconsistent"
    assert "M L^2 T^-2" in result.reason


def test_verify_proves_equivalence_against_a_reference() -> None:
    result = verify("E = m c^2", against="m c^2 = E")
    assert result.equivalence == "equivalent"
    assert result.ok


def test_verify_flags_a_contradicting_reference() -> None:
    result = verify("E = m c^2", against="E = m c^4")
    assert result.equivalence == "different"
    assert not result.ok


def test_verify_is_unknown_not_ok_for_garbage() -> None:
    result = verify(r"\bad{ latex")
    assert not result.parsed
    assert result.dimensions == "unknown"
    assert result.ok is False


def test_verify_result_is_json_serialisable() -> None:
    payload = verify("E = m c^2").to_dict()
    assert json.loads(json.dumps(payload))["dimensions"] == "consistent"


# ─── check: the full gate over in-memory sources ──────────────────────────────


def test_check_accepts_a_supported_claim() -> None:
    result = check(
        brief="energy mass equivalence relativity",
        draft=r"the rest energy obeys $E = m c^2$",
        sources=SOURCES,
    )
    assert isinstance(result, ClaimResult)
    assert result.action == "generate"
    assert result.top_source_id == "rel-1"
    assert 0.0 <= result.confidence <= 1.0


def test_check_abstains_on_impossible_physics_however_well_worded() -> None:
    result = check(
        brief="energy mass equivalence relativity",
        draft=r"the rest energy obeys $E = m c^3$",
        sources=SOURCES,
    )
    assert result.action == "abstain"
    assert "dimension" in result.detail.lower()


def test_check_abstains_without_sources() -> None:
    result = check(brief="energy mass", draft="energy is conserved", sources=[])
    assert result.action == "abstain"
    assert result.top_source_id is None


def test_check_ranks_sources_and_exposes_them() -> None:
    result = check(
        brief="momentum mass velocity relativity",
        draft=r"momentum obeys $p = m v$",
        sources=SOURCES,
    )
    assert result.sources
    assert result.sources[0].id in {"rel-1", "rel-2"}
    assert result.sources[0].score >= result.sources[-1].score


def test_claim_result_is_json_serialisable() -> None:
    payload = check(
        brief="energy mass equivalence relativity",
        draft=r"$E = m c^2$",
        sources=SOURCES,
    ).to_dict()
    assert json.loads(json.dumps(payload))["action"] in {"generate", "abstain"}


# ─── Client: configuration held once, reused per call ─────────────────────────


def test_client_holds_sources_for_repeated_calls() -> None:
    client = Formulagate(sources=SOURCES)
    first = client.check("energy mass equivalence relativity", r"$E = m c^2$")
    second = client.check("momentum mass velocity relativity", r"$p = m v$")
    assert first.action == "generate"
    assert second.action == "generate"


def test_client_can_disable_the_physics_layer() -> None:
    permissive = Formulagate(sources=SOURCES, use_physics=False)
    result = permissive.check("energy mass equivalence relativity", r"$E = m c^3$")
    assert result.action == "generate"
    assert result.physics is None


def test_client_accepts_source_objects_as_well_as_dicts() -> None:
    typed = [Source(id=s["id"], text=s["text"], formula=s["formula"]) for s in SOURCES]
    result = Formulagate(sources=typed).check(
        "energy mass equivalence relativity", r"$E = m c^2$"
    )
    assert result.action == "generate"


def test_client_rejects_a_source_without_an_id() -> None:
    with pytest.raises(ValueError):
        Formulagate(sources=[{"text": "no id", "formula": "$E = m c^2$"}])


def test_calibration_artifacts_load_through_the_client(tmp_path) -> None:
    from formulagate.calibration import CalibrationParams, save_calibration

    path = tmp_path / "cal.json"
    save_calibration(path, CalibrationParams(a=6.0, b=-3.0, threshold=0.6))
    client = Formulagate(sources=SOURCES, calibration=path)
    result = client.check("energy mass equivalence relativity", r"$E = m c^2$")
    assert result.threshold == pytest.approx(0.6)


def test_default_client_never_loads_a_neural_model() -> None:
    """The SDK promises millisecond, GPU-free decisions by default.

    Checked in a subprocess because the rest of the suite may already have
    imported the model for other tests.
    """

    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent(
        """
        import sys
        from formulagate.sdk import Formulagate
        gate = Formulagate(sources=[{"id": "a", "text": "energy mass", "formula": "$E = m c^2$"}])
        gate.check("energy mass equivalence", "the rest energy obeys $E = m c^2$")
        print("torch" in sys.modules, "sentence_transformers" in sys.modules)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().endswith("False False")


def test_version_is_exposed() -> None:
    from formulagate.sdk import __sdk_version__

    assert __sdk_version__.count(".") >= 1
