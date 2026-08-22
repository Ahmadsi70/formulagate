# SDK Reference

## Formulagate class

```python
class Formulagate:
    """Configured gate: sources and calibration are supplied once, then reused."""

    def __init__(
        self,
        sources: Iterable[Source | Mapping[str, Any]] | None = None,
        *,
        calibration: str | Path | None = None,
        fused_calibration: str | Path | None = None,
        use_physics: bool = True,
        semantic: bool = False,
        top_k: int = 3,
        conformal: "ConformalCalibration | None" = None,
    ) -> None: ...

    def verify(self, latex: str, against: str | None = None) -> VerifyResult: ...
    def check(
        self,
        brief: str,
        draft: str,
        sources: Iterable[Source | Mapping[str, Any]] | None = None,
        top_k: int | None = None,
    ) -> ClaimResult: ...
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sources` | `Iterable[Source]` | `None` | Knowledge base records |
| `calibration` | `str \| Path` | `None` | Platt scaling artifact path |
| `fused_calibration` | `str \| Path` | `None` | Similarity+physics artifact |
| `use_physics` | `bool` | `True` | Enable dimensional veto |
| `semantic` | `bool` | `False` | Use embedding scoring |
| `top_k` | `int` | `3` | Number of ranked sources to consider |
| `conformal` | `ConformalCalibration` | `None` | Risk-controlled threshold |

## Source dataclass

```python
@dataclass(frozen=True)
class Source:
    id: str
    text: str = ""
    formula: str = ""
    domain: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)
```

## VerifyResult

```python
@dataclass(frozen=True)
class VerifyResult:
    formula: str
    parsed: bool
    ok: bool
    dimensions: str       # "consistent" | "inconsistent" | "unknown"
    reason: str = ""
    symbols: tuple[str, ...] = ()
    structure_hash: str | None = None
    equivalence: str | None = None
    equivalence_method: str | None = None

    @property
    def refuted(self) -> bool: ...  # True if algebra proved wrong
    def to_dict(self) -> dict[str, Any]: ...
```

## ClaimResult

```python
@dataclass(frozen=True)
class ClaimResult:
    action: str            # "generate" | "abstain"
    detail: str
    confidence: float | None = None
    threshold: float | None = None
    combined_score: float | None = None
    physics: dict[str, float] | None = None
    model: str = "platt"
    sources: tuple[RankedSource, ...] = ()
    domain: str = "general"

    @property
    def ok(self) -> bool: ...  # True if action == "generate"
    def to_dict(self) -> dict[str, Any]: ...
```

## Module-level convenience functions

```python
from formulagate.sdk import verify, check

# One-off verify
result = verify(r"$E = m c^2$", against=r"$m c^2 = E$")

# One-off check
decision = check(
    brief="What is E=mc^2?",
    draft="Energy equals mass times c squared",
    sources=[{"id": "1", "text": "...", "formula": "E = mc^2"}],
)
```
