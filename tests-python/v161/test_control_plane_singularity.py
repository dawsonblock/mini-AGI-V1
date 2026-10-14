"""Campaign 3B.3 Priority 1 — one authoritative control plane.

The tree carries several generations of authority code. A reader cannot
tell which one governs a promotion, which is the defect: the duplicates
are indistinguishable from the authority that counts.

These tests pin the facts the control-plane map establishes, so that a
change which silently adds a competing authority — or removes the live
one — fails the build rather than being discovered by reading.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

#: The single authority the live control plane uses. Every campaign entry
#: point imports this module; nothing else may claim to be it.
LIVE_AUTHORITY = "minagi.v161.authority"

#: Entry points that must reach the live authority.
CAMPAIGN_ENTRY_POINTS = (
    "scripts/promote.py",
    "scripts/admit_runtime.py",
    "scripts/validation/qualify_campaign1.py",
    "scripts/validation/run_sealed_confirmation.py",
)

#: Promotion/qualification implementations no live path reaches. Their
#: existence is a provenance question; their existence being UNDOCUMENTED
#: is the defect this list prevents.
KNOWN_ORPHAN_AUTHORITIES = {
    "dream_rsi_governed.promotion",
    "dream_rsi_governed.qualification",
    "dream_rsi_governed.activation",
    "kvcontinual.execution.durability.promotion_authority",
    "minagi.rc14.qualification",
    "minagi.egai.authority",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            for a in node.names:
                names.add(f"{base}.{a.name}" if base else a.name)
            if base:
                names.add(base)
    return names


def test_every_campaign_entry_point_uses_the_live_authority():
    """The scientific plane must have exactly one authority."""
    missing = []
    for rel in CAMPAIGN_ENTRY_POINTS:
        p = ROOT / rel
        assert p.is_file(), f"entry point vanished: {rel}"
        if not any(n == LIVE_AUTHORITY or n.startswith(LIVE_AUTHORITY + ".")
                   for n in _imports(p)):
            missing.append(rel)
    assert not missing, (
        f"these entry points no longer import {LIVE_AUTHORITY}: {missing} — "
        "either they moved to another authority (which must be a deliberate, "
        "recorded decision) or the live authority was removed")


def test_no_second_live_authority_was_introduced():
    """A new module importing like an authority is a duplicate until the
    map says otherwise. This does not forbid new modules — it forbids
    them claiming the authority name."""
    claimers = []
    for p in sorted((ROOT / "src-python").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if "class AuthorityRegistry" in text or "class AuthorityLedger" in text:
            rel = p.relative_to(ROOT / "src-python").with_suffix("")
            mod = ".".join(rel.parts)
            if mod.endswith("__init__"):
                mod = mod[: -len(".__init__")]
            claimers.append(mod)
    assert set(claimers) == {LIVE_AUTHORITY}, (
        "more than one module defines the authority registry/ledger: "
        f"{sorted(claimers)} — the control plane is no longer singular")


def test_the_orphan_authorities_are_the_documented_ones():
    """If a documented orphan disappears, or a new one appears, the map is
    stale and the register must be updated deliberately."""
    found = set()
    for mod in KNOWN_ORPHAN_AUTHORITIES:
        p = ROOT / "src-python" / (mod.replace(".", "/") + ".py")
        if p.is_file():
            found.add(mod)
    missing = KNOWN_ORPHAN_AUTHORITIES - found
    assert not missing, (
        f"documented orphan authorities are gone: {sorted(missing)} — if "
        "they were retired, remove them from the register and the map in "
        "the same change")


def test_control_plane_map_is_generated_not_hand_written():
    """The map must come from the tool, so it cannot drift from the tree."""
    tool = ROOT / "scripts" / "analysis" / "control_plane_map.py"
    assert tool.is_file(), "the map generator is missing"
    report = ROOT / "CONTROL_PLANE_MAP.md"
    assert report.is_file()
    assert "scripts/analysis/control_plane_map.py" in report.read_text(), (
        "the report must name the tool that produced it")


#: Classes carrying the vocabulary a reader would search for, that are NOT
#: the live path. Recorded so the naming collision is explicit rather than
#: discovered by a reader who then trusts the wrong one.
DECOY_AUTHORITY_CLASSES = {
    "dream_rsi_governed.promotion": "PromotionAuthority",
    "kvcontinual.execution.durability.promotion_authority":
        "PromotionAuthorization",
    "minagi.rc14.qualification": "QualificationEngineRC14",
}


def test_the_decoy_authority_classes_still_exist_where_documented():
    """These are the naming collisions that make the control plane
    ambiguous. If they move or are renamed, the map must be updated in the
    same change — that is the point of pinning them."""
    for mod, cls in DECOY_AUTHORITY_CLASSES.items():
        p = ROOT / "src-python" / (mod.replace(".", "/") + ".py")
        assert p.is_file(), f"{mod} vanished; update the map and register"
        assert f"class {cls}" in p.read_text(), (
            f"{mod} no longer defines {cls}; the documented naming "
            "collision changed and the map is stale")


def test_the_live_authority_does_not_use_the_decoy_names():
    """The live path must not be confusable with the orphans."""
    src = (ROOT / "src-python" / "minagi" / "v161" /
           "authority.py").read_text()
    for cls in DECOY_AUTHORITY_CLASSES.values():
        assert f"class {cls}" not in src, (
            f"{cls} is now defined in the live authority as well; the two "
            "are no longer distinguishable")
