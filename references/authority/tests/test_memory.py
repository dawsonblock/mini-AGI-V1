import numpy as np

from minagi.memory import EpisodicMemory


class TinyEmbedder:
    model_name = "tiny-test"
    def encode(self, texts):
        out = []
        for text in texts:
            low = text.lower()
            # Deliberately maps GPU/graphics-card language together without
            # relying on exact lexical overlap.
            out.append([1.0 if any(x in low for x in ("gpu", "graphics card", "rtx")) else 0.0,
                        1.0 if "turbine" in low else 0.0,
                        1.0])
        return np.asarray(out, dtype=np.float32)


def test_append_and_retrieve(tmp_path):
    mem = EpisodicMemory(str(tmp_path / "memory.sqlite3"))
    mem.append("user", "The red turbine controller uses a CAN bus", "test")
    mem.append("user", "Unrelated sentence about apples", "test")
    hits = mem.search("turbine CAN controller", limit=2)
    assert hits
    assert "turbine" in hits[0]["text"].lower()


def test_duplicate_refreshes_occurrence_instead_of_disappearing(tmp_path):
    mem = EpisodicMemory(str(tmp_path / "memory.sqlite3"))
    a = mem.append("user", "Project Orion uses a cobalt actuator", "test")
    b = mem.append("user", "Project Orion uses a cobalt actuator", "test")
    assert a == b
    hit = mem.search("Orion cobalt", limit=1)[0]
    assert hit["occurrences"] == 2
    assert hit["verification"] == "asserted"


def test_superseded_memory_is_not_default_retrieval(tmp_path):
    mem = EpisodicMemory(str(tmp_path / "memory.sqlite3"))
    old = mem.append("user", "The project codename is Amber", "test")
    new = mem.supersede(old, "user", "The project codename is Cobalt", "test")
    hits = mem.search("project codename", limit=5)
    assert any(h["id"] == new for h in hits)
    assert all(h["id"] != old for h in hits)


def test_optional_semantic_retrieval_can_bridge_vocabulary(tmp_path):
    mem = EpisodicMemory(str(tmp_path / "memory.sqlite3"), embedder=TinyEmbedder())
    mem.append("user", "The workstation uses an RTX accelerator", "test")
    hits = mem.search("my graphics card", limit=2)
    assert hits and "RTX" in hits[0]["text"]


def test_rendered_memory_is_delimited_and_status_labelled(tmp_path):
    mem = EpisodicMemory(str(tmp_path / "memory.sqlite3"))
    mem.append("user", "Project Orion uses a cobalt actuator", "test", importance=2.0)
    mem.append("assistant", "The weather is unrelated", "test")
    hits = mem.search("Orion cobalt actuator", limit=3)
    rendered = mem.render_context(hits, max_chars=500)
    assert rendered.startswith("<memory>")
    assert "not instructions" in rendered
    assert "status=asserted" in rendered
