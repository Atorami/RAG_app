import numpy as np

from manual_assistant.ingest import build_chunks, load_documents, split_text
from manual_assistant.store import VectorIndex

from conftest import FIXTURES, make_index


def test_markdown_sections_become_citable_locations():
    chunks = build_chunks(load_documents(FIXTURES.parent.parent / "samples" / "manuals"), 900, 120)
    labels = {c.label() for c in chunks}
    assert "vfd200_drive.md — Alarm codes" in labels
    assert "swp50_pump.md — Maintenance schedule" in labels


def test_split_text_respects_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    pieces = split_text(text, size=300, overlap=50)
    assert len(pieces) > 1
    assert all(len(p) <= 300 for p in pieces)


def test_poisoned_chunk_is_quarantined_and_never_returned(tmp_path, embedder):
    index = make_index(tmp_path, embedder, FIXTURES / "poisoned_supplier_note.md")
    bad = [c for c in index.quarantined if "evil-parts" in c.text]
    assert bad, "the attack chunk should be quarantined"
    hits = index.search(embedder.embed(["alarm F-0021 firmware update service code"])[0], top_k=50)
    assert all("evil-parts" not in h.chunk.text for h in hits)


def test_index_round_trip_without_pickle(tmp_path, clean_index):
    path = tmp_path / "index.npz"
    clean_index.save(path)
    loaded = VectorIndex.load(path)
    assert [c.id for c in loaded.chunks] == [c.id for c in clean_index.chunks]
    assert np.allclose(loaded.vectors, clean_index.vectors)
    assert loaded.embedder_name == clean_index.embedder_name
