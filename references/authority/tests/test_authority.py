import pytest

from minagi.authority import WriterLease, read_writer_metadata


def test_writer_lease_is_exclusive_and_reusable(tmp_path):
    a = WriterLease(tmp_path, "first").acquire()
    meta = read_writer_metadata(tmp_path)
    assert meta and meta["purpose"] == "first"
    with pytest.raises(RuntimeError):
        WriterLease(tmp_path, "second").acquire()
    a.release()
    b = WriterLease(tmp_path, "second").acquire()
    b.release()
