from gullivers_router.training import store


def test_append_and_read_records_round_trip(tmp_path):
    path = tmp_path / "nested" / "data.jsonl"
    store.append(path, {"id": "a", "response": "one"})
    store.append(path, {"id": "b", "response": "two"})

    assert list(store.read_records(path)) == [
        {"id": "a", "response": "one"},
        {"id": "b", "response": "two"},
    ]


def test_completed_ids_reports_written_ids(tmp_path):
    path = tmp_path / "data.jsonl"
    store.append(path, {"id": "a", "response": "one"})
    store.append(path, {"id": "b", "response": "two"})

    assert store.completed_ids(path) == {"a", "b"}


def test_read_records_missing_file_is_empty(tmp_path):
    assert list(store.read_records(tmp_path / "absent.jsonl")) == []


def test_read_map_projects_key_and_value(tmp_path):
    path = tmp_path / "data.jsonl"
    store.append(path, {"id": "a", "response": "one"})
    store.append(path, {"id": "b", "response": "two"})

    assert store.read_map(path) == {"a": "one", "b": "two"}


def test_write_and_read_json_round_trip(tmp_path):
    path = tmp_path / "job.json"
    store.write_json(path, {"job_name": "abc"})

    assert store.read_json(path) == {"job_name": "abc"}


def test_read_json_missing_file_is_none(tmp_path):
    assert store.read_json(tmp_path / "absent.json") is None
