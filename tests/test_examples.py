import json
from pathlib import Path


def test_practice_tasks_match_runtime_contract():
    tasks = json.loads(Path("examples/practice_tasks.json").read_text(encoding="utf-8"))

    assert len(tasks) == 8
    assert all(set(task) == {"task_id", "prompt"} for task in tasks)
    assert all(isinstance(task["task_id"], str) and task["task_id"] for task in tasks)
    assert all(isinstance(task["prompt"], str) and task["prompt"] for task in tasks)
