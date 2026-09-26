from kuulo_node.outbox import Outbox


def bodies(box: Outbox) -> list[str]:
    return [body for _, _, body in box.peek(1000)]


def test_messages_come_back_in_order():
    box = Outbox()
    for i in range(5):
        box.put("/v1/observations", f"o{i}")
    assert bodies(box) == ["o0", "o1", "o2", "o3", "o4"]
    assert len(box) == 5


def test_delete_removes_only_given_rows():
    box = Outbox()
    for i in range(3):
        box.put("/v1/observations", f"o{i}")
    first, _, third = (row_id for row_id, _, _ in box.peek(3))
    box.delete([first, third])
    assert bodies(box) == ["o1"]


def test_survives_reopen(tmp_path):
    path = tmp_path / "outbox.db"
    box = Outbox(path)
    box.put("/v1/heartbeats", "h0")
    box.put("/v1/observations", "o0")
    box.close()
    again = Outbox(path)
    assert [(p, b) for _, p, b in again.peek(10)] == [("/v1/heartbeats", "h0"),
                                                      ("/v1/observations", "o0")]


def test_full_drops_oldest_heartbeat_first():
    box = Outbox(cap=3)
    box.put("/v1/observations", "o1")
    box.put("/v1/heartbeats", "h1")
    box.put("/v1/observations", "o2")
    box.put("/v1/observations", "o3")
    assert bodies(box) == ["o1", "o2", "o3"] and box.dropped == 1


def test_full_without_heartbeats_drops_oldest_observation_and_counts():
    box = Outbox(cap=2)
    for i in range(4):
        box.put("/v1/observations", f"o{i}")
    assert bodies(box) == ["o2", "o3"] and box.dropped == 2


def test_order_is_kept_after_drops_and_reopen(tmp_path):
    path = tmp_path / "outbox.db"
    box = Outbox(path, cap=3)
    for i in range(5):
        box.put("/v1/observations", f"o{i}")
    box.close()
    assert bodies(Outbox(path, cap=3)) == ["o2", "o3", "o4"]
