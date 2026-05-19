from __future__ import annotations

import asyncio

import pytest

from server.events import EventBus


@pytest.mark.asyncio
async def test_publish_delivers_to_single_subscriber() -> None:
    bus = EventBus()
    q = bus.subscribe()
    await bus.publish({"type": "test"})
    event = await asyncio.wait_for(q.get(), timeout=1.0)
    assert event["type"] == "test"


@pytest.mark.asyncio
async def test_publish_broadcasts_to_multiple_subscribers() -> None:
    bus = EventBus()
    q1 = bus.subscribe()
    q2 = bus.subscribe()
    await bus.publish({"type": "broadcast"})
    e1 = await asyncio.wait_for(q1.get(), timeout=1.0)
    e2 = await asyncio.wait_for(q2.get(), timeout=1.0)
    assert e1["type"] == e2["type"] == "broadcast"


@pytest.mark.asyncio
async def test_unsubscribe_removes_queue() -> None:
    bus = EventBus()
    q = bus.subscribe()
    assert bus.subscriber_count == 1
    bus.unsubscribe(q)
    assert bus.subscriber_count == 0


@pytest.mark.asyncio
async def test_full_queue_subscriber_is_dropped_silently() -> None:
    bus = EventBus()
    q = bus.subscribe()
    # Fill queue to capacity (maxsize=1000)
    for _ in range(1000):
        q.put_nowait({"type": "x"})
    # Next publish cannot fit — subscriber must be dropped (not raise)
    await bus.publish({"type": "overflow"})
    assert bus.subscriber_count == 0


@pytest.mark.asyncio
async def test_publish_to_no_subscribers_is_noop() -> None:
    bus = EventBus()
    await bus.publish({"type": "nothing"})  # must not raise
