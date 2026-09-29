from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.event_bus import event_bus

router = APIRouter()


@router.websocket("/ws/twin")
async def twin_websocket(websocket: WebSocket) -> None:
    """Real-time channel for digital twin state deltas and simulation ticks.
    Frontend subscribes once and receives `simulation_tick`, `bed_update`,
    `global_saturation_update`, and `ai_recommendation` messages."""
    await event_bus.subscribe(websocket)
    try:
        while True:
            await websocket.receive_text()  # client is push-only consumer; ignore inbound pings
    except WebSocketDisconnect:
        await event_bus.unsubscribe(websocket)
