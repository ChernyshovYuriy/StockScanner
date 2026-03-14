import json

from time_utils import market_now


def log(service: str, run_id: str, event: str, **fields):
    payload = {
        "ts": market_now(),
        "service": service,
        "run_id": run_id,
        "event": event,
        **fields,
    }
    print(json.dumps(payload, default=str))
