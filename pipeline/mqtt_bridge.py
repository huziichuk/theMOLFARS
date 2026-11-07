"""Builder question Q-07: feed live MQTT telemetry into the demo without changing the product.

    pip install paho-mqtt
    python pipeline/mqtt_bridge.py --broker localhost --topic "plant/+/telemetry" --api http://localhost:8000

Topic plant/<ASSET-ID>/telemetry, payload JSON {"timestamp": "...", "values": {"vibration_mm_s": 3.1, ...}}
(timestamp optional). Each message becomes POST /api/telemetry, the same table the synthetic pack filled, so the
anomaly detector, the copilot and the sparklines use live data at once. An OPC UA client would post the same way.
"""
import argparse
import json
import urllib.request


def forward(api: str, asset_id: str, payload: dict) -> int:
    body = json.dumps({"asset_id": asset_id, "timestamp": payload.get("timestamp"), "values": payload["values"],
                       "source": "mqtt"}).encode()
    req = urllib.request.Request(f"{api}/api/telemetry", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="plant/+/telemetry")
    ap.add_argument("--api", default="http://localhost:8000")
    a = ap.parse_args()
    import paho.mqtt.client as mqtt  # optional dependency: only this bridge needs it

    def on_message(_client, _userdata, msg):
        try:
            asset_id = msg.topic.split("/")[1]
            print(f"[mqtt] {msg.topic} -> {forward(a.api, asset_id, json.loads(msg.payload))}")
        except Exception as e:  # one bad message must not stop the bridge
            print(f"[mqtt] skipped {msg.topic}: {type(e).__name__} {e}")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect(a.broker, a.port)
    client.subscribe(a.topic)
    print(f"[mqtt] {a.broker}:{a.port} {a.topic} -> {a.api}/api/telemetry")
    client.loop_forever()


if __name__ == "__main__":
    main()
