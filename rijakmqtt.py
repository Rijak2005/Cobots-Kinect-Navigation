import time
import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
PORT = 1883

TOPIC_GRIP = "rijakisthebest/cobots/grip"
TOPIC_TURN = "rijakisthebest/cobots/turn"

def make_client():
    # Works on paho-mqtt v1 and v2
    try:
        return mqtt.Client(
            client_id=f"nav-{int(time.time())}",
            clean_session=True,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
        )
    except Exception:
        return mqtt.Client(client_id=f"nav-{int(time.time())}", clean_session=True)

c = make_client()

def on_connect(client, userdata, flags, rc):
    print("Connected rc=", rc)

def on_disconnect(client, userdata, rc):
    print("Disconnected rc=", rc)

c.on_connect = on_connect
c.on_disconnect = on_disconnect

c.connect(BROKER, PORT, keepalive=30)
c.loop_start()

time.sleep(1.0)

print("Publishing OPEN")
c.publish(TOPIC_GRIP, "OPEN", qos=0, retain=False)

time.sleep(10.0)
print("Publishing CLOSE")
c.publish(TOPIC_GRIP, "CLOSE", qos=0, retain=False)

time.sleep(1.0)
c.loop_stop()
c.disconnect()