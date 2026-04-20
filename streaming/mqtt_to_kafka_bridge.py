import paho.mqtt.client as mqtt
from kafka import KafkaProducer
import json

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "sensors/eeg"

KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "eeg_stream"

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected to MQTT broker with result code {reason_code}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        # Load the JSON chunk (shape: [time_samples, num_channels])
        payload = json.loads(msg.payload.decode())
        
        # We simply pass it through to Kafka
        producer.send(KAFKA_TOPIC, value=payload)
        
    except Exception as e:
        print(f"Error processing message: {e}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "MQTT_Kafka_Bridge")
client.on_connect = on_connect
client.on_message = on_message

print("Connecting to MQTT to bridge to Kafka...")
client.connect(MQTT_BROKER, MQTT_PORT, 60)

client.loop_forever()
