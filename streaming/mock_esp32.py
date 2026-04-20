import paho.mqtt.client as mqtt
import json
import time
import numpy as np

MQTT_BROKER = "localhost" # Connects to docker mosquitto
MQTT_PORT = 1883
TOPIC = "sensors/eeg"

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "MockESP32")
client.connect(MQTT_BROKER, MQTT_PORT)

print(f"Connected to {MQTT_BROKER}. Starting mock transmission...")

try:
    while True:
        # Generate a mock signal of shape [30 channels, 1001 samples]
        # Using smaller numbers for pure testing, but actual model wants 1001
        
        # To make it realistic for our ESP32 which sends chunks (e.g. 10 samples x 30 channels)
        # However, for pure mockup we can just send the whole [30, 1001] if we want, 
        # but let's mimic the ESP32 chunking behavior.
        
        chunk = np.random.randn(10, 30).tolist()
        
        payload = json.dumps(chunk)
        client.publish(TOPIC, payload)
        print("Published chunk of shape (10, 30)")
        
        time.sleep(0.05) # Send roughly ~20 times per second
        
except KeyboardInterrupt:
    print("Stopping...")
    client.disconnect()
