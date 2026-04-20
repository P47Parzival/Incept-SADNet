#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ------------- CONFIGURATION -------------
const char* ssid = "Airtel_dhru_1078";
const char* password = "Air@13811";
const char* mqtt_server = "192.168.1.7"; // IP address of PC running docker-compose
const int mqtt_port = 1883;
const char* mqtt_topic = "sensors/eeg";

WiFiClient espClient;
PubSubClient client(espClient);

// Neural Network requires [30, 1001]
// The ESP32 doesn't have enough RAM to comfortably buffer and JSON-serialize 30x1001 floats (30,030 floats = ~120KB just for raw floats without JSON overhead)
// So we will transmit in smaller chunks or downsampled representations depending on your exact hardware setup.
// To perfectly mimic the expected structure, we will send a payload indicating a chunk of signals.
// For this example, we generate and send dummy signal values.

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("ESP32Client")) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  
  // Optional: Set buffer size if sending large packets (e.g. 4096 bytes)
  client.setBufferSize(4096); 
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // We recommend sending data in smaller chunks (e.g., 10 samples for 30 channels at a time)
  // The Python backend will accumulate them until it has 1001 samples.
  
  // Allocate a large JSON document (Adjust size depending on payload)
  DynamicJsonDocument doc(4096);
  JsonArray payload = doc.to<JsonArray>();

  // Send 10 samples of 30 channels
  for (int sampleIdx = 0; sampleIdx < 10; sampleIdx++) {
    JsonArray channelData = payload.createNestedArray();
    for (int channelIdx = 0; channelIdx < 30; channelIdx++) {
      // Replace this with actual `analogRead` or sensor data reading
      float dummyValue = (float)random(-1000, 1000) / 100.0; 
      channelData.add(dummyValue);
    }
  }

  // Serialize and send
  String output;
  serializeJson(doc, output);
  
  Serial.print("Publishing: ");
  Serial.println(output.substring(0, 50) + "..."); // Print start of payload
  
  client.publish(mqtt_topic, output.c_str());

  // Delay based on sampling rate. If 512Hz, 10 samples take approx ~19.5ms. 
  // For demonstration, we use a slower delay to avoid flooding immediately.
  delay(50); 
}
