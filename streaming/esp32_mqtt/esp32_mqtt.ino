#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ------------- CONFIGURATION -------------
const char* ssid = "Airtel_dhru_1078";
const char* password = "Air@13811";
const char* mqtt_server = "192.168.1.9"; // IP address of PC running docker-compose
const int mqtt_port = 1883;
const char* mqtt_topic = "sensors/eeg";

// --- LED CONFIGURATION ---
const int BLUE_LED_PIN = 2; // Change '2' to whichever GPIO pin your blue LED is wired to
TaskHandle_t blinkTaskHandle = NULL;

// FreeRTOS Task for strictly independent blinking
void blinkTask(void *pvParameters) {
  pinMode(BLUE_LED_PIN, OUTPUT);
  while(true) {
    digitalWrite(BLUE_LED_PIN, HIGH);
    vTaskDelay(3000 / portTICK_PERIOD_MS); // Wait 3 seconds
    digitalWrite(BLUE_LED_PIN, LOW);
    vTaskDelay(3000 / portTICK_PERIOD_MS); // Wait 3 seconds
  }
}
// -------------------------

WiFiClient espClient;
PubSubClient client(espClient);

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
  
  // --- LED SETUP (FreeRTOS Task) ---
  // This spins up a completely separate thread just for blinking the LED
  // It will never be blocked by WiFi or MQTT delays!
  xTaskCreate(
    blinkTask,   /* Task function. */
    "BlinkTask", /* String with name of task. */
    1024,        /* Stack size in bytes. */
    NULL,        /* Parameter passed as input of the task */
    1,           /* Priority of the task. */
    &blinkTaskHandle); /* Task handle. */
  // -----------------

  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  
  // Optional: Set buffer size if sending large packets (e.g. 4096 bytes)
  client.setBufferSize(4096); 
}

void loop() {
  // The LED blinking is now handled invisibly by the FreeRTOS BlinkTask.
  // We no longer need millis() logic here!

  if (!client.connected()) {
    reconnect();
  }
  client.loop();

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