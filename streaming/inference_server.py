import asyncio
import json
import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from kafka import KafkaConsumer
from fastapi.middleware.cors import CORSMiddleware
import threading

import sys
import os
# Ensure we can import the model from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import InceptSADNet

# ----------------- MODEL SETUP ----------------- #
# This will initialize the model with RANDOM WEIGHTS since we do not have process the .ckpt yet!
device = torch.device('cpu') 
class DummyConfig:
    device = device
    # Required for classification head
    num_classes = 3

model_config = DummyConfig()
model = InceptSADNet.Model(config=model_config)
model = model.to(device)
model.eval()

classes = ["Active", "Neutral", "Drowsy"]

# ----------------- GLOBAL STATE ----------------- #
# We need to buffer incoming packets until we reach 1001 samples
# Shape goal: [1001, 30]
eeg_buffer = []

# Connected WebSocket clients
clients = set()

# Reference to the main asyncio event loop (set during startup)
main_event_loop = None

# ----------------- KAFKA CONSUMER BACKGROUND TASK ----------------- #
def consume_kafka_loop():
    global eeg_buffer
    consumer = KafkaConsumer(
        'eeg_stream',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    print("Started Kafka Consumer Thread.")
    
    for message in consumer:
        chunk = message.value  # Expected to be shape [num_samples, 30]
        
        # Append chunk data to buffer
        # Assume chunk is a list of lists: [[ch1..ch30], [ch1..ch30], ...]
        eeg_buffer.extend(chunk)
        
        # When buffer hits our window size (1001), process it!
        if len(eeg_buffer) >= 1001:
            # Extract exactly 1001 samples
            window = eeg_buffer[:1001]
            eeg_buffer = eeg_buffer[1001:] # Keep the remainder
            
            # Convert to numpy, then tensor
            # Expected input to model: [batch_size, 30, 1001]
            input_tensor = torch.tensor(window, dtype=torch.float32) # [1001, 30]
            # Permute logic as handled in InceptSADNet Model class: it expects [batch, 1001, 30] and does permute internally
            # Wait, looking at InceptSADNet `forward`: 
            # x.permute(0, 2, 1) # -> [batch, 30, 1001]
            # So we pass [1, 1001, 30] to it.
            input_tensor = input_tensor.unsqueeze(0) # [1, 1001, 30]
            
            with torch.no_grad():
                logits = model(input_tensor)
                prediction_idx = logits.argmax(axis=-1).item()
                prediction_label = classes[prediction_idx]
            
            # Prepare broadcast dictionary
            # Just send a tiny subset of the raw signal to the frontend for visualization
            raw_sample = window[-1] # The last frame of 30 channels
            
            broadcast_data = {
                "prediction": prediction_label,
                "raw_signal": raw_sample,     # array of 30 floats
                "buffer_size": len(eeg_buffer)
            }
            
            # Send to all connected React clients via WebSockets
            # Use the captured main_event_loop since we're in a background thread
            if clients and main_event_loop is not None:
                for client in list(clients):
                    asyncio.run_coroutine_threadsafe(client.send_json(broadcast_data), main_event_loop)

# ----------------- LIFESPAN (replaces deprecated on_event) ----------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_event_loop
    # Capture the running event loop so the Kafka thread can schedule coroutines
    main_event_loop = asyncio.get_running_loop()
    # Start the Kafka consumer thread in the background
    thread = threading.Thread(target=consume_kafka_loop, daemon=True)
    thread.start()
    yield
    # Cleanup (if needed) goes here

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# Allow CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
    except Exception as e:
        print(f"Client disconnected: {e}")
    finally:
        clients.discard(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ws="wsproto")
