import React, { useEffect, useState, useRef } from 'react';

// NOTE: You need to install chart.js and react-chartjs-2 if you want to use the chart
// npm install chart.js react-chartjs-2
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function App() {
  const [prediction, setPrediction] = useState("Waiting for data...");
  const [signalData, setSignalData] = useState([]);
  
  // We'll just visualize Channel 1 for simplicity in the chart
  const WEBSOCKET_URL = "ws://localhost:8000/ws";

  useEffect(() => {
    const ws = new WebSocket(WEBSOCKET_URL);

    ws.onopen = () => {
      console.log("Connected to AI Streaming Server");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("Received AI Update:", data);
      
      setPrediction(data.prediction);

      // Keep only the last 50 data points for UI Chart
      setSignalData((prev) => {
        const newData = [...prev, data.raw_signal[0]]; // Index 0 represents Channel 1
        if (newData.length > 50) newData.shift();
        return newData;
      });
    };

    ws.onclose = () => {
      console.log("Disconnected. Reconnecting...");
      // Add reconnection logic if desired
    };

    return () => {
      ws.close();
    };
  }, []);

  const chartData = {
    labels: Array.from({ length: signalData.length }, (_, i) => i.toString()),
    datasets: [
      {
        label: 'EEG Channel 1 Activity',
        data: signalData,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      }
    ]
  };

  const getStatusColor = (state) => {
    switch (state) {
      case "Active": return "green";
      case "Neutral": return "gray";
      case "Drowsy": return "red";
      default: return "black";
    }
  };

  return (
    <div style={{ padding: '40px', fontFamily: 'sans-serif' }}>
      <h1>Driver Cognitive State Monitor</h1>
      
      <div style={{
        marginTop: '20px',
        padding: '20px', 
        border: '2px solid #ccc',
        borderRadius: '10px',
        backgroundColor: '#f9f9f9',
        display: 'inline-block'
      }}>
        <h2>Current Status: </h2>
        <h1 style={{ color: getStatusColor(prediction), fontSize: '48px', margin: 0 }}>
          {prediction}
        </h1>
      </div>

      <div style={{ marginTop: '40px', width: '800px' }}>
        <h3>Real-time EEG Stream (Channel 1)</h3>
        {signalData.length > 0 ? (
          <Line data={chartData} options={{ animation: false }} />
        ) : (
          <p>No signal detected yet. Waiting for ESP32 & Kafka streams...</p>
        )}
      </div>
    </div>
  );
}

export default App;
