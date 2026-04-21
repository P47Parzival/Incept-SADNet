import React, { useEffect, useState } from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { Activity, Brain, AlertTriangle, Radio, RefreshCcw, LayoutGrid } from 'lucide-react';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

const WEBSOCKET_URL = "ws://localhost:8000/ws";

// A highly optimized, lightweight chart for the 30-channel grid
const MiniChart = ({ data, color }: { data: number[], color: string }) => {
  const chartData = {
    labels: Array.from({ length: 60 }, (_, i) => i.toString()), // Fixed generic labels
    datasets: [{
      data: data,
      borderColor: color,
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.1
    }]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: { x: { display: false }, y: { display: false } },
    layout: { padding: { top: 5, bottom: 5 } }
  };

  return <Line data={chartData} options={options as any} />;
};

function App() {
  const [prediction, setPrediction] = useState<string>("Initializing...");
  // Now managing a 2D array: 30 channels, each keeping up to 60 data points
  const [signalData, setSignalData] = useState<number[][]>(Array.from({ length: 30 }, () => []));
  const [isConnected, setIsConnected] = useState<boolean>(false);

  useEffect(() => {
    let ws: WebSocket;

    const connectWebSocket = () => {
      ws = new WebSocket(WEBSOCKET_URL);

      ws.onopen = () => {
        setIsConnected(true);
        setPrediction("Awaiting Stream...");
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setPrediction(data.prediction);

        // data.raw_signal is an array of size 30
        setSignalData((prevGrid) => {
          return prevGrid.map((channelHistory, idx) => {
            const newHistory = [...channelHistory, data.raw_signal[idx]];
            if (newHistory.length > 60) newHistory.shift();
            return newHistory;
          });
        });
      };

      ws.onclose = () => {
        setIsConnected(false);
        setPrediction("Disconnected");
        setTimeout(connectWebSocket, 3000); // Auto-reconnect
      };
    };

    connectWebSocket();

    return () => {
      if (ws) ws.close();
    };
  }, []);

  const getStatusVisuals = (state: string) => {
    switch (state) {
      case "Active":
        return {
          glow: "shadow-[0_0_30px_rgba(34,197,94,0.4)]",
          border: "border-green-500/50",
          text: "text-green-400",
          bg: "bg-green-500/10",
          icon: <Activity className="w-12 h-12 text-green-400" />,
          chartColor: "#4ade80"
        };
      case "Neutral":
        return {
          glow: "shadow-[0_0_30px_rgba(168,162,158,0.4)]",
          border: "border-stone-500/50",
          text: "text-stone-300",
          bg: "bg-stone-500/10",
          icon: <Brain className="w-12 h-12 text-stone-300" />,
          chartColor: "#a8a29e"
        };
      case "Drowsy":
        return {
          glow: "shadow-[0_0_30px_rgba(239,68,68,0.6)]",
          border: "border-red-500/50",
          text: "text-red-400",
          bg: "bg-red-500/10",
          icon: <AlertTriangle className="w-12 h-12 text-red-400 animate-pulse" />,
          chartColor: "#ef4444"
        };
      default:
        return {
          glow: "",
          border: "border-white/10",
          text: "text-gray-400",
          bg: "bg-gray-800/50",
          icon: <RefreshCcw className="w-12 h-12 text-gray-500 animate-spin" />,
          chartColor: "#6366f1"
        };
    }
  };

  const visuals = getStatusVisuals(prediction);

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-gray-100 font-sans selection:bg-indigo-500/30 overflow-x-hidden">

      {/* Background Orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-600/20 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/20 blur-[120px] rounded-full" />
      </div>

      <main className="relative max-w-7xl mx-auto px-6 py-12 flex flex-col gap-8 z-10">

        {/* Header */}
        <header className="flex justify-between items-center pb-6 border-b border-white/5">
          <div className="flex items-center gap-4">
            <div className="flex items-center justify-center">
              <img src="/logo.png" alt="InceptSADNet Logo" className="w-16 h-16 object-contain" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
                InceptSADNet
              </h1>
              <p className="text-sm text-gray-500 font-medium tracking-wide">COGNITIVE STATE MONITOR</p>
            </div>
          </div>

          <div className="flex items-center gap-3 px-4 py-2 rounded-full bg-white/5 border border-white/10 backdrop-blur-md">
            <span className="relative flex h-3 w-3">
              {isConnected && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>}
              <span className={`relative inline-flex rounded-full h-3 w-3 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
            </span>
            <span className="text-sm font-semibold text-gray-300">
              {isConnected ? 'ESP32 Connected' : 'System Offline'}
            </span>
          </div>
        </header>

        {/* Top Widgets Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Main Status Panel */}
          <div className={`col-span-1 lg:col-span-1 rounded-3xl p-8 border backdrop-blur-xl transition-all duration-500 ${visuals.glow} ${visuals.border} ${visuals.bg} flex flex-col items-center justify-center min-h-[300px]`}>
            <div className="mb-6 p-4 rounded-full bg-black/20">
              {visuals.icon}
            </div>
            <h2 className="text-gray-400 font-medium tracking-widest text-sm mb-2 uppercase">Current State</h2>
            <h1 className={`text-5xl font-black tracking-tight ${visuals.text} transition-colors duration-500`}>
              {prediction}
            </h1>
          </div>

          {/* Metrics Panel */}
          <div className="col-span-1 lg:col-span-2 grid grid-cols-2 gap-6">
            <div className="rounded-3xl p-8 bg-white/5 border border-white/10 backdrop-blur-xl flex flex-col justify-center">
              <div className="flex items-center gap-3 mb-4">
                <Radio className="w-5 h-5 text-indigo-400" />
                <h3 className="text-gray-400 font-medium tracking-wide">Live Stream Rate</h3>
              </div>
              <p className="text-4xl font-bold text-white mb-1">512 <span className="text-lg text-gray-500 font-normal">Hz</span></p>
              <p className="text-sm text-gray-500">30 Channels via MQTT</p>
            </div>

            <div className="rounded-3xl p-8 bg-white/5 border border-white/10 backdrop-blur-xl flex flex-col justify-center">
              <div className="flex items-center gap-3 mb-4">
                <Activity className="w-5 h-5 text-purple-400" />
                <h3 className="text-gray-400 font-medium tracking-wide">Model Latency</h3>
              </div>
              <p className="text-4xl font-bold text-white mb-1">~42 <span className="text-lg text-gray-500 font-normal">ms</span></p>
              <p className="text-sm text-gray-500">Inference via InceptSADNet</p>
            </div>
          </div>

        </div>

        {/* Live Chart Section - 30 Channels Grid */}
        <div className="rounded-3xl p-6 bg-white/5 border border-white/10 backdrop-blur-xl flex flex-col">
          <div className="flex items-center justify-between mb-6 px-2">
            <h2 className="font-semibold text-gray-300 tracking-wide flex items-center gap-2">
              <LayoutGrid className="w-5 h-5" /> Full Scalp Telemetry
            </h2>
            <div className="px-3 py-1 rounded-md bg-indigo-500/20 text-indigo-300 text-xs font-bold border border-indigo-500/20">
              30 CHANNELS
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 lg:grid-cols-6 gap-4">
            {signalData.map((channelHistory, idx) => (
              <div key={idx} className="bg-black/20 border border-white/5 rounded-xl p-3 h-32 flex flex-col hover:border-white/20 transition-colors overflow-hidden">
                <div className="text-[11px] text-gray-500 font-mono mb-2 flex justify-between z-10">
                  <span>CH_{(idx + 1).toString().padStart(2, '0')}</span>
                </div>
                <div className="grow relative w-full h-full">
                  {channelHistory.length > 0 ? (
                    <MiniChart data={channelHistory} color={visuals.chartColor} />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-full h-px bg-white/10 border-dashed"></div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

        </div>

      </main>
    </div>
  );
}

export default App;
