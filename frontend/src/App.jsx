import React, { useState, useRef, useEffect } from 'react';
// import { Capacitor } from '@capacitor/core'; // Uncomment if using Capacitor

const RiceGuardApp = () => {
  // ---------------------------------------------------------------------------
  // CONFIGURATION
  // ---------------------------------------------------------------------------
  // ⚠️ Point this to your FastAPI Backend URL
  // If testing on Android Emulator use "http://10.0.2.2:8000"
  // If testing on Physical Phone use "http://YOUR_PC_IP:8000"
  const API_BASE = import.meta.env.VITE_API_BASE || "http://10.187.60.131:8000"; 
  // ---------------------------------------------------------------------------

  // --- STATE ---
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
   
  // History State
  const [history, setHistory] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Camera & Sensor State
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [guidance, setGuidance] = useState('Ready to capture');
  const [canCapture, setCanCapture] = useState(false);
   
  // Tap-to-Focus State
  const [focusBox, setFocusBox] = useState({ x: 0, y: 0, visible: false });
   
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const lastVibrate = useRef(0);

  // Wake up backend & Fetch History
  useEffect(() => {
    fetch(`${API_BASE}/health`).catch(() => console.log("Backend not waking up yet..."));
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/history`);
      const data = await res.json();
      if (Array.isArray(data)) {
        setHistory(data);
        console.log("History loaded:", data); // Debugging
      }
    } catch (e) {
      console.error("History fetch failed", e);
    }
  };

  // --- HELPER: VIBRATION ---
  const triggerVibration = (duration = 200) => {
    const now = Date.now();
    if (now - lastVibrate.current > 500) { 
      if (typeof navigator.vibrate === 'function') navigator.vibrate(duration);
      lastVibrate.current = now;
    }
  };

  // --- GYROSCOPE (With Tilt Vibration) ---
  useEffect(() => {
    if (!isCameraOpen) return;
    
    if (!window.DeviceOrientationEvent) {
       setCanCapture(true);
       return;
    }

    const handleOrientation = (e) => {
      const pitch = e.beta || 0;
      const roll = e.gamma || 0;

      if (Math.abs(pitch) > 10 || Math.abs(roll) > 10) { // Increased tolerance slightly
        setGuidance("Keep phone flat");
        setCanCapture(false);
        triggerVibration(50); 
      } else {
        setGuidance("Ready to capture");
        setCanCapture(true);
      }
    };
    
    window.addEventListener('deviceorientation', handleOrientation);
    return () => window.removeEventListener('deviceorientation', handleOrientation);
  }, [isCameraOpen]);

  // --- HANDLERS ---
  const handleFileChange = (e) => {
    const file = e instanceof File ? e : e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResults(null);
      setError('');
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
      stopCamera();
    }
  };

  const handleHistoryClick = (item) => {
    // Load a history item into the main view
    setPreview(item.url);
    // If we have stats saved in history, we could ideally restore them here
    // For now, we just show the image as per the requirements
    setResults(null); 
    setError(`Loaded scan from ${item.timestamp}`);
    setSidebarOpen(false); // Close sidebar on selection
    stopCamera();
  };

  // --- TAP TO FOCUS LOGIC ---
  const handleTapToFocus = async (e) => {
    if (!videoRef.current || !videoRef.current.srcObject) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setFocusBox({ x, y, visible: true });
    
    setTimeout(() => {
        setFocusBox(prev => ({ ...prev, visible: false }));
    }, 1000);

    const track = videoRef.current.srcObject.getVideoTracks()[0];
    const capabilities = track.getCapabilities();

    if (capabilities.focusMode) {
      try {
        await track.applyConstraints({ advanced: [{ focusMode: 'auto' }] });
        setTimeout(async () => {
            await track.applyConstraints({ advanced: [{ focusMode: 'continuous' }] });
        }, 200);
      } catch (err) {}
    }
  };

  const startCamera = async () => {
    setIsCameraOpen(true);
    setPreview('');
    setResults(null);
    setError('');
    setCanCapture(false);
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: 'environment', 
          aspectRatio: 1, 
          width: { ideal: 1080 },
          height: { ideal: 1080 }
        }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => videoRef.current.play();
      }
    } catch (err) {
      setError("Camera access denied.");
      setIsCameraOpen(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) videoRef.current.srcObject.getTracks().forEach(t => t.stop());
    setIsCameraOpen(false);
  };

  const capturePhoto = () => {
    if (!canCapture || !videoRef.current || !canvasRef.current) return;

    const vid = videoRef.current;
    const size = Math.min(vid.videoWidth, vid.videoHeight);
    const startX = (vid.videoWidth - size) / 2;
    const startY = (vid.videoHeight - size) / 2;

    canvasRef.current.width = size;
    canvasRef.current.height = size;
    const ctx = canvasRef.current.getContext('2d');
    ctx.drawImage(vid, startX, startY, size, size, 0, 0, size, size);
    
    canvasRef.current.toBlob(blob => {
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
      handleFileChange(file);
    }, 'image/jpeg');
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: formData });
      
      // --- DISTANCE LOGIC INTEGRATION START ---
      // We read the body first because we might need the error detail
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        // If it's a 400 error, it's likely our "Camera Too Far/Close" check
        if (response.status === 400 && data.detail) {
            setError(`⚠️ ${data.detail}`); // Display the distance warning
            triggerVibration(300); // Vibrate to alert user
        } else {
            throw new Error(data.detail || "Server error");
        }
        return; // Stop here, do not proceed to setResults
      }
      // --- DISTANCE LOGIC INTEGRATION END ---

      if (data.total_grains === 0 && !data.warnings?.includes("Screenshot detected")) {
        setResults(null);
        setError("No rice grains detected.");
        triggerVibration(500); 
      } else {
        setResults(data);
        
        // --- OPTIMISTIC UPDATE ---
        const newHistoryItem = {
           url: data.image_url || `data:image/jpeg;base64,${data.visualization}`,
           timestamp: data.timestamp, 
           stats: {
             total: data.total_grains,
             whole: data.whole_grains,
             broken: data.broken_grains
           }
        };

        setHistory(prevHistory => [newHistoryItem, ...prevHistory]);
      }
      
    } catch (err) {
      console.error(err);
      setError(err.message || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    stopCamera();
    setSelectedFile(null);
    setPreview('');
    setResults(null);
    setError('');
  };

  const displayImage = results?.visualization 
    ? `data:image/jpeg;base64,${results.visualization}` 
    : preview;

  const getPct = (val) => {
    if (!results || results.total_grains === 0) return "0%";
    return ((val / results.total_grains) * 100).toFixed(1) + "%";
  };

  return (
    <div className="main-layout">
      {/* -------------------------------------------------------- */}
      {/* GLOBAL STYLES */}
      {/* -------------------------------------------------------- */}
      <style>{`
        body { margin: 0; background-color: #f8fafc; overflow-x: hidden; }
        .main-layout { position: relative; min-height: 100vh; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        
        /* SIDEBAR DRAWER STYLES */
        .sidebar-overlay { 
          position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
          background: rgba(0,0,0,0.5); z-index: 45; opacity: 0; pointer-events: none; transition: opacity 0.3s;
        }
        .sidebar-overlay.open { opacity: 1; pointer-events: auto; }
        
        .sidebar { 
          position: fixed; top: 0; left: 0; height: 100%; width: 280px; 
          background: #ffffff; box-shadow: 2px 0 12px rgba(0,0,0,0.1); 
          transform: translateX(-100%); transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          z-index: 50; display: flex; flex-direction: column;
        }
        .sidebar.open { transform: translateX(0); }
        
        .sidebar-header { padding: 1.5rem; border-bottom: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center; }
        .sidebar-title { font-size: 1.2rem; font-weight: 700; color: #1e293b; margin: 0; }
        .close-btn { background: none; border: none; font-size: 1.5rem; color: #64748b; cursor: pointer; padding: 0; line-height: 1; }
        
        .history-list { padding: 1rem; overflow-y: auto; flex: 1; }
        .history-item { display: flex; align-items: center; gap: 12px; padding: 10px; border-radius: 8px; cursor: pointer; transition: 0.2s; margin-bottom: 8px; border: 1px solid #f1f5f9; }
        .history-item:hover { background: #f8fafc; border-color: #cbd5e0; }
        .thumb { width: 50px; height: 50px; border-radius: 6px; object-fit: cover; background: #eee; border: 1px solid #e2e8f0; }
        .meta { display: flex; flex-direction: column; font-size: 0.85rem; width: 100%; }
        .meta-time { color: #1e293b; font-weight: 600; font-size: 0.95rem; }
        .meta-subtitle { font-size: 0.75rem; color: #64748b; margin-top: 2px; display: flex; justify-content: space-between; }
        
        /* FLOATING HISTORY TOGGLE BUTTON */
        .history-toggle {
          position: fixed; bottom: 24px; left: 24px; z-index: 40;
          background: #0f172a; color: white; border: none;
          width: 56px; height: 56px; border-radius: 50%;
          box-shadow: 0 4px 12px rgba(0,0,0,0.25);
          display: flex; align-items: center; justify-content: center;
          cursor: pointer; transition: transform 0.2s;
          font-size: 1.5rem;
        }
        .history-toggle:active { transform: scale(0.95); }

        /* MAIN CONTENT STYLES */
        .app-content { width: 100%; max-width: 1200px; margin: 0 auto; padding: 1.5rem; }
        .card { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }
        .header { text-align: center; margin-bottom: 2rem; }
        .header h1 { margin: 0; color: #1a202c; font-size: 2rem; }
        .header p { color: #64748b; margin-top: 0.5rem; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
        .upload-zone { border: 2px dashed #cbd5e0; border-radius: 12px; padding: 2rem; text-align: center; transition: 0.3s; cursor: pointer; display: block; background: #fff; }
        .upload-zone:hover { border-color: #000; background: #f1f5f9; }
        
        .video-container { position: relative; width: 100%; padding-top: 100%; overflow: hidden; border-radius: 8px; background: #000; margin-bottom: 1rem; cursor: crosshair; }
        .video-display { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }
        
        .focus-box {
            position: absolute; width: 60px; height: 60px;
            border: 2px solid #fbbf24; box-shadow: 0 0 8px rgba(251, 191, 36, 0.6);
            transform: translate(-50%, -50%); pointer-events: none;
            z-index: 20; animation: focusPulse 0.6s ease-out forwards;
        }
        @keyframes focusPulse {
            0% { transform: translate(-50%, -50%) scale(1.4); opacity: 0.5; }
            50% { transform: translate(-50%, -50%) scale(1.0); opacity: 1; }
            100% { transform: translate(-50%, -50%) scale(1.0); opacity: 0; }
        }

        .img-display { width: 100%; border-radius: 8px; object-fit: contain; background: #1a202c; max-height: 400px; margin-bottom: 1rem; }
        
        .stats-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        .stats-table th { text-align: left; padding: 12px; border-bottom: 2px solid #edf2f7; color: #64748b; font-size: 0.75rem; text-transform: uppercase; }
        .stats-table td { padding: 12px; border-bottom: 1px solid #f1f5f9; }
        .label { color: #475569; font-weight: 500; }
        .value { text-align: right; font-weight: 700; color: #1e293b; }
        .badge { padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
        .bg-good { background: #dcfce7; color: #166534; }
        .bg-broken { background: #fee2e2; color: #991b1b; }
        .bg-neutral { background: #f1f5f9; color: #475569; }
        .btn { padding: 12px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; border: none; transition: all 0.2s; font-size: 1rem; }
        .btn-primary { background: #000000; color: white; width: 100%; }
        .btn-camera { background: #000000; color: white; margin-bottom: 1rem; width: 100%; }
        .btn-primary:disabled { background: #94a3b8; cursor: not-allowed; }
        
        .overlay-badge {
            position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
            padding: 8px 16px; border-radius: 20px; font-weight: 600; color: white;
            backdrop-filter: blur(4px); z-index: 10; font-size: 0.9rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            text-align: center; white-space: nowrap;
        }

        .instruction-tip {
            position: absolute; 
            bottom: 60px; /* Positioned above the Ready/Status badge */
            left: 50%; 
            transform: translateX(-50%);
            background: rgba(0,0,0,0.6);
            color: rgba(255,255,255,0.9);
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 0.8rem;
            z-index: 10;
            white-space: nowrap;
            pointer-events: none;
        }

        .error-msg { color: #dc2626; background: #fef2f2; padding: 1rem; border-radius: 8px; border: 1px solid #fecaca; margin-top: 1rem; font-size: 0.875rem; }
      `}</style>

      {/* --- HISTORY DRAWER & OVERLAY --- */}
      <div className={`sidebar-overlay ${sidebarOpen ? 'open' : ''}`} onClick={() => setSidebarOpen(false)} />
      
      <div className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
           <h2 className="sidebar-title">Recent Scans</h2>
           <button className="close-btn" onClick={() => setSidebarOpen(false)}>×</button>
        </div>
        <div className="history-list">
          {history.length === 0 ? (
            <div style={{textAlign: 'center', color: '#cbd5e0', padding: '20px', fontSize: '0.9rem'}}>No history found</div>
          ) : (
            history.map((item, idx) => (
              <div key={idx} className="history-item" onClick={() => handleHistoryClick(item)}>
                <img src={item.url} alt="scan" className="thumb" />
                <div className="meta">
                   {/* UPDATED: Displays the exact time sent by backend (e.g., 6:00 pm) */}
                   <span className="meta-time">{item.timestamp}</span>
                   <div className="meta-subtitle">
                     <span>Rice Analysis</span>
                     {item.stats && <span>{item.stats.total} Grains</span>}
                   </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* FLOATING TOGGLE BUTTON */}
      <button className="history-toggle" onClick={() => setSidebarOpen(true)} title="View History">
        🕒
      </button>

      {/* --- EXISTING APP CONTENT --- */}
      <div className="app-content">
        <div className="header">
          <h1>RiceGuard AI</h1>
          <p>Automated grain analysis & classification</p>
        </div>

        <div className="grid">
          {/* LEFT COLUMN: Camera/Input */}
          <div className="card">
            {!preview && !isCameraOpen && (
              <div>
                <button className="btn btn-camera" onClick={startCamera}>Open Live Camera</button>
                <label className="upload-zone">
                  <input type="file" hidden onChange={handleFileChange} accept="image/*" />
                  <p style={{margin: 0, fontWeight: '500'}}>Upload from Gallery</p>
                  <p style={{margin: '0.5rem 0 0', fontSize: '0.8rem', color: '#94a3b8'}}>Tap to browse images</p>
                </label>
              </div>
            )}

            {isCameraOpen && (
              <div>
                {/* VIDEO CONTAINER (Tap to Focus) */}
                <div className="video-container" onClick={handleTapToFocus}>
                  <video ref={videoRef} autoPlay playsInline muted className="video-display" />
                  
                  {/* Visual Focus Box */}
                  {focusBox.visible && (
                      <div 
                          className="focus-box" 
                          style={{ left: focusBox.x, top: focusBox.y }}
                      />
                  )}

                  {/* 📏 INSTRUCTION TIP BOX */}
                  <div className="instruction-tip">
                      📏 Position camera 10-12cm from rice
                  </div>

                  <div className="overlay-badge" style={{ background: canCapture ? '#16a34a' : '#dc2626' }}>
                    {guidance}
                  </div>
                </div>
                
                <div style={{display: 'flex', gap: '0.5rem'}}>
                  <button className="btn btn-primary" onClick={capturePhoto} disabled={!canCapture}>
                    {canCapture ? "Capture Photo" : "Level Phone"}
                  </button>
                  <button className="btn" style={{background: '#e2e8f0', color: '#475569', width: 'auto'}} onClick={stopCamera}>
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {preview && (
              <div>
                <img src={displayImage} className="img-display" alt="Analysis Preview" />
                <div style={{display: 'flex', gap: '10px', flexDirection: 'column'}}>
                  <button className="btn btn-primary" onClick={handleAnalyze} disabled={loading}>
                    {loading ? 'Processing...' : 'Analyze Grain'}
                  </button>
                  <button className="btn" style={{background: '#e2e8f0', color: '#475569', width: '100%'}} onClick={handleReset}>
                    Reset
                  </button>
                </div>
              </div>
            )}
            
            <canvas ref={canvasRef} style={{display: 'none'}} />
            {error && <div className="error-msg">{error}</div>}
          </div>

          {/* RIGHT COLUMN: Results */}
          <div className="card">
            <h3 style={{ margin: '0 0 1.5rem', textAlign: 'center', color: '#1e293b' }}>Inspection Report</h3>
            
            {!results ? (
              <div style={{color: '#94a3b8', textAlign: 'center', padding: '4rem 0'}}>
                Analyze an image to generate report
              </div>
            ) : (
              <table className="stats-table">
                <thead>
                  <tr>
                    <th>Parameter</th>
                    <th style={{textAlign: 'right'}}>Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td className="label">Total Grains</td><td className="value">{results.total_grains}</td></tr>
                  <tr><td className="label">Whole Grains</td><td className="value"><span className="badge bg-good">{getPct(results.whole_grains)}</span></td></tr>
                  <tr><td className="label">Broken Grains</td><td className="value"><span className="badge bg-broken">{getPct(results.broken_grains)}</span></td></tr>
                  <tr><td className="label">Chalky Grains</td><td className="value"><span className="badge bg-neutral">{getPct(results.chalky_grains)}</span></td></tr>
                  <tr><td className="label">Foreign Matter</td><td className="value"><span className="badge bg-neutral">{getPct(results.foreign_matter)}</span></td></tr>
                  <tr><td className="label">Avg Width</td><td className="value">{results.avg_width_mm} mm</td></tr>
                  <tr><td className="label">Avg Length</td><td className="value">{results.avg_length_mm} mm</td></tr>
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiceGuardApp;