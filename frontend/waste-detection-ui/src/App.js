import React, { useState, useRef, useEffect } from "react";
import { Upload, Image, Recycle, Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { API_URL } from './config';

const modes = ["Image", "Video", "Webcam"];
const validImageTypes = ["image/jpeg", "image/jpg", "image/png", "image/bmp"];
const validVideoTypes = ["video/mp4", "video/avi"];

function App() {
  const [file, setFile] = useState(null);
  const [model, setModel] = useState("YOLOv10s");
  const [resultImage, setResultImage] = useState("");
  const [resultData, setResultData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [mode, setMode] = useState("Image");
  const [videoStream, setVideoStream] = useState(null);
  const [webcamActive, setWebcamActive] = useState(false);
  const [webcamUrl, setWebcamUrl] = useState("");
  const videoRef = useRef(null);

  useEffect(() => {
    if (mode === "Webcam") {
      startWebcam();
    } else {
      stopWebcam();
    }
    // eslint-disable-next-line
  }, [mode]);

  useEffect(() => {
    if (mode === "Video" && resultImage) {
      const video = document.querySelector('video');
      const loadingOverlay = document.getElementById('video-loading');
      
      if (video && loadingOverlay) {
        const hideLoading = () => {
          loadingOverlay.style.display = 'none';
        };
        
        const showLoading = () => {
          loadingOverlay.style.display = 'flex';
        };
        
        video.addEventListener('canplay', hideLoading);
        video.addEventListener('loadeddata', hideLoading);
        video.addEventListener('loadstart', showLoading);
        
        // Hide loading after 5 seconds as fallback
        const timeout = setTimeout(hideLoading, 5000);
        
        return () => {
          video.removeEventListener('canplay', hideLoading);
          video.removeEventListener('loadeddata', hideLoading);
          video.removeEventListener('loadstart', showLoading);
          clearTimeout(timeout);
        };
      }
    }
  }, [mode, resultImage]);

  const startWebcam = async () => {
    try {
      setError("");
      setWebcamActive(true);
      setWebcamUrl(`${API_URL}/predict_webcam?model=${model}`);
    } catch (err) {
      setError("Failed to start webcam");
      setWebcamActive(false);
    }
  };

  const stopWebcam = async () => {
    try {
      setWebcamActive(false);
      setWebcamUrl("");
      if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        setVideoStream(null);
      }
    } catch (err) {
      console.error("Error stopping webcam:", err);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const type = selectedFile.type;
      if (
        (mode === "Image" && validImageTypes.includes(type)) ||
        (mode === "Video" && validVideoTypes.includes(type))
      ) {
        setFile(selectedFile);
        setError("");
      } else {
        setError(`Please select a valid ${mode.toLowerCase()} file`);
        setFile(null);
      }
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      const type = droppedFile.type;
      if (
        (mode === "Image" && validImageTypes.includes(type)) ||
        (mode === "Video" && validVideoTypes.includes(type))
      ) {
        setFile(droppedFile);
        setError("");
      } else {
        setError(`Please select a valid ${mode.toLowerCase()} file`);
        setFile(null);
      }
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleUpload = async () => {
    if (!file && mode !== "Webcam") {
      setError("Please select a file first");
      return;
    }

    if (mode === "Webcam") {
      setWebcamUrl(`${API_URL}/predict_webcam?model=${model}`);
      return;
    }

    setLoading(true);
    setError("");
    setResultImage("");
    setResultData([]);

    try {
      const formData = new FormData();
      formData.append("model", model);
      let endpoint = "/predict";
      if (mode === "Image") {
        formData.append("file", file);
        endpoint = "/predict";
      } else if (mode === "Video") {
        formData.append("video", file);
        endpoint = "/predict_video";
      }
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      
      if (mode === "Video") {
        setResultImage(`${API_URL}${data.video_url}`);
        setResultData([{
          class_name: "Video Processing Complete",
          confidence: 1.0,
          message: data.message || "Video processed successfully",
          file_type: data.file_type || "video/mp4",
          file_extension: data.file_extension || ".mp4"
        }]);
      } else {
        setResultImage(`${API_URL}${data.image_url}`);
        setResultData(data.results || []);
      }
    } catch (err) {
      console.error("Upload error:", err);
      setError("Failed to process. Please check your input and try again.");
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return "bg-green-100 text-green-800 border-green-200";
    if (confidence >= 0.6) return "bg-yellow-100 text-yellow-800 border-yellow-200";
    return "bg-red-100 text-red-800 border-red-200";
  };

  return (
    <>
      <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet" />
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
      
      <style jsx>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        .font-sans {
          font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }
        
        .animate-fade-in {
          animation: fadeIn 0.5s ease-in-out;
        }
        
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>

      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 font-sans">
        {/* Header */}
        <div className="bg-white shadow-sm border-b border-gray-100">
          <div className="max-w-6xl mx-auto px-6 py-8">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <Recycle className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-3xl font-bold text-gray-900">Waste Object Detection</h1>
            </div>
            <p className="text-gray-600">Upload an image to detect and classify waste objects using advanced YOLO models</p>
          </div>
        </div>

        <div className="max-w-6xl mx-auto px-6 py-8">
          {/* Mode Switcher */}
          <div className="flex gap-3 mb-4">
            {modes.map((m) => (
              <button
                key={m}
                onClick={() => { setMode(m); setFile(null); setResultImage(""); setError(""); }}
                className={`px-4 py-2 rounded-lg font-medium border ${
                  mode === m
                    ? "bg-blue-600 text-white border-blue-600"
                    : "bg-white text-gray-700 border-gray-200 hover:border-blue-300"
                }`}
              >
                {m}
              </button>
            ))}
          </div>

          {/* Upload Section */}
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8 mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-6 flex items-center gap-2">
              <Upload className="w-5 h-5" />
              {mode === "Image" && "Upload Image"}
              {mode === "Video" && "Upload Video"}
              {mode === "Webcam" && "Live Camera Detection"}
            </h2>
            
            <div className="grid md:grid-cols-2 gap-8">
              {/* File/Video/Webcam Upload Area */}
              <div>
                {mode === "Webcam" ? (
                  <div className="border rounded-xl p-4">
                    {webcamActive && webcamUrl ? (
                      <div className="relative">
                        <img
                          src={webcamUrl}
                          alt="Real-time Detection"
                          className="rounded-xl w-full max-w-md mx-auto border"
                          onError={(e) => {
                            console.error("Webcam stream error");
                            setError("Failed to load webcam stream");
                          }}
                        />
                        <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
                          Live Detection
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-4 p-8">
                        <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center">
                          <Upload className="w-8 h-8 text-gray-400" />
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-medium text-gray-900">Webcam Not Active</p>
                          <p className="text-sm text-gray-500 mt-1">Click "Start Detection" to begin</p>
                        </div>
                      </div>
                    )}
                    <div className="mt-3 text-center">
                      <p className="text-sm text-gray-600">
                        {webcamActive ? "Real-time object detection active" : "Webcam preview (click to start detection)"}
                      </p>
                    </div>
                  </div>
                ) : (
                  <div
                    className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
                      dragOver
                        ? "border-blue-400 bg-blue-50"
                        : file
                        ? "border-green-400 bg-green-50"
                        : "border-gray-300 hover:border-gray-400 hover:bg-gray-50"
                    }`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                  >
                    <input
                      type="file"
                      accept={mode === "Image" ? "image/*" : mode === "Video" ? validVideoTypes.join(",") : undefined}
                      onChange={handleFileChange}
                      className="hidden"
                      id="file-upload"
                    />
                    <label htmlFor="file-upload" className="cursor-pointer">
                      <div className="flex flex-col items-center gap-4">
                        <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                          file ? "bg-green-100" : "bg-gray-100"
                        }`}>
                          {file ? (
                            <CheckCircle className="w-8 h-8 text-green-600" />
                          ) : (
                            mode === "Image" ? <Image className="w-8 h-8 text-gray-400" /> : <Upload className="w-8 h-8 text-gray-400" />
                          )}
                        </div>
                        <div>
                          <p className="text-lg font-medium text-gray-900">
                            {file ? `${mode} Selected` : `Choose a ${mode.toLowerCase()}`}
                          </p>
                          <p className="text-sm text-gray-500 mt-1">
                            {file ? file.name : `Drop your ${mode.toLowerCase()} here or click to browse`}
                          </p>
                          {file && (
                            <p className="text-xs text-gray-400 mt-1">
                              {(file.size / 1024 / 1024).toFixed(2)} MB
                            </p>
                          )}
                        </div>
                      </div>
                    </label>
                  </div>
                )}
              </div>

              {/* Model Selection and Upload Button */}
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-3">
                    Select Detection Model
                  </label>
                  <div className="grid grid-cols-2 gap-3">
                    {["YOLOv10s", "YOLOv10m", "YOLOv11n", "YOLOv12s"].map((modelOption) => (
                      <button
                        key={modelOption}
                        onClick={() => {
                          setModel(modelOption);
                          if (mode === "Webcam" && webcamActive) {
                            setWebcamUrl(`${API_URL}/predict_webcam?model=${modelOption}`);
                          }
                        }}
                        className={`p-3 rounded-lg border text-sm font-medium transition-all duration-200 ${
                          model === modelOption
                            ? "bg-blue-600 text-white border-blue-600 shadow-md"
                            : "bg-white text-gray-700 border-gray-200 hover:border-blue-300 hover:bg-blue-50"
                        }`}
                      >
                        {modelOption}
                      </button>
                    ))}
                  </div>
                </div>
                <button
                  onClick={handleUpload}
                  disabled={(!file && mode !== "Webcam") || loading}
                  className={`w-full py-4 px-6 rounded-xl font-semibold text-white transition-all duration-300 flex items-center justify-center gap-3 ${
                    (!file && mode !== "Webcam") || loading
                      ? "bg-gray-300 cursor-not-allowed"
                      : "bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 transform hover:scale-105 shadow-lg hover:shadow-xl"
                  }`}
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Recycle className="w-5 h-5" />
                      {mode === "Image" && "Detect Objects"}
                      {mode === "Video" && "Detect in Video"}
                      {mode === "Webcam" && (webcamActive ? "Detection Active" : "Start Detection")}
                    </>
                  )}
                </button>
                
                {/* Webcam Controls */}
                {mode === "Webcam" && (
                  <div className="flex gap-3">
                    <button
                      onClick={startWebcam}
                      disabled={webcamActive}
                      className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all duration-200 ${
                        webcamActive
                          ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                          : "bg-green-600 text-white hover:bg-green-700"
                      }`}
                    >
                      Start
                    </button>
                    <button
                      onClick={stopWebcam}
                      disabled={!webcamActive}
                      className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all duration-200 ${
                        !webcamActive
                          ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                          : "bg-red-600 text-white hover:bg-red-700"
                      }`}
                    >
                      Stop
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-8 animate-fade-in">
              <div className="flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-red-600" />
                <p className="text-red-800 font-medium">{error}</p>
              </div>
            </div>
          )}

          {/* Results Section */}
          {resultImage && (
            <div className="space-y-8 animate-fade-in">
              {/* Result Image/Video */}
              <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8">
                <h2 className="text-xl font-semibold text-gray-900 mb-6">
                  {mode === "Video" ? "Video Detection Results" : "Detection Results"}
                </h2>
                <div className="rounded-xl overflow-hidden border border-gray-200">
                  {mode === "Video" ? (
                    <div>
                      <div className="relative">
                        <video
                          controls
                          preload="metadata"
                          autoPlay
                          muted
                          playsInline
                          className="w-full h-auto rounded-lg"
                          onLoadStart={() => {
                            console.log("Video loading started");
                            setError(""); // Clear any previous errors
                          }}
                          onLoadedData={() => {
                            console.log("Video data loaded");
                            setError(""); // Clear any previous errors
                          }}
                          onCanPlay={() => {
                            console.log("Video can play");
                            setError(""); // Clear any previous errors
                          }}
                          onPlay={() => console.log("Video started playing")}
                          onPause={() => console.log("Video paused")}
                          onEnded={() => console.log("Video ended")}
                          onError={(e) => {
                            console.error("Video load error:", e);
                            console.error("Video error details:", e.target.error);
                            setError(`Video may not be playable in this browser. Try downloading it instead.`);
                          }}
                        >
                          <source src={resultImage} type="video/mp4" />
                          <source src={resultImage.replace('.mp4', '.avi')} type="video/avi" />
                          <source src={resultImage.replace('.avi', '.mp4')} type="video/mp4" />
                          Your browser does not support the video tag.
                        </video>
                        
                        {/* Video loading overlay */}
                        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-lg" id="video-loading">
                          <div className="text-white text-center">
                            <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2" />
                            <p>Loading video...</p>
                          </div>
                        </div>
                      </div>
                      
                      {/* Video controls and info */}
                      <div className="mt-4 space-y-3">
                        <div className="flex items-center justify-between text-sm text-gray-600">
                          <span>🎬 Processed video with object detection</span>
                          <span>
                            📹 {resultData[0]?.file_extension === '.mp4' ? 'MP4 (Browser Compatible)' : 'AVI (Download Recommended)'}
                          </span>
                        </div>
                        
                        {/* Browser compatibility notice */}
                        {resultData[0]?.file_extension === '.avi' && (
                          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                            <div className="flex items-center gap-2">
                              <span className="text-yellow-600">⚠️</span>
                              <span className="text-yellow-800 text-sm">
                                AVI format detected. Some browsers may not play this format. Use the download button below for guaranteed playback.
                              </span>
                            </div>
                          </div>
                        )}
                        
                        {/* Video playback controls */}
                        <div className="flex gap-2 justify-center">
                          <button
                            onClick={() => {
                              const video = document.querySelector('video');
                              if (video) {
                                video.currentTime = 0;
                                video.play();
                              }
                            }}
                            className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
                          >
                            🔄 Restart
                          </button>
                          <button
                            onClick={() => {
                              const video = document.querySelector('video');
                              if (video) {
                                video.playbackRate = video.playbackRate === 1 ? 2 : 1;
                              }
                            }}
                            className="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700"
                          >
                            ⚡ Speed
                          </button>
                        </div>
                      </div>
                      
                      {/* Download link as fallback */}
                      <div className="mt-4 text-center">
                        <a 
                          href={resultImage} 
                          download 
                          className="inline-flex items-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                        >
                          📥 Download Processed Video
                        </a>
                        <p className="text-sm text-gray-600 mt-2">
                          Download to play locally if browser playback doesn't work
                        </p>
                      </div>
                    </div>
                  ) : (
                    <img
                      src={resultImage}
                      alt="Detection Result"
                      className="w-full h-auto"
                      onError={(e) => {
                        console.error("Image load error");
                        setError("Failed to load result image");
                      }}
                    />
                  )}
                </div>
              </div>

              {/* Detected Objects */}
              {resultData.length > 0 && (
                <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8">
                  <h2 className="text-xl font-semibold text-gray-900 mb-6">
                    {mode === "Video" ? "Processing Results" : `Detected Objects (${resultData.length})`}
                  </h2>
                  
                  <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
                    {resultData.map((detection, index) => (
                      <div
                        key={index}
                        className="bg-gray-50 rounded-lg p-4 border border-gray-200 hover:shadow-md transition-shadow duration-200"
                      >
                        <div className="flex items-center justify-between mb-3">
                          <span className="font-medium text-gray-900">
                            {mode === "Video" ? "Status" : `Object ${index + 1}`}
                          </span>
                          <span
                            className={`px-3 py-1 rounded-full text-xs font-medium border ${getConfidenceColor(
                              detection.confidence
                            )}`}
                          >
                            {mode === "Video" ? "Complete" : `${(detection.confidence * 100).toFixed(1)}%`}
                          </span>
                        </div>
                        
                        <div className="space-y-2 text-sm">
                          <div>
                            <span className="text-gray-600">
                              {mode === "Video" ? "Message:" : "Class:"}
                            </span>
                            <span className="ml-2 font-medium text-gray-900">
                              {detection.class_name || detection.name || detection.message || 'Unknown'}
                            </span>
                          </div>
                          
                          {detection.bbox && mode !== "Video" && (
                            <div>
                              <span className="text-gray-600">Bounding Box:</span>
                              <div className="mt-1 text-xs text-gray-500 font-mono">
                                [{detection.bbox.map(coord => coord.toFixed(1)).join(', ')}]
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Raw Data Toggle */}
                  <details className="group">
                    <summary className="cursor-pointer text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors">
                      <span className="group-open:hidden">Show Raw Detection Data</span>
                      <span className="hidden group-open:inline">Hide Raw Detection Data</span>
                    </summary>
                    <div className="mt-4 bg-gray-900 rounded-lg p-4 overflow-auto">
                      <pre className="text-xs text-gray-300 whitespace-pre-wrap">
                        {JSON.stringify(resultData, null, 2)}
                      </pre>
                    </div>
                  </details>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default App;