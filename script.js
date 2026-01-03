let model;
let mediaRecorder;
let recordedChunks = [];
const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const uploadInput = document.getElementById('uploadVideo');
const downloadBtn = document.getElementById('downloadBtn');

// 1. Initialize AI Model
async function init() {
    try {
        // Load the TFLite model from your local path
        model = await tflite.loadTFLiteModel('./models/best_float32.tflite');
        status.innerText = "AI Ready! Select Video or Start Camera.";
        setupWebcam();
    } catch (err) {
        console.error("Model failed to load:", err);
        status.innerText = "Error loading AI Engine.";
    }
}

// 2. Setup Webcam (Default)
async function setupWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false
    });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        video.play();
        processLoop();
    };
}

// 3. Handle Video Upload
uploadInput.onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Stop webcam if it's running
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }

    video.src = URL.createObjectURL(file);
    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        startRecording(); // Start capturing the canvas
        video.play();
        processLoop();
    };
};

// 4. Recording Logic
function startRecording() {
    recordedChunks = [];
    
    // 1. Define the MP4 options
    const options = {
        mimeType: 'video/mp4; codecs="avc1.424028, mp4a.40.2"', // Standard MP4 codec
        videoBitsPerSecond: 2500000 // 2.5 Mbps for clear quality
    };

    // 2. Check if the browser supports this specific MP4 type
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        console.warn("Direct MP4 not supported, falling back to WebM");
        options.mimeType = 'video/webm'; 
    }

    const stream = canvas.captureStream(30);
    mediaRecorder = new MediaRecorder(stream, options);

    mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
        // 3. Create the Blob as an MP4
        const blob = new Blob(recordedChunks, { type: options.mimeType });
        const url = URL.createObjectURL(blob);
        
        downloadBtn.href = url;
        // Ensure the extension is .mp4
        downloadBtn.download = `Damage_Report_${Date.now()}.mp4`; 
        downloadBtn.style.display = 'inline-block';
    };

    mediaRecorder.start();
}

// 5. Unified AI Processing Loop
async function processLoop() {
    if (video.paused || video.ended) {
        if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
        return;
    }

    // Start memory scope
    tf.engine().startScope();

    // Pre-processing
    const img = tf.browser.fromPixels(video);
    const resized = tf.image.resizeBilinear(img, [640, 640]);
    const expanded = resized.expandDims(0);
    const normalized = expanded.div(255.0); // Normalization

    // Run Inference
    const result = await model.predict(normalized);
    const data = result.dataSync();

    // Draw frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Draw AI boxes (YOLO11 structure)
    drawBoxes(data);

    tf.engine().endScope();
    requestAnimationFrame(processLoop);
}

function drawBoxes(data) {
    const numDets = 8400; 
    let found = false;

    for (let i = 0; i < numDets; i++) {
        let score = data[i + numDets * 4]; 
        if (score > 0.45) {
            found = true;
            let x_c = data[i] * canvas.width;
            let y_c = data[i + numDets] * canvas.height;
            let w = data[i + numDets * 2] * canvas.width;
            let h = data[i + numDets * 3] * canvas.height;

            ctx.strokeStyle = "#FF0000";
            ctx.lineWidth = 3;
            ctx.strokeRect(x_c - w/2, y_c - h/2, w, h);
        }
    }
    status.innerText = found ? "⚠️ DAMAGE DETECTED" : "Scanning...";
}

init();