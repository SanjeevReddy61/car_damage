const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const uploadInput = document.getElementById('uploadVideo');
const downloadBtn = document.getElementById('downloadBtn');

let model;
let isModelReady = false;
let mediaRecorder;
let recordedChunks = [];

// --- 1. Load Local Model ---
async function loadModel() {
    try {
        await tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/');
        model = await tflite.loadTFLiteModel('./models/best_float32.tflite');
        isModelReady = true;
        status.innerText = "Ready! Upload a video to detect damages.";
    } catch (e) {
        status.innerText = "AI Load Error. Check /models/ folder.";
    }
}

// --- 2. Handle File Upload ---
uploadInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    video.src = URL.createObjectURL(file);
    video.onloadedmetadata = () => {
        // Set canvas to match the video's actual resolution
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        startRecording(); // Prepare the recorder
        video.play();
        processFrame(); // Start frame-by-frame analysis
    };
});

// --- 3. The Frame-by-Frame AI Loop ---
async function processFrame() {
    if (video.paused || video.ended) {
        stopRecording();
        return;
    }

    // 1. Run AI Inference
    const result = await tf.tidy(() => {
        const img = tf.browser.fromPixels(video);
        const resized = tf.image.resizeBilinear(img, [640, 640]);
        return model.predict(resized.expandDims(0).div(255.0));
    });

    // 2. Draw the original video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 3. Draw Red Boxes on top
    drawDetections(result);

    // Continue to next frame
    requestAnimationFrame(processFrame);
}

function drawDetections(result) {
    const data = result.dataSync();
    const numDetections = 8400; // YOLOv8/10 Output format

    for (let i = 0; i < numDetections; i++) {
        let score = data[i + numDetections * 4];
        if (score > 0.45) {
            let x = data[i] * canvas.width;
            let y = data[i + numDetections] * canvas.height;
            let w = data[i + numDetections * 2] * canvas.width;
            let h = data[i + numDetections * 3] * canvas.height;

            ctx.strokeStyle = "red";
            ctx.lineWidth = 5;
            ctx.strokeRect(x - w / 2, y - h / 2, w, h);
            
            ctx.fillStyle = "red";
            ctx.font = "bold 20px Arial";
            ctx.fillText(`DAMAGE ${Math.round(score * 100)}%`, x - w / 2, y - h / 2 - 10);
        }
    }
}

// --- 4. Video Recording Logic ---
function startRecording() {
    recordedChunks = [];
    const stream = canvas.captureStream(30); // Capture 30 FPS from canvas
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    
    mediaRecorder.ondataavailable = (e) => recordedChunks.push(e.data);
    mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        downloadBtn.href = URL.createObjectURL(blob);
        downloadBtn.download = `Damage_Report_${Date.now()}.webm`;
        downloadBtn.style.display = "inline-block";
        status.innerText = "Scan Complete! Click Download.";
    };
    mediaRecorder.start();
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
}

loadModel();