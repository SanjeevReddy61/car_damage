let model;
let mediaRecorder, recordedChunks = [];
let isScanning = false;
let currentFacingMode = 'environment'; // Start with back camera

const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const uploadInput = document.getElementById('uploadVideo');
const downloadBtn = document.getElementById('downloadBtn');
const alertBox = document.getElementById('detection-alert');
const scanBtn = document.getElementById('scanBtn');

async function init() {
    try {
        // Force WebGL for mobile hardware acceleration
        await tf.setBackend('webgl'); 
        await tf.ready();
        
        model = await tflite.loadTFLiteModel('./models/best_float32.tflite');
        status.innerText = "AI ENGINE ACTIVE (GPU)";
        setupWebcam();
    } catch (err) {
        console.error(err);
        status.innerText = "ENGINE ERROR";
    }
}

async function setupWebcam() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: { ideal: 640 }, 
                height: { ideal: 480 },
                facingMode: currentFacingMode 
            },
            audio: false
        });
        video.srcObject = stream;
        sync();
    } catch (err) {
        console.error(err);
        status.innerText = "UPLOAD VIDEO";
    }
}

function switchCamera() {
    currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
    setupWebcam();
}

function toggleScan() {
    isScanning = !isScanning;
    if (isScanning) {
        scanBtn.innerHTML = '<i class="fas fa-stop"></i> STOP SCANNING';
        scanBtn.classList.add('btn-active');
    } else {
        scanBtn.innerHTML = '<i class="fas fa-play"></i> START SCANNING';
        scanBtn.classList.remove('btn-active');
        clearBlueprint();
        alertBox.classList.add('hidden');
    }
}

uploadInput.onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }
    video.src = URL.createObjectURL(file);
    sync();
    startRecording();
};

function sync() {
    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        video.play();
        process();
    };
}

function clearBlueprint() {
    document.querySelectorAll('.car-part').forEach(p => p.classList.remove('damage-detected'));
}

function updateMap(xNorm) {
    clearBlueprint();
    let parts = [];
    if (xNorm < 0.33) {
        parts = ['hood', 'front-bumper', 'left-headlight', 'right-headlight'];
    } else if (xNorm < 0.66) {
        parts = ['roof', 'front-left-door', 'front-right-door', 'rear-left-door', 'rear-right-door'];
    } else {
        parts = ['trunk', 'rear-bumper'];
    }

    parts.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.add('damage-detected');
    });

    document.getElementById('damage-summary').innerHTML = 
        `<p style="color:#ff0055;font-weight:bold">ALERTS: ${parts.length} AREAS FLAGGED</p>`;
}

async function process() {
    if (video.paused || video.ended) {
        if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
        return;
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const input = tf.browser
        .fromPixels(video)
        .resizeBilinear([640, 640])
        .div(255.0)
        .expandDims(0);

    const output = await model.predict(input);
    const data = output.dataSync();

    let found = false;
    const NUM = 8400;

    for (let i = 0; i < NUM; i++) {
        const conf = data[i + NUM * 4];
        if (conf > 0.25) { 
            const x = data[i] * canvas.width;
            const y = data[i + NUM] * canvas.height;
            const w = data[i + NUM * 2] * canvas.width;
            const h = data[i + NUM * 3] * canvas.height;

            ctx.strokeStyle = "#ff0055";
            ctx.lineWidth = 6;
            ctx.strokeRect(x - w / 2, y - h / 2, w, h);

            if (isScanning) {
                found = true;
                updateMap(data[i]);
            }
            break; 
        }
    }

    alertBox.classList.toggle('hidden', !found);
    tf.dispose([input, output]);
    requestAnimationFrame(process);
}

function startRecording() {
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(canvas.captureStream(30), { mimeType: 'video/webm' });
    mediaRecorder.ondataavailable = e => recordedChunks.push(e.data);
    mediaRecorder.onstop = () => {
        const url = URL.createObjectURL(new Blob(recordedChunks, { type: 'video/webm' }));
        downloadBtn.href = url;
        downloadBtn.download = `Report_${Date.now()}.webm`;
        downloadBtn.style.display = "inline-flex";
    };
    mediaRecorder.start();
}

init();