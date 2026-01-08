/*************************************************
 * CONFIG
 *************************************************/
const INPUT_SIZE = 640;
const CAR_CONF = 0.45;
const DAMAGE_CONF = 0.45;
const MIN_DAMAGE_AREA = 0.002;

let carModel, damageModel;
let running = false;
let mediaRecorder = null;
let recordedChunks = [];
let isUploadMode = false;

/*************************************************
 * DOM
 *************************************************/
const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");
const uploadInput = document.getElementById("uploadVideo");
const alertBox = document.getElementById("detection-alert");
const downloadBtn = document.getElementById("downloadBtn");

/*************************************************
 * BLUEPRINT
 *************************************************/
function resetBlueprint() {
    document.querySelectorAll(".car-part")
        .forEach(p => p.classList.remove("damage-detected"));
}

function mark(...ids) {
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.add("damage-detected");
    });
}

function updateBlueprint(xNorm, yNorm) {
    resetBlueprint();

    if (yNorm < 0.2) mark("rear-bumper");
    else if (yNorm < 0.35) mark("trunk");
    else if (yNorm < 0.5) xNorm < 0.5 ? mark("rear-left-door") : mark("rear-right-door");
    else if (yNorm < 0.65) mark("roof");
    else if (yNorm < 0.8) xNorm < 0.5 ? mark("front-left-door") : mark("front-right-door");
    else if (yNorm < 0.9) mark("hood");
    else mark("front-bumper");

    const count = document.querySelectorAll(".damage-detected").length;
    document.getElementById("damage-summary").innerHTML =
        `<p style="color:#ff0055;font-weight:bold">
            ALERTS: ${count} PANELS IMPACTED
        </p>`;
}

/*************************************************
 * INIT
 *************************************************/
async function init() {
    status.innerText = "LOADING MODELS...";
    carModel = await tflite.loadTFLiteModel("./models/car_float32.tflite");
    damageModel = await tflite.loadTFLiteModel("./models/best1_float32.tflite");
    status.innerText = "AI ENGINE ACTIVE (2-STAGE YOLOv11)";
    setupCamera();
}

/*************************************************
 * CAMERA (LIVE – NO RECORDING)
 *************************************************/
async function setupCamera() {
    try {
        isUploadMode = false;
        disableDownload();

        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: "environment" } },
            audio: false
        });

        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            resizeCanvas();
            running = true;
            detectLoop();
        };
    } catch {
        status.innerText = "UPLOAD VIDEO";
    }
}

/*************************************************
 * VIDEO UPLOAD (RECORD ENABLED)
 *************************************************/
uploadInput.onchange = e => {
    const file = e.target.files[0];
    if (!file) return;

    isUploadMode = true;
    enableDownload();

    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }

    video.src = URL.createObjectURL(file);
    video.onloadedmetadata = () => {
        video.play();
        resizeCanvas();
        startRecording();
        running = true;
        detectLoop();
    };
};

/*************************************************
 * RECORDING
 *************************************************/
function startRecording() {
    recordedChunks = [];
    const stream = canvas.captureStream(30);

    mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });

    mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: "video/webm" });
        downloadBtn.href = URL.createObjectURL(blob);
        downloadBtn.download = `inspection_${Date.now()}.webm`;
    };

    mediaRecorder.start();
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
}

/*************************************************
 * DOWNLOAD BUTTON CONTROL
 *************************************************/
function enableDownload() {
    downloadBtn.style.opacity = "1";
    downloadBtn.style.pointerEvents = "auto";
}

function disableDownload() {
    downloadBtn.style.opacity = "0.4";
    downloadBtn.style.pointerEvents = "none";
}

/*************************************************
 * CANVAS
 *************************************************/
function resizeCanvas() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

/*************************************************
 * MAIN LOOP (UNCHANGED)
 *************************************************/
async function detectLoop() {
    if (!running || video.paused || video.ended) {
        if (isUploadMode) stopRecording();
        return;
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const frame = tf.tidy(() =>
        tf.browser.fromPixels(video)
            .resizeBilinear([INPUT_SIZE, INPUT_SIZE])
            .div(255)
            .expandDims(0)
    );

    const carOut = await carModel.predict(frame);
    const carData = carOut.dataSync();
    tf.dispose(carOut);

    const carBox = findBestCar(carData);

    if (carBox) {
        drawBox(carBox, "#00f2ff");

        const crop = cropCar(frame, carBox);
        const dmgOut = await damageModel.predict(crop);
        const dmgData = dmgOut.dataSync();
        tf.dispose([crop, dmgOut]);

        const damaged = processDamage(dmgData, carBox);
        alertBox.classList.toggle("hidden", !damaged);
    } else {
        alertBox.classList.add("hidden");
        resetBlueprint();
    }

    tf.dispose(frame);
    requestAnimationFrame(detectLoop);
}

/*************************************************
 * HELPERS (UNCHANGED)
 *************************************************/
function findBestCar(data) {
    const N = 8400, attrs = data.length / N;
    let best = null, bestScore = 0;

    for (let i = 0; i < N; i++) {
        let maxCls = 0;
        for (let c = 4; c < attrs; c++) maxCls = Math.max(maxCls, data[i + N * c]);
        if (maxCls > CAR_CONF && maxCls > bestScore) {
            bestScore = maxCls;
            best = { x: data[i], y: data[i + N], w: data[i + N * 2], h: data[i + N * 3] };
        }
    }
    return best;
}

function processDamage(data, car) {
    const N = 8400, attrs = data.length / N;

    for (let i = 0; i < N; i++) {
        let maxCls = 0;
        for (let c = 4; c < attrs; c++) maxCls = Math.max(maxCls, data[i + N * c]);
        if (maxCls < DAMAGE_CONF) continue;

        const w = data[i + N * 2], h = data[i + N * 3];
        if (w * h < MIN_DAMAGE_AREA) continue;

        const x = car.x + (data[i] - 0.5) * car.w;
        const y = car.y + (data[i + N] - 0.5) * car.h;

        drawBox({ x, y, w: w * car.w, h: h * car.h }, "#ff0055");
        updateBlueprint(x, y);
        return true;
    }
    return false;
}

function cropCar(frame, car) {
    return tf.image.cropAndResize(
        frame,
        [[car.y - car.h / 2, car.x - car.w / 2, car.y + car.h / 2, car.x + car.w / 2]],
        [0],
        [INPUT_SIZE, INPUT_SIZE]
    );
}

function drawBox(b, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(
        (b.x - b.w / 2) * canvas.width,
        (b.y - b.h / 2) * canvas.height,
        b.w * canvas.width,
        b.h * canvas.height
    );
}

/*************************************************
 * START
 *************************************************/
init();
