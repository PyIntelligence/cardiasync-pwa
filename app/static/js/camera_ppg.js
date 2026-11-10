let stream = null;
let videoElement = document.createElement('video');
let canvas = document.createElement('canvas');
let ctx = canvas.getContext('2d');
let ppgData = [];

// ==========================================
// 1. FLUXO DE CALIBRAÇÃO (Uso em calibrate.html)
// ==========================================

async function startCalibrationCapture() {
    const statusEl = document.getElementById("captureStatus");
    statusEl.innerText = "Status: Posicione o dedo sobre a câmera com o Flash ligado...";
    
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { exact: "environment" },
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 }
            }
        });

        const track = stream.getVideoTracks()[0];
        const capabilities = track.getCapabilities();
        if (capabilities.torch) {
            await track.applyConstraints({ advanced: [{ torch: true }] });
        }

        videoElement.srcObject = stream;
        await videoElement.play();
        ppgData = [];
        processCalibrationFrame();
    } catch (err) {
        console.error("Erro na câmera:", err);
        statusEl.innerText = "Erro: Acesso à câmera ou flash negado/indisponível.";
    }
}

function processCalibrationFrame() {
    if (!stream) return;

    canvas.width = 100;
    canvas.height = 100;
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = frame.data;
    
    let redSum = 0;
    for (let i = 0; i < pixels.length; i += 4) {
        redSum += pixels[i]; // Canal vermelho (Variação de volume sanguíneo)
    }
    
    const avgRed = redSum / (pixels.length / 4);
    ppgData.push(avgRed);

    if (ppgData.length < 300) { // ~10 segundos a 30fps
        requestAnimationFrame(processCalibrationFrame);
    } else {
        stopCamera();
        submitCalibration();
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

async function submitCalibration() {
    const sysRef = document.getElementById("sysRef").value;
    const diaRef = document.getElementById("diaRef").value;
    const statusEl = document.getElementById("captureStatus");

    if (!sysRef || !diaRef) {
        alert("Preencha os valores do esfigmomanômetro!");
        return;
    }

    // Calcula features simples do vetor ppgData
    const meanVal = ppgData.reduce((a, b) => a + b, 0) / ppgData.length;
    const maxVal = Math.max(...ppgData);
    const minVal = Math.min(...ppgData);
    const amplitude = maxVal - minVal;

    const mockSignalFeatures = {
        mean_amplitude: parseFloat(amplitude.toFixed(2)),
        mean_intensity: parseFloat(meanVal.toFixed(2)),
        heart_rate: 75
    };

    const response = await fetch('/api/calibrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            user_id: 'usuario_demo',
            systolic_ref: parseInt(sysRef),
            diastolic_ref: parseInt(diaRef),
            signal_features: mockSignalFeatures
        })
    });

    const result = await response.json();
    if (result.success) {
        statusEl.innerText = "Sucesso! Calibração salva e vinculada.";
    } else {
        statusEl.innerText = "Erro ao salvar calibração.";
    }
}

// ==========================================================
// 2. FUNÇÃO REUTILIZÁVEL DE MEDIÇÃO (Uso em measure.html)
// ==========================================================

function capturePPGSignal(seconds = 10) {
    return new Promise(async (resolve, reject) => {
        let localStream = null;
        let localVideo = document.createElement('video');
        let localCanvas = document.createElement('canvas');
        let localCtx = localCanvas.getContext('2d');
        let intensities = [];
        const totalFrames = seconds * 30; // 30 FPS * N segundos

        try {
            localStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { exact: "environment" },
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 }
                }
            });

            const track = localStream.getVideoTracks()[0];
            const capabilities = track.getCapabilities();
            if (capabilities.torch) {
                await track.applyConstraints({ advanced: [{ torch: true }] });
            }

            localVideo.srcObject = localStream;
            await localVideo.play();

            function readMeasureFrame() {
                if (intensities.length >= totalFrames) {
                    localStream.getTracks().forEach(t => t.stop());
                    resolve(intensities);
                    return;
                }

                localCanvas.width = 100;
                localCanvas.height = 100;
                localCtx.drawImage(localVideo, 0, 0, localCanvas.width, localCanvas.height);
                const frameData = localCtx.getImageData(0, 0, localCanvas.width, localCanvas.height);
                const pixels = frameData.data;

                let redSum = 0;
                for (let i = 0; i < pixels.length; i += 4) {
                    redSum += pixels[i];
                }
                intensities.push(redSum / (pixels.length / 4));

                requestAnimationFrame(readMeasureFrame);
            }

            readMeasureFrame();
        } catch (err) {
            if (localStream) {
                localStream.getTracks().forEach(t => t.stop());
            }
            reject(err);
        }
    });
}