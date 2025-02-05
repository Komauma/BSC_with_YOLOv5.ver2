// DOM要素の取得
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const feedback = document.getElementById('feedback');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');

// モデルとセッションの変数
let session;
let isDetectionRunning = false;

// ビデオの初期化
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.play();
    })
    .catch(err => handleError("Error accessing camera: " + err.message));

// モデルをロード
async function loadModel() {
    feedback.innerText = "Loading model... Please wait.";
    startButton.disabled = true;
    try {
        session = await ort.InferenceSession.create('https://drive.google.com/uc?export=download&id=1s8eJIHE_6es5kb41MqCSazMEbFXIFsXQ');
        feedback.innerText = "Model loaded. Click 'Start Detection' to begin.";
        startButton.disabled = false;
    } catch (err) {
        handleError("Failed to load model: " + err.message);
    }
}

// 検出を開始する関数
async function startDetection() {
    if (!session) {
        handleError("Model is not loaded yet!");
        return;
    }

    isDetectionRunning = true;
    feedback.innerText = "Detection running...";

    const run = async () => {
        if (!isDetectionRunning) return;

        // 映像をCanvasに描画
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        // 前処理：画像をテンソルに変換
        const inputTensor = preprocess(imageData);

        // 推論を実行
        try {
            const results = await session.run({ input: inputTensor });
            postprocess(results);
        } catch (err) {
            handleError("Error during detection: " + err.message);
        }

        // 次のフレームをリクエスト
        if (isDetectionRunning) {
            requestAnimationFrame(run);
        }
    };

    run();
}

// 検出を停止する関数
function stopDetection() {
    isDetectionRunning = false;
    feedback.innerText = "Detection stopped. Click 'Start Detection' to resume.";
}

// 画像の前処理
function preprocess(imageData) {
    const { data, width, height } = imageData;
    const tensor = new Float32Array(width * height * 3);

    for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
        tensor[j] = data[i] / 255;
        tensor[j + 1] = data[i + 1] / 255;
        tensor[j + 2] = data[i + 2] / 255;
    }

    return new ort.Tensor('float32', tensor, [1, 3, height, width]);
}

// 結果を処理してCanvasに描画
function postprocess(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.font = '16px Arial';
    ctx.fillStyle = 'red';

    const boxes = results['output'][0];
    boxes.forEach(box => {
        const [x, y, width, height] = box;
        ctx.strokeRect(x, y, width, height);
        ctx.fillText('Shuttle', x, y - 5);
    });
}

// エラーメッセージの処理
function handleError(message) {
    feedback.innerText = message;
    feedback.style.color = "red";
    console.error(message);
}

// ページ読み込み時にモデルをロード
loadModel();
