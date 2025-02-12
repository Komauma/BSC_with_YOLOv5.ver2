// DOM要素の取得
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const feedback = document.getElementById('feedback');

// モデルとセッションの変数
let session;
let isDetectionRunning = false;

// ビデオの初期化
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.play();
    })
    .catch(err => {
        feedback.innerText = "Error accessing camera: " + err.message;
        feedback.style.color = "red";
    });

// GitHub Releases からモデルをロードする関数
async function loadModel() {
    feedback.innerText = "Loading model... Please wait.";
    const modelUrl = 'https://badminton-shuttle-checker.web.app/best.onnx'; // 正しいURLに変更

    try {
        // best.onnx をダウンロード
        const response = await fetch(modelUrl);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        // ArrayBuffer に変換
        const arrayBuffer = await response.arrayBuffer();

        // ONNX Runtime にロード
        session = await ort.InferenceSession.create(arrayBuffer);
        feedback.innerText = "Model loaded. Click 'Start Detection' to begin.";
    } catch (err) {
        feedback.innerText = "Failed to load model: " + err.message;
        feedback.style.color = "red";
    }
}

// 検出を開始する関数
async function startDetection() {
    if (!session) {
        feedback.innerText = "Model is not loaded yet!";
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
            feedback.innerText = "Error during detection: " + err.message;
            feedback.style.color = "red";
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

    // 画像のピクセルデータを正規化し、3チャンネル（RGB）に変換
    for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
        tensor[j] = data[i] / 255;       // R
        tensor[j + 1] = data[i + 1] / 255; // G
        tensor[j + 2] = data[i + 2] / 255; // B
    }

    // テンソルを返す（[1, 3, height, width]の形に変換）
    return new ort.Tensor('float32', tensor, [1, 3, height, width]);
}

// 結果を処理してCanvasに描画
function postprocess(results) {
    const boxes = results['output'][0];  // モデルの出力に依存
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;

    // 各バウンディングボックスを描画
    boxes.forEach(box => {
        const [x, y, width, height] = box;
        ctx.strokeRect(x, y, width, height);
    });
}

// ページ読み込み時にモデルをロード
loadModel();
