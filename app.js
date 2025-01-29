const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Webカメラ映像を取得
navigator.mediaDevices.getUserMedia({
    video: true
}).then(stream => {
    video.srcObject = stream;
    video.play();
}).catch(err => {
    console.error("Error accessing camera: " + err);
});

let session;

async function loadModel() {
    // YOLOv5モデル（ONNX）の読み込み
    session = await ort.InferenceSession.create('model.onnx');
    console.log("Model loaded.");
}

async function runDetection() {
    // Canvas上にビデオフレームを描画
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // 前処理：画像をテンソル形式に変換
    const input = preprocess(imageData);

    // 推論実行
    const feeds = { input: input };
    const results = await session.run(feeds);

    // 結果を処理
    postprocess(results);

    requestAnimationFrame(runDetection);
}

function preprocess(imageData) {
    let tensor = new Float32Array(imageData.data.length / 4 * 3);
    return new ort.Tensor('float32', tensor, [1, 3, canvas.height, canvas.width]);
}

function postprocess(results) {
    const boxes = results['output'][0];  // 結果の解析
    boxes.forEach(box => {
        const [x, y, width, height] = box;
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);
    });
}

video.onloadeddata = () => {
    loadModel().then(() => {
        runDetection();
    });
};
