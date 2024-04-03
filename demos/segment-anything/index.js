// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run segment-anything with webgpu and webnn in onnxruntime-web.
//

// the image size on canvas
const MAX_WIDTH = 500;
const MAX_HEIGHT = 500;

// the image size supported by the model
const MODEL_WIDTH = 1024;
const MODEL_HEIGHT = 1024;

const MODELS = {
    sam_b: [
        {
            name: "SAM ViT-B Encoder (FP16)",
            url: "sam_vit_b_01ec64.encoder-fp16.onnx",
            size: '171MB',
        },
        {
            name: "SAM ViT-B Decoder",
            url: "sam_vit_b_01ec64.decoder.onnx",
            size: "15.7MB",
        },
    ],
    sam_b_int8: [
        {
            name: "SAM ViT-B Encoder (INT8)",
            url: "sam_vit_b-encoder-int8.onnx",
            size: "95.6MB",
        },
        {
            name: "SAM ViT-B Decoder (INT8)",
            url: "sam_vit_b-decoder-int8.onnx",
            size: "4.52MB",
        },
    ],
};

const config = getConfig();

let canvas;
let filein;
let decoder_latency;

var image_embeddings;
var points = [];
var labels = [];
var imageImageData;
var isClicked = false;
var maskImageData;

function log(i) { 
    console.log(i); 
    document.getElementById('status').innerText += `\n[${getDateTime()}] ${i}`;
}

/**
 * create config from url
 */
function getConfig() {
    const query = window.location.search.substring(1);
    var config = {
        host: location.href.includes("github.io") ? "https://huggingface.co/onnxruntime-web-temp/demo/resolve/main/segment-anything" : "models",
        model: "sam_b",
        provider: "webnn",
        device: "gpu",
        threads: "1",
    };
    let vars = query.split("&");
    for (var i = 0; i < vars.length; i++) {
        let pair = vars[i].split("=");
        if (pair[0] in config) {
            config[pair[0]] = decodeURIComponent(pair[1]);
        } else if (pair[0].length > 0) {
            throw new Error("unknown argument: " + pair[0]);
        }
    }
    config.threads = parseInt(config.threads);
    config.local = parseInt(config.local);
    return config;
}

/**
 * clone tensor
 */
function cloneTensor(t) {
    return new ort.Tensor(t.type, Float32Array.from(t.data), t.dims);
}

/*
 * create feed for the original facebook model
 */
function feedForSam(emb, points, labels) {
    const maskInput = new ort.Tensor(new Float32Array(256 * 256), [1, 1, 256, 256]);
    const hasMask = new ort.Tensor(new Float32Array([0]), [1,]);
    const origianlImageSize = new ort.Tensor(new Float32Array([MODEL_HEIGHT, MODEL_WIDTH]), [2,]);
    const pointCoords = new ort.Tensor(new Float32Array(points), [1, points.length / 2, 2]);
    const pointLabels = new ort.Tensor(new Float32Array(labels), [1, labels.length]);

    return {
        "image_embeddings": cloneTensor(emb.image_embeddings),
        "point_coords": pointCoords,
        "point_labels": pointLabels,
        "mask_input": maskInput,
        "has_mask_input": hasMask,
        "orig_im_size": origianlImageSize
    }
}

/*
 * Handle cut-out event
 */
async function handleCut(event) {
    if (points.length == 0) {
        return;
    }

    const [w, h] = [canvas.width, canvas.height];

    // canvas for cut-out
    const cutCanvas = new OffscreenCanvas(w, h);
    const cutContext = cutCanvas.getContext('2d');
    const cutPixelData = cutContext.getImageData(0, 0, w, h);

    // need to rescale mask to image size
    const maskCanvas = new OffscreenCanvas(w, h);
    const maskContext = maskCanvas.getContext('2d');
    maskContext.drawImage(await createImageBitmap(maskImageData), 0, 0);
    const maskPixelData = maskContext.getImageData(0, 0, w, h);

    // copy masked pixels to cut-out
    for (let i = 0; i < maskPixelData.data.length; i += 4) {
        if (maskPixelData.data[i] > 0) {
            for (let j = 0; j < 4; ++j) {
                const offset = i + j;
                cutPixelData.data[offset] = imageImageData.data[offset];
            }
        }
    }
    cutContext.putImageData(cutPixelData, 0, 0);

    // Download image 
    const link = document.createElement('a');
    link.download = 'image.png';
    link.href = URL.createObjectURL(await cutCanvas.convertToBlob());
    link.click();
    link.remove();
}

async function decoder(points, labels) {
    let ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = imageImageData.width;
    canvas.height = imageImageData.height;
    ctx.putImageData(imageImageData, 0, 0);

    if (points.length > 0) {
        // need to wait for encoder to be ready
        if (image_embeddings === undefined) {
            await MODELS[config.model][0].sess;
        }

        // wait for encoder to deliver embeddings
        const emb = await image_embeddings;

        // the decoder
        const session = MODELS[config.model][1].sess;

        const feed = feedForSam(emb, points, labels);
        const start = performance.now();
        const res = await session.run(feed);
        decoder_latency.innerText = `${(performance.now() - start).toFixed(1)}`;

        for (let i = 0; i < points.length; i += 2) {
            ctx.fillStyle = 'blue';
            ctx.fillRect(points[i], points[i + 1], 10, 10);
        }
        const mask = res.masks;
        maskImageData = mask.toImageData();
        ctx.globalAlpha = 0.3;
        ctx.drawImage(await createImageBitmap(maskImageData), 0, 0);
    }
}

function getPoint(event) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.trunc(event.clientX - rect.left);
    const y = Math.trunc(event.clientY - rect.top);
    return [x, y];
}

/**
 * handler mouse move event
 */
async function handleMouseMove(event) {
    if (isClicked) {
        return;
    }
    try {
        isClicked = true;
        canvas.style.cursor = "wait";
        const point = getPoint(event);
        await decoder([...points, point[0], point[1]], [...labels, 1]);
    }
    finally {
        canvas.style.cursor = "default";
        isClicked = false;
    }
}

/**
 * handler to handle click event on canvas
 */
async function handleClick(event) {
    if (isClicked) {
        return;
    }
    try {
        isClicked = true;
        canvas.style.cursor = "wait";

        const point = getPoint(event);
        const label = 1;
        points.push(point[0]);
        points.push(point[1]);
        labels.push(label);
        await decoder(points, labels);
    }
    finally {
        canvas.style.cursor = "default";
        isClicked = false;
    }
}

/**
 * handler called when image available
 */
async function handleImage(img) {
    points = [];
    labels = [];
    filein.disabled = true;
    decoder_latency.innerText = "";
    canvas.style.cursor = "wait";
    image_embeddings = undefined;

    let width = img.width;
    let height = img.height;
    if (width > height) {
        if (width > MAX_WIDTH) {
            height = height * (MAX_WIDTH / width);
            width = MAX_WIDTH;
        }
    } else {
        if (height > MAX_HEIGHT) {
            width = width * (MAX_HEIGHT / height);
            height = MAX_HEIGHT;
        }
    }
    width = Math.round(width);
    height = Math.round(height);
    canvas.width = width;
    canvas.height = height;
    
    var ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, width, height);

    imageImageData = ctx.getImageData(0, 0, width, height);

    const t = await ort.Tensor.fromImage(imageImageData, { resizedWidth: MODEL_WIDTH, resizedHeight: MODEL_HEIGHT });
    const feed = (config.isSlimSam) ? { "pixel_values": t } : { "input_image": t };
    console.log(MODELS[config.model]);
    const session = await MODELS[config.model][0].sess;

    const start = performance.now();
    image_embeddings = session.run(feed);
    image_embeddings.then(() => {
        log(`[Session Run] Encoder execution time: ${(performance.now() - start).toFixed(2)}ms`);
        log(`[Session Run] Ready to segment image`);
        log(`[Session Run] Please move the mouse to a random spot of the image`);
        canvas.style.cursor = "default";
    });
    filein.disabled = false;
}

// Get model via Origin Private File System
async function getModelOPFS(name, url, updateModel) {
    const root = await navigator.storage.getDirectory();
    let fileHandle;

    async function updateFile() {
        const response = await fetch(url);
        const buffer = await readResponse(name, response);
        fileHandle = await root.getFileHandle(name, { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(buffer);
        await writable.close();
        return buffer;
    }

    if (updateModel) {
        return await updateFile();
    }

    try {
        fileHandle = await root.getFileHandle(name);
        const blob = await fileHandle.getFile();
        let buffer = await blob.arrayBuffer();
        if (buffer) {
            // if (name == 'text_encoder') {
            //     textEncoderFetchProgress = 20.00;
            // } else if (name == 'unet') {
            //     unetFetchProgress = 50.00;
            // } else if (name == 'vae_decoder') {
            //     vaeDecoderFetchProgress = 8.00;
            // }

            // progress = textEncoderFetchProgress + unetFetchProgress + vaeDecoderFetchProgress + textEncoderCompileProgress + unetCompileProgress + vaeDecoderCompileProgress;
            // updateLoadWave(progress.toFixed(2));
            return buffer;
        }

    } catch (e) {
        return await updateFile();
    }
}

async function readResponse(name, response) {
    const contentLength = response.headers.get('Content-Length');
    let total = parseInt(contentLength ?? '0');
    let buffer = new Uint8Array(total);
    let loaded = 0;

    const reader = response.body.getReader();
    async function read() {
        const { done, value } = await reader.read();
        if (done) return;

        let newLoaded = loaded + value.length;
        fetchProgress = (newLoaded / contentLength) * 100;

        // if (name == 'text_encoder') {
        //     textEncoderFetchProgress = 0.20 * fetchProgress;
        // } else if (name == 'unet') {
        //     unetFetchProgress = 0.50 * fetchProgress;
        // } else if (name == 'vae_decoder') {
        //     vaeDecoderFetchProgress = 0.08 * fetchProgress;
        // }

        // progress = textEncoderFetchProgress + unetFetchProgress + vaeDecoderFetchProgress + textEncoderCompileProgress + unetCompileProgress + vaeDecoderCompileProgress;

        // updateLoadWave(progress.toFixed(2));

        if (newLoaded > total) {
            total = newLoaded;
            let newBuffer = new Uint8Array(total);
            newBuffer.set(buffer);
            buffer = newBuffer;
        }
        buffer.set(value, loaded);
        loaded = newLoaded;
        return read();
    }

    await read();
    return buffer;
}

/*
 * load models one at a time
 */
async function load_models(models) {
    log("[Load] ONNX Runtime Execution Provider: " + config.provider);
 
    for (const [id, model] of Object.entries(models)) {
        let start;
        try {
            let name = models[id].name;
            start = performance.now();
            const opt = {
                executionProviders: [config.provider],
                enableMemPattern: false,
                enableCpuMemArena: false,
                extra: {
                    session: {
                        disable_prepacking: "1",
                        use_device_allocator_for_initializers: "1",
                        use_ort_model_bytes_directly: "1",
                        use_ort_model_bytes_for_initializers: "1"
                    }
                },
            };
            // sam-b-encoder for WebNN is slow, as which contains 24 Einsum nodes,
            // WebNN EP is working on Einsum op implementation at https://github.com/microsoft/onnxruntime/pull/19558.
            if (config.provider == 'webnn') {
                opt.executionProviders = [{
                    name: "webnn",
                    deviceType: "gpu",
                }];
                opt.freeDimensionOverrides = {
                    num_points: 1,
                };
            }

            let modelUrl = `${config.host}/${models[id].url}`;
            log(`[Load] Loading ${name} · ${models[id].size}`);

            let modelBuffer = await getModelOPFS(name, modelUrl, false);
            log(`[Load] ${name} load time: ${(performance.now() - start).toFixed(2)}ms`);
            log(`[Session Create] Creating ${name}`);
            start = performance.now();
            const extra_opt = model.opt || {};
            const sess_opt = { ...opt, ...extra_opt };
            model.sess = await ort.InferenceSession.create(modelBuffer, sess_opt);
            log(`[Session Create] ${name} create time: ${(performance.now() - start).toFixed(2)}ms`);
        } catch (e) {
            log(`[Session Create] ${name} failed, ${e}`);
        }
    }
    placeholder.setAttribute('class', 'none');
}

async function main() {
    const model = MODELS[config.model];

    canvas = document.getElementById("img_canvas");
    canvas.style.cursor = "wait";

    filein = document.getElementById("file-in");
    decoder_latency = document.getElementById("decoder_latency");

    document.getElementById("clear-button").addEventListener("click", () => {
        points = [];
        labels = [];
        decoder(points, labels);
    });

    let img = document.getElementById("original-image");

    await load_models(MODELS[config.model]).then(() => {
        canvas.addEventListener("click", handleClick);
        canvas.addEventListener("mousemove", handleMouseMove);
        document.getElementById("cut-button").addEventListener("click", handleCut);

        // image upload
        filein.onchange = function (evt) {
            let target = evt.target || window.event.src, files = target.files;
            if (FileReader && files && files.length) {
                let fileReader = new FileReader();
                fileReader.onload = () => {
                    img.onload = () => handleImage(img);
                    img.src = fileReader.result;
                }
                fileReader.readAsDataURL(files[0]);
            }
        };
        handleImage(img);
    }, (e) => {
        log(e);
    });
}

async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter()
        return adapter.features.has('shader-f16')
    } catch (e) {
        return false
    }
}

const padNumber = (num, fill) => {
    let len = ('' + num).length;
    return Array(fill > len ? fill - len + 1 || 0 : 0).join(0) + num;
};

const getDateTime = () => {
    let date = new Date(),
        m = padNumber(date.getMonth() + 1, 2),
        d = padNumber(date.getDate(), 2),
        hour = padNumber(date.getHours(), 2),
        min = padNumber(date.getMinutes(), 2),
        sec = padNumber(date.getSeconds(), 2);
    return `${m}/${d} ${hour}:${min}:${sec}`;
};

const getOrtDevVersion = async () => {
    const response = await fetch('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/');
    const htmlString = await response.text();
    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlString, 'text/html');
    let selectElement = doc.querySelector('.path li');
    selectElement = doc.querySelector('select.versions.select-css');
    const options = Array.from(selectElement.querySelectorAll('option')).map(
        (option) => option.value
    );
    return options[0].replace('onnxruntime-web@', '');
};

const checkWebNN = async () => {
    let status = document.querySelector('#webnnstatus');
    let circle = document.querySelector('#circle');
    let info = document.querySelector('#info');
    let webnnStatus = await webNnStatus();

    if (webnnStatus.webnn) {
        status.setAttribute('class', 'green');
        info.innerHTML = 'WebNN supported';
    } else {
        if (webnnStatus.error) {
            status.setAttribute('class', 'red');
            info.innerHTML = 'WebNN not supported: ' + webnnStatus.error;
        } else {
            status.setAttribute('class', 'red');
            info.innerHTML = 'WebNN not supported';
        }
    }

    if (getQueryValue('provider') && getQueryValue('provider').toLowerCase().indexOf('webnn') == -1) {
        circle.setAttribute('class', 'none');
        info.innerHTML = '';
    }
};

const webNnStatus = async () => {
    let result = {};
    try {
        const context = await navigator.ml.createContext();
        if (context) {
            try {
                const builder = new MLGraphBuilder(context);
                if (builder) {
                    result.webnn = true;
                    return result;
                } else {
                    result.webnn = false;
                    return result;
                }
            } catch (e) {
                result.webnn = false;
                result.error = e.message;
                return result;
            }
        } else {
            result.webnn = false;
            return result;
        }
    } catch (ex) {
        result.webnn = false;
        result.error = ex.message;
        return result;
    }
};

const setupORT = async () => {
    const ortversion = document.querySelector('#ortversion');
    removeElement('onnxruntime-web');
    let ortVersion = await getOrtDevVersion();
    // let ortLink = '';
    // if (ortVersion && ortVersion.length > 4) {
    //     await loadScript('onnxruntime-web', `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/ort.all.min.js`);
    //     ortLink = `https://www.npmjs.com/package/onnxruntime-web/v/${ortVersion}`
    //     ortversion.innerHTML = `ONNX Runtime Web: <a href="${ortLink}">${ortVersion}</a><br/>[To do: Use WebNN EP of ORT Web 1.18 release version]`;
    // } else {
    //     await loadScript('onnxruntime-web', './dist/ort.all.min.js');
    //     ortversion.innerHTML = `ONNX Runtime Web: Test version`;
    // }
    await loadScript('onnxruntime-web', './dist/ort.all.min.js');
    ortversion.innerHTML = `ONNX Runtime Web: <a href="https://github.com/microsoft/onnxruntime/pull/19558">Test version with einsum op support</a>`;
}

const loadScript = async (id, url) => {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.onload = resolve;
        script.onerror = reject;
        script.id = id;
        script.src = url;
        if (url.startsWith('http')) {
            script.crossOrigin = 'anonymous';
        }
        document.body.append(script);
    })
}

const removeElement = async (id) => {
    let el = document.querySelector(id);
    if (el) {
        el.parentNode.removeChild(el);
    }
}

const getQueryValue = (name) => {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

let placeholder;

const ui = async () => {
    await setupORT();

    // ort.env.wasm.wasmPaths = 'dist/';
    ort.env.wasm.numThreads = config.threads;
    // ort.env.wasm.proxy = config.provider == "wasm";

    const title = document.querySelector('#title');
    const backends = document.querySelector('#backends');
    if (getQueryValue('provider') && getQueryValue('provider').toLowerCase().indexOf('webgpu') > -1) {
        title.innerHTML = 'WebGPU';
        backends.innerHTML = '<a href="index.html?provider=wasm&model=sam_b_int8" title="Wasm backend">Wasm</a> · <a href="index.html" title="WebNN backend">WebNN</a>';
    } else if (getQueryValue('provider') && getQueryValue('provider').toLowerCase().indexOf('wasm') > -1){
        title.innerHTML = 'Wasm';
        backends.innerHTML = '<a href="index.html?provider=webgpu&model=sam_b" title="WebGPU backend">WebGPU</a> · <a href="index.html" title="WebNN backend">WebNN</a>';
    } else {
        title.innerHTML = 'WebNN';
        backends.innerHTML = '· <a href="index.html?provider=wasm&model=sam_b_int8" title="Wasm backend">Wasm</a> · <a href="index.html?provider=webgpu&model=sam_b" title="WebGPU backend">WebGPU</a>';
    }
    await checkWebNN();
    placeholder = document.querySelector('#placeholder div');

    const fp16 = await hasFp16();
    if (config.provider == 'webgpu' && !fp16) {
        log("Your GPU or Browser doesn't support webgpu/f16");
    } else if (config.provider == 'webnn' && !("ml" in navigator) && typeof MLGraphBuilder == 'undefined') {
        log("Your Browser doesn't support WebNN");
    } else {
        await main();
    }
}

document.addEventListener('DOMContentLoaded', ui, false);