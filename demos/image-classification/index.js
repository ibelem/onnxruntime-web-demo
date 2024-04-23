import { imagenetClassesTopK } from './image-classification'

function log(i) { 
    console.log(i); 
    document.getElementById('status').innerText += `\n[${getDateTime()}] ${i}`;
}

let modelCompileProgress = 0;
let modelFetchProgress = 0;
let progressBar;
let progressInfo;
let sess;

const config = {
    host: location.href.includes("github.io") ? "https://huggingface.co/onnxruntime-web-temp/demo/resolve/main/image-classification" : "models",
    model: "mobilenetv2-fp16.onnx",
    name: "MobileNet v2",
    dataType: "FP16",
    size: "7.42 MB",
    provider: "webnn",
    deviceType: "gpu",
    threads: "1",
};

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
            modelFetchProgress = 80.00;
            progress = modelFetchProgress + modelCompileProgress;
            updateProgressBar(progress.toFixed(2));
            progressInfo.innerHTML = `Loading ${config.name} model · ${progress.toFixed(2)}%`;
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

        modelFetchProgress = 0.80 * fetchProgress;
        progress = modelFetchProgress + modelCompileProgress;
        updateProgressBar(progress.toFixed(2));
        progressInfo.innerHTML = `Loading ${config.name} model · ${progress.toFixed(2)}%`;

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

async function load_model(config) {
    log("[Load] ONNX Runtime Execution Provider: " + config.provider);
        let start;
        try {
            start = performance.now();
            const opt = {
                executionProviders: [config.provider]
            };
            if (config.provider == 'webnn') {
                opt.executionProviders = [{
                    name: "webnn",
                    deviceType: "gpu",
                }];
                // opt.freeDimensionOverrides = {
                //     num_points: num_points,
                // };
            }

            let modelUrl = `${config.host}/${config.model}`;
            log(`[Load] Loading ${config.name} · ${config.size}`);

            let modelBuffer = await getModelOPFS(config.model, modelUrl, false);
            log(`[Load] ${config.name} load time: ${(performance.now() - start).toFixed(2)}ms`);
            log(`[Session Create] Creating ${config.name}`);
            start = performance.now();
            sess = await ort.InferenceSession.create(modelBuffer, opt);
            log(`[Session Create] ${config.name} create time: ${(performance.now() - start).toFixed(2)}ms`);

            modelCompileProgress = 100;
            progress = modelFetchProgress + modelCompileProgress;
            updateProgressBar(progress.toFixed(2));
            progressInfo.innerHTML = `${config.name} compiled · ${progress.toFixed(2)}%`;
        } catch (e) {
            log(`[Session Create] ${config.name} failed, ${e.message}`);
        }
 
    placeholder.setAttribute('class', 'none');
    canvas.setAttribute('class', '');
}

const handleImage = (img) => {
    console.log('Handle image');
}

async function main() {
    canvas.style.cursor = "wait";
    let img = document.getElementById("original-image");
    await load_model(config).then(() => {
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
    let ortLink = '';
    if (ortVersion && ortVersion.length > 4) {
        await loadScript('onnxruntime-web', `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/ort.all.min.js`);
        ortLink = `https://www.npmjs.com/package/onnxruntime-web/v/${ortVersion}`
        ortversion.innerHTML = `ONNX Runtime Web: <a href="${ortLink}">${ortVersion}</a>`;
    } else {
        await loadScript('onnxruntime-web', './dist/ort.all.min.js');
        ortversion.innerHTML = `ONNX Runtime Web: Test version`;
    }
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

const updateProgressBar = (progress) => {
    progressBar.style.width = `${progress}%`;
}

const ui = async () => {
    canvas = document.querySelector("#img_canvas");
    filein = document.querySelector("#file-in");
    progressBar = document.querySelector('#progress-bar');
    progressInfo = document.querySelector('#progress-info');
    indicator = document.querySelector('#indicator');
    canvas.setAttribute('class', 'none');
    await setupORT();

    // ort.env.wasm.wasmPaths = 'dist/';
    ort.env.wasm.numThreads = config.threads;
    // ort.env.wasm.proxy = config.provider == "wasm";

    const title = document.querySelector('#title');
    const backends = document.querySelector('#backends');
    if (getQueryValue('provider') && getQueryValue('provider').toLowerCase().indexOf('webgpu') > -1) {
        title.innerHTML = 'WebGPU';
        } else if (getQueryValue('provider') && getQueryValue('provider').toLowerCase().indexOf('wasm') > -1){
        title.innerHTML = 'Wasm';
        } else {
        title.innerHTML = 'WebNN';
        }
        backends.innerHTML = ' · <a href="index.html?provider=wasm" title="Wasm backend">Wasm</a> <a href="index.html?provider=webnn&devicetype=cpu" title="WebNN CPU backend">WebNN CPU</a> · <a href="index.html?provider=webgpu" title="WebGPU backend">WebGPU</a> <a href="index.html?provider=webnn&devicetype=gpu" title="WebNN GPU backend">WebNN GPU</a> · <a href="index.html?provider=webnn&devicetype=npu" title="WebNN NPU backend">WebNN NPU</a>';
        
    await checkWebNN();
    await main();
}

document.addEventListener('DOMContentLoaded', ui, false);