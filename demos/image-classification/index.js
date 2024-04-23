import {
  setupORT,
  webNnStatus,
  getQueryValue,
  getTime,
} from "../../assets/js/common_utils.js";
import { softmax } from "../../assets/js/math.js";
import { isDict } from "../../assets/js/data_type.js";
import { imagenetClassesTopK } from "./image-classification.js";
import ndarray from 'ndarray';
import ops from 'ndarray-ops';

function log(i) {
  console.log(i);
  document.getElementById("status").innerText += `\n[${getTime()}] ${i}`;
}

let progress;
let fetchProgress;
let modelCompileProgress = 0;
let modelFetchProgress = 0;
let sess;

const config = {
  host: location.href.includes("github.io")
    ? "https://huggingface.co/onnxruntime-web-temp/demo/resolve/main/image-classification"
    : "models",
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
      modelFetchProgress = 80.0;
      progress = modelFetchProgress + modelCompileProgress;
      updateProgressBar(progress.toFixed(2));
      progressInfo.innerHTML = `Loading ${
        config.name
      } model · ${progress.toFixed(2)}%`;
      return buffer;
    }
  } catch (e) {
    return await updateFile();
  }
}

async function readResponse(name, response) {
  const contentLength = response.headers.get("Content-Length");
  let total = parseInt(contentLength ?? "0");
  let buffer = new Uint8Array(total);
  let loaded = 0;

  const reader = response.body.getReader();
  async function read() {
    const { done, value } = await reader.read();
    if (done) return;

    let newLoaded = loaded + value.length;
    fetchProgress = (newLoaded / contentLength) * 100;

    modelFetchProgress = 0.8 * fetchProgress;
    progress = modelFetchProgress + modelCompileProgress;
    updateProgressBar(progress.toFixed(2));
    progressInfo.innerHTML = `Loading ${config.name} model · ${progress.toFixed(
      2
    )}%`;

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

async function loadAndCreate(config) {
  log("[Load] ONNX Runtime Execution Provider: " + config.provider);
  let start;
  try {
    start = performance.now();
    const opt = {
      executionProviders: [config.provider],
    };
    if (config.provider == "webnn") {
      opt.executionProviders = [
        {
          name: "webnn",
          deviceType: "gpu",
        },
      ];
      // opt.freeDimensionOverrides = {
      //     num_points: num_points,
      // };
    }

    let modelUrl = `${config.host}/${config.model}`;
    log(`[Load] Loading ${config.name} · ${config.size}`);

    let modelBuffer = await getModelOPFS(config.model, modelUrl, false);
    log(
      `[Load] ${config.name} load time: ${(performance.now() - start).toFixed(
        2
      )}ms`
    );
    progressInfo.innerHTML = `Creating ${
      config.name
    } session· ${progress.toFixed(2)}%`;
    log(`[Session Create] Creating ${config.name}`);
    start = performance.now();
    sess = await ort.InferenceSession.create(modelBuffer, opt);
    log(
      `[Session Create] ${config.name} create time: ${(
        performance.now() - start
      ).toFixed(2)}ms`
    );

    modelCompileProgress = 20;
    progress = modelFetchProgress + modelCompileProgress;
    updateProgressBar(progress.toFixed(2));
    progressInfo.innerHTML = `${config.name} compiled · ${progress.toFixed(
      2
    )}%`;
  } catch (e) {
    log(`[Session Create] ${config.name} failed, ${e.message}`);
  }

  placeholder.setAttribute("class", "none");
  canvas.setAttribute("class", "");
}

let sessionBackend = config.provider;
let modelLoading = true;
let modelInitializing = true;
let modelLoadingError = false;
let sessionRunning = false;
let inferenceTime = 0;
let imageURLInput = "";
let imageURLSelect = null;
let imageURLSelectList;
let imageLoading = false;
let imageLoadingError = false;
let output = [];
let modelFile = new ArrayBuffer(0);

const preprocess = (ctx) => {
  const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  const { data, width, height } = imageData;

  // data processing
  const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
  const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [
    1,
    3,
    width,
    height,
  ]);

  ops.assign(
    dataProcessedTensor.pick(0, 0, null, null),
    dataTensor.pick(null, null, 0)
  );
  ops.assign(
    dataProcessedTensor.pick(0, 1, null, null),
    dataTensor.pick(null, null, 1)
  );
  ops.assign(
    dataProcessedTensor.pick(0, 2, null, null),
    dataTensor.pick(null, null, 2)
  );

  ops.divseq(dataProcessedTensor, 255);
  ops.subseq(dataProcessedTensor.pick(0, 0, null, null), 0.485);
  ops.subseq(dataProcessedTensor.pick(0, 1, null, null), 0.456);
  ops.subseq(dataProcessedTensor.pick(0, 2, null, null), 0.406);

  ops.divseq(dataProcessedTensor.pick(0, 0, null, null), 0.229);
  ops.divseq(dataProcessedTensor.pick(0, 1, null, null), 0.224);
  ops.divseq(dataProcessedTensor.pick(0, 2, null, null), 0.225);

  const tensor = new ort.Tensor(
    "float32",
    new Float32Array(width * height * 3),
    [1, 3, width, height]
  );
  tensor.data.set(dataProcessedTensor.data);
  return tensor;
};

const getPredictedClass = (res) => {
  if (!res || res.length === 0) {
    const empty = [];
    for (let i = 0; i < 5; i++) {
      empty.push({ name: "-", probability: 0, index: 0 });
    }
    return empty;
  }
  const output = softmax(Array.prototype.slice.call(res));
  return imagenetClassesTopK(output, 5);
};

const getTensor = (type, data, dims) => {
  let typedArray;
  if (type === "bool") {
    return new ort.Tensor(type, [data], [1]);
  } else if (type === "int8") {
    typedArray = Int8Array;
  } else if (type === "uint8") {
    typedArray = Uint8Array;
  } else if (type === "uint16") {
    typedArray = Uint16Array;
  } else if (type === "float16") {
    typedArray = Uint16Array;
  } else if (type === "float32") {
    typedArray = Float32Array;
  } else if (type === "int32") {
    typedArray = Int32Array;
  } else if (type === "int64") {
    typedArray = BigInt64Array;
  }

  let _data;
  if (Array.isArray(data) || ArrayBuffer.isView(data)) {
    _data = data;
  } else {
    let size = 1;
    dims.forEach((dim) => {
      size *= dim;
    });
    if (data === "random") {
      _data = typedArray.from({ length: size }, () => Math.random());
    } else if (data === "ramp") {
      _data = typedArray.from({ length: size }, (_, i) => i);
    } else {
      _data = typedArray.from({ length: size }, () => data);
    }
  }
  return new ort.Tensor(type, _data, dims);
};

const getFeeds = () => {
  let feeds = {};
  let inputs = [{ input: ["float16", "random", [1, 3, 224, 224], {}] }];
  for (let input of inputs) {
    if (isDict(input)) {
      for (let key in input) {
        let value = input[key];
        feeds[key] = getTensor(value[0], value[1], value[2]);
      }
    }
  }

  return feeds;
};

const runModel = async (model, preprocessedData) => {
  const start = new Date();
  try {
    let feeds = getFeeds();
    console.log(feeds);
    console.log(model.inputNames[0]);
    feeds[model.inputNames[0]] = preprocessedData;
    const outputData = await model.run(feeds);
    const end = new Date();
    const inferenceTime = end.getTime() - start.getTime();
    const output = outputData[model.outputNames[0]];

    return [output, inferenceTime];
  } catch (e) {
    console.error(e);
    throw new Error();
  }
};

const handleImage = (img) => {
  console.log("Handle image");
};

const initSession = async () => {
  const ctx = canvas.getContext("2d");
  const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  const preprocessedData = preprocess(ctx);
  runModel(sess, preprocessedData);
};

async function main() {
  canvas.style.cursor = "wait";
  let img = document.getElementById("original-image");
  await loadAndCreate(config).then(
    () => {
      filein.onchange = function (evt) {
        let target = evt.target || window.event.src,
          files = target.files;
        if (FileReader && files && files.length) {
          let fileReader = new FileReader();
          fileReader.onload = () => {
            img.onload = () => handleImage(img);
            img.src = fileReader.result;
          };
          fileReader.readAsDataURL(files[0]);
        }
      };
      // handleImage(img);
      initSession();
    },
    (e) => {
      log(e);
    }
  );
}

const checkWebNN = async () => {
  let status = document.querySelector("#webnnstatus");
  let circle = document.querySelector("#circle");
  let info = document.querySelector("#info");
  let webnnStatus = await webNnStatus();

  if (webnnStatus.webnn) {
    status.setAttribute("class", "green");
    info.innerHTML = "WebNN supported";
  } else {
    if (webnnStatus.error) {
      status.setAttribute("class", "red");
      info.innerHTML = "WebNN not supported: " + webnnStatus.error;
    } else {
      status.setAttribute("class", "red");
      info.innerHTML = "WebNN not supported";
    }
  }

  if (
    getQueryValue("provider") &&
    getQueryValue("provider").toLowerCase().indexOf("webnn") == -1
  ) {
    circle.setAttribute("class", "none");
    info.innerHTML = "";
  }
};

const updateProgressBar = (progress) => {
  progressBar.style.width = `${progress}%`;
};

let canvas;
let filein;
let progressBar;
let progressInfo;
let indicator;

const ui = async () => {
  canvas = document.querySelector("#img_canvas");
  filein = document.querySelector("#file-in");
  progressBar = document.querySelector("#progress-bar");
  progressInfo = document.querySelector("#progress-info");
  indicator = document.querySelector("#indicator");
  canvas.setAttribute("class", "none");
  await setupORT();

  // ort.env.wasm.wasmPaths = 'dist/';
  ort.env.wasm.numThreads = config.threads;
  // ort.env.wasm.proxy = config.provider == "wasm";

  const title = document.querySelector("#title");
  const backends = document.querySelector("#backends");
  if (
    getQueryValue("provider") &&
    getQueryValue("provider").toLowerCase().indexOf("webgpu") > -1
  ) {
    title.innerHTML = "WebGPU";
  } else if (
    getQueryValue("provider") &&
    getQueryValue("provider").toLowerCase().indexOf("wasm") > -1
  ) {
    title.innerHTML = "Wasm";
  } else {
    title.innerHTML = "WebNN";
  }
  backends.innerHTML =
    ' · <a href="index.html?provider=wasm" title="Wasm backend">Wasm</a> <a href="index.html?provider=webnn&devicetype=cpu" title="WebNN CPU backend">WebNN CPU</a> · <a href="index.html?provider=webgpu" title="WebGPU backend">WebGPU</a> <a href="index.html?provider=webnn&devicetype=gpu" title="WebNN GPU backend">WebNN GPU</a> · <a href="index.html?provider=webnn&devicetype=npu" title="WebNN NPU backend">WebNN NPU</a>';

  await checkWebNN();
  await main();
};

document.addEventListener("DOMContentLoaded", ui, false);
