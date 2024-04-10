// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run whisper in onnxruntime-web.
//

import { Whisper } from "./whisper.js";
import { loadScript, removeElement, getQueryValue, getQueryVariable, randomNumber, getOrtDevVersion, webNnStatus, log, concatBuffer, concatBufferArray, logUser } from "./utils.js";
import VADBuilder, { VADMode, VADEvent } from "./vad/embedded.js";

const kSampleRate = 16000;
const kIntervalAudio_ms = 1000;
const kSteps = kSampleRate * 30;
const kDelay = 100;

// whisper class
let whisper;

let provider = "webnn";
let dataType = "float32";

// audio context
var context = null;
let mediaRecorder;
let stream;

// some dom shortcuts
let fileUpload;
let labelFileUpload;
let record;
let speech;
let progress;
let audio_src;
let outputText;

// for audio capture
// This enum states the current speech state.
const SpeechStates = {
  UNINITIALIZED: 0,
  PROCESSING: 1,
  PAUSED: 2,
  FINISHED: 3,
};
let speechState = SpeechStates.UNINITIALIZED;

let streamingNode = null;
let sourceNode = null;
let audioChunks = []; // member {isSubChunk: boolean, data: Float32Array}
let subAudioChunks = [];
let chunkLength = 1 / 25; // length in sec of one audio chunk from AudioWorklet processor, recommended by vad
let maxChunkLength = 1; // max audio length for an audio processing
let silenceAudioCounter = 0;
// check if last audio processing is completed, to avoid race condition
let lastProcessingCompleted = true;
// check if last speech processing is completed when restart speech
let lastSpeechCompleted = true;

// involve webrtcvad to detect voice activity
let VAD = null;
let vad = null;

let singleAudioChunk = null; // one time audio process buffer
let subAudioChunkLength = 0; // length of a sub audio chunk
let speechToText = "";
let subText = "";

const blacklistTags = [
  "[inaudible]",
  " [inaudible]",
  "[ Inaudible ]",
  "[INAUDIBLE]",
  " [INAUDIBLE]",
  "[BLANK_AUDIO]",
  " [BLANK_AUDIO]",
  " [no audio]",
  "[no audio]",
  "[silent]",
];

function updateConfig() {
  const query = window.location.search.substring("1");
  const providers = ["webnn", "webgpu", "wasm"];
  const dataTypes = ["float32", "float16"];
  let vars = query.split("&");
  for (let i = 0; i < vars.length; i++) {
    let pair = vars[i].split("=");
    if (pair[0] == "provider" && providers.includes(pair[1])) {
      provider = pair[1];
    }
    if (pair[0] == "dataType" && dataTypes.includes(pair[1])) {
      dataType = pair[1];
    }
    if (pair[0] == "maxChunkLength") {
      maxChunkLength = parseInt(pair[1]);
    }
  }
}

// transcribe active
function busy() {
  progress.parentNode.style.display = "block";
  document.getElementById("outputText").innerHTML = "";
  document.getElementById("latency").innerText = "";
}

// transcribe done
function ready() {
  labelFileUpload.setAttribute('class', 'file-upload-label');
  fileUpload.disabled = false;
  record.disabled = false;
  speech.disabled = false;
  progress.style.width = "0%";
  // progress.parentNode.style.display = "none";
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// process audio buffer
async function process_audio(audio, starttime, idx, pos) {
  if (idx < audio.length) {
    // not done
    try {
      // update progress bar
      progress.style.width = ((idx * 100) / audio.length).toFixed(1) + "%";
      await sleep(kDelay);
      // run inference for 30 sec
      const xa = audio.slice(idx, idx + kSteps);
      const ret = await whisper.run(xa, kSampleRate);
      // append results to outputText
      outputText.innerHTML += ret;
      logUser(ret);
      // outputText.scrollTop = outputText.scrollHeight;
      await sleep(kDelay);
      process_audio(audio, starttime, idx + kSteps, pos + 30);
    } catch (e) {
      log(`Error · ${e.message}`);
      ready();
    }
  } else {
    // done with audio buffer
    const processing_time = (performance.now() - starttime) / 1000;
    const total = audio.length / kSampleRate;
    document.getElementById("latency").innerText = `${(
      total / processing_time
    ).toFixed(1)} x realtime`;
    log(
      `${
        document.getElementById("latency").innerText
      }, Total ${processing_time.toFixed(
        1
      )}s processing time for ${total.toFixed(1)}s audio`
    );
    ready();
  }
}

// transcribe audio source
async function transcribe_file() {
  if (audio_src.src == "") {
    log("Error · Set some Audio input");
    return;
  }

  busy();
  log("Starting transcribe ...");
  try {
    const buffer = await (await fetch(audio_src.src)).arrayBuffer();
    const audioBuffer = await context.decodeAudioData(buffer);
    var offlineContext = new OfflineAudioContext(
      audioBuffer.numberOfChannels,
      audioBuffer.length,
      audioBuffer.sampleRate
    );
    var source = offlineContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineContext.destination);
    source.start();
    const renderedBuffer = await offlineContext.startRendering();
    const audio = renderedBuffer.getChannelData(0);
    process_audio(audio, performance.now(), 0, 0);
  } catch (e) {
    log(`Error · ${e.message}`);
    ready();
  }
}

// start recording
async function startRecord() {
  if (mediaRecorder === undefined) {
    try {
      if (!stream) {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            autoGainControl: false,
            noiseSuppression: false,
            latency: 0,
          },
        });
      }
      mediaRecorder = new MediaRecorder(stream);
    } catch (e) {
      // record.innerText = "Record";
      log(`Preprocessing · Access to Microphone, ${e.message}`);
    }
  }
  let recording_start = performance.now();
  let chunks = [];

  mediaRecorder.ondataavailable = (e) => {
    chunks.push(e.data);
    document.getElementById("latency").innerText = `recorded: ${(
      (performance.now() - recording_start) /
      1000
    ).toFixed(1)}sec`;
  };

  mediaRecorder.onstop = () => {
    const blob = new Blob(chunks, { type: "audio/ogg; codecs=opus" });
    log(
      `Preprocessing · Recorded ${((performance.now() - recording_start) / 1000).toFixed(
        1
      )}sec audio`
    );
    audio_src.src = window.URL.createObjectURL(blob);
  };
  mediaRecorder.start(kIntervalAudio_ms);
}

// stop recording
function stopRecord() {
  if (mediaRecorder) {
    mediaRecorder.stop();
    transcribe_file();
    mediaRecorder = undefined;
  }
}

// start speech
async function startSpeech() {
  speechState = SpeechStates.PROCESSING;
  await captureAudioStream();
  if (streamingNode != null) {
    streamingNode.port.postMessage({ message: "STOP_PROCESSING", data: false });
  }
}

// stop speech
async function stopSpeech() {
  if (streamingNode != null) {
    streamingNode.port.postMessage({ message: "STOP_PROCESSING", data: true });
    speechState = SpeechStates.PAUSED;
  }
  silenceAudioCounter = 0;
  // push last singleAudioChunk to audioChunks, in case it is ignored.
  if (singleAudioChunk != null) {
    audioChunks.push({ isSubChunk: false, data: singleAudioChunk });
    singleAudioChunk = null;
    if (
      lastProcessingCompleted &&
      lastSpeechCompleted &&
      audioChunks.length > 0
    ) {
      await processAudioBuffer();
    }
  }
  // if (stream) {
  //     stream.getTracks().forEach(track => track.stop());
  // }
  // if (context) {
  //     // context.close().then(() => context = null);
  //     await context.suspend();
  // }
}

// use AudioWorklet API to capture real-time audio
async function captureAudioStream() {
  try {
    if (context && context.state === "suspended") {
      await context.resume();
    }
    // Get user's microphone and connect it to the AudioContext.
    if (!stream) {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          autoGainControl: false,
          noiseSuppression: true,
          latency: 0,
        },
      });
    }
    if (streamingNode) {
      return;
    }

    VAD = await VADBuilder();
    vad = new VAD(VADMode.AGGRESSIVE, kSampleRate);

    // clear output context
    outputText.innerHTML = "";
    sourceNode = new MediaStreamAudioSourceNode(context, {
      mediaStream: stream,
    });
    await context.audioWorklet.addModule("streaming-processor.js");
    const streamProperties = {
      numberOfChannels: 1,
      sampleRate: context.sampleRate,
      chunkLength: chunkLength,
    };
    streamingNode = new AudioWorkletNode(context, "streaming-processor", {
      processorOptions: streamProperties,
    });

    streamingNode.port.onmessage = async (e) => {
      if (e.data.message === "START_TRANSCRIBE") {
        const frame = VAD.floatTo16BitPCM(e.data.buffer); // VAD requires Int16Array input
        const res = vad.processBuffer(frame);
        // has voice
        if (res == VADEvent.VOICE) {
          singleAudioChunk = concatBuffer(singleAudioChunk, e.data.buffer);
          // meet max audio chunk length for a single process, split it.
          if (singleAudioChunk.length >= kSampleRate * maxChunkLength) {
            if (subAudioChunkLength == 0) {
              // subAudioChunkLength >= kSampleRate * maxChunkLength
              subAudioChunkLength = singleAudioChunk.length;
            }
            audioChunks.push({ isSubChunk: true, data: singleAudioChunk });
            singleAudioChunk = null;
          }

          silenceAudioCounter = 0;
        } else {
          // no voice
          silenceAudioCounter++;
          // if only one silence chunk exists between two voice chunks,
          // just treat it as a continous audio chunk.
          if (singleAudioChunk != null && silenceAudioCounter > 1) {
            audioChunks.push({ isSubChunk: false, data: singleAudioChunk });
            singleAudioChunk = null;
          }
        }

        // new audio is coming, and no audio is processing
        if (lastProcessingCompleted && audioChunks.length > 0) {
          await processAudioBuffer();
        }
      }
    };

    sourceNode.connect(streamingNode).connect(context.destination);
  } catch (e) {
    log(`Error · Capturing audio - ${e.message}`);
  }
}

async function processAudioBuffer() {
  lastProcessingCompleted = false;
  let processBuffer = audioChunks[0].data;
  // it is sub audio chunk, need to do rectification at last sub chunk
  if (audioChunks[0].isSubChunk) {
    subAudioChunks.push(processBuffer);
    // if the speech is pause, and it is the last audio chunk, concat the subAudioChunks to do rectification
    if (speechState == SpeechStates.PAUSED && audioChunks.length == 1) {
      processBuffer = concatBufferArray(subAudioChunks);
      subAudioChunks = []; // clear subAudioChunks
    }
  } else {
    // concat all subAudoChunks to do rectification
    if (subAudioChunks.length > 0) {
      subAudioChunks.push(processBuffer); // append sub chunk's next neighbor
      processBuffer = concatBufferArray(subAudioChunks);
      subAudioChunks = []; // clear subAudioChunks
    }
  }
  // if total length of subAudioChunks >= 10 sec,
  // force to break it from subAudioChunks to reduce latency
  // because it has to wait for more than 10 sec to do audio processing.
  if (subAudioChunks.length * maxChunkLength >= 10) {
    processBuffer = concatBufferArray(subAudioChunks);
    subAudioChunks = [];
  }
  // ignore too small audio chunk, e.g. 0.16 sec
  // per testing, audios less than 0.16 sec are almost blank audio
  if (processBuffer.length > kSampleRate * 0.16) {
    const start = performance.now();
    const ret = await whisper.run(processBuffer, kSampleRate);
    console.log(
      `${processBuffer.length / kSampleRate} sec audio processing time: ${(
        (performance.now() - start) /
        1000
      ).toFixed(2)} sec`
    );
    console.log("Result: ", ret);
    // TODO? throttle the un-processed audio chunks?
    //In order to catch up latest audio to achieve real-time effects.
    console.log("Un-processed audio chunk length: ", audioChunks.length - 1);
    // ignore slient, inaudible audio output, i.e. '[BLANK_AUDIO]'
    if (!blacklistTags.includes(ret)) {
      if (subAudioChunks.length > 0) {
        subText += ret;
        outputText.innerHTML = speechToText + subText;
      } else {
        speechToText += ret;
        outputText.innerHTML = speechToText;
      }
      // outputText.scrollTop = outputText.scrollHeight;
    }
  }
  lastProcessingCompleted = true;
  audioChunks.shift(); // remove processed chunk
  if (subAudioChunks.length == 0) {
    // clear subText
    subText = "";
  }
  if (audioChunks.length > 0) {
    // recusive audioBuffer in audioChunks
    lastSpeechCompleted = false;
    await processAudioBuffer();
  } else {
    lastSpeechCompleted = true;
  }
}

const checkWebNN = async () => {
  let status = document.querySelector("#webnnstatus");
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
    getQueryValue("provider").toLowerCase().indexOf("webgpu") > -1
  ) {
    status.innerHTML = "";
  }
};

const setupORT = async () => {
  const ortversion = document.querySelector("#ortversion");
  removeElement("onnxruntime-web");
  let ortVersion = await getOrtDevVersion();
  let ortLink = "";
  if (ortVersion && ortVersion.length > 4) {
    await loadScript(
      "onnxruntime-web",
      `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/ort.all.min.js`
    );
    ortLink = `https://www.npmjs.com/package/onnxruntime-web/v/${ortVersion}`;
    ortversion.innerHTML = `ONNX Runtime Web: <a href="${ortLink}">${ortVersion}</a>`;
  } else {
    await loadScript("onnxruntime-web", "./dist/ort.all.min.js");
    ortversion.innerHTML = `ONNX Runtime Web: Test version`;
  }
};

const ui = async () => {
  audio_src = document.querySelector("audio");
  labelFileUpload = document.getElementById("label-file-upload");
  fileUpload = document.getElementById("file-upload");
  record = document.getElementById("record");
  speech = document.getElementById("speech");
  progress = document.getElementById("progress");
  outputText = document.getElementById("outputText");
  labelFileUpload.setAttribute('class', 'file-upload-label disabled');
  fileUpload.disabled = true;
  record.disabled = true;
  speech.disabled = true;
  // progress.parentNode.style.display = "none";

  await setupORT();
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  if (
    getQueryValue("provider") &&
    getQueryValue("provider").toLowerCase().indexOf("webgpu") > -1
  ) {
    title.innerHTML = "WebGPU";
  }
  await checkWebNN();
  updateConfig();

  // click on Record
  record.addEventListener("click", (e) => {
    if (record.getAttribute('class').indexOf("active") == -1) {
      record.setAttribute('class', 'active');
      startRecord();
    } else {
      record.setAttribute('class', '');
      stopRecord();
    }
  });

  // click on Speech
  speech.addEventListener("click", async (e) => {
    if (speech.getAttribute('class').indexOf("active") == -1) {
      if (!lastSpeechCompleted) {
        log("[Session Run] Last speech-to-text has not completed yet, try later...");
        return;
      }
      subText = "";
      speech.setAttribute('class', 'active')
      await startSpeech();
    } else {
      speech.setAttribute('class', '');
      await stopSpeech();
    }
  });

  // drop file
  fileUpload.onchange = function (evt) {
    let target = evt.target || window.event.src,
      files = target.files;
    if(files && files[0]) {
      audio_src.src = URL.createObjectURL(files[0]);
      transcribe_file();
    }
  };

  log(`ONNX Runtime Web Execution Provider loaded · ${provider.toUpperCase()}`);
  try {
    const whisper_url = location.href.includes("github.io")
      ? "https://huggingface.co/onnxruntime-web-temp/demo/resolve/main/whisper-base"
      : "./models/";
    whisper = new Whisper(whisper_url, provider, dataType);
    await whisper.create_whisper_processor();
    await whisper.create_whisper_tokenizer();
    await whisper.create_ort_sessions();
    log("Ready to transcribe ...");
    ready();
    context = new AudioContext({
      sampleRate: kSampleRate,
      channelCount: 1,
      echoCancellation: false,
      autoGainControl: true,
      noiseSuppression: true,
    });
    if (!context) {
      throw new Error(
        "no AudioContext, make sure domain has access to Microphone"
      );
    }

  } catch (e) {
    log(`Error · ${e.message}`);
  }
};

document.addEventListener("DOMContentLoaded", ui, false);
