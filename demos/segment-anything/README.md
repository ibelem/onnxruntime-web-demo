# Segment-Anything: Browser-Based Image Segmentation with WebGPU, WebNN and ONNX Runtime Web

This repository contains an example of running [Segment-Anything](https://github.com/facebookresearch/segment-anything), an encoder/decoder model for image segmentation, in a browser using [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) with WebGPU.

You can try out the live demo [here](https://guschmue.github.io/ort-webgpu/segment-anything/index.html).

## Model Overview

Segment-Anything creates embeddings for an image using an encoder. These embeddings are then used by the decoder to create and update the segmentation mask. The decoder can run in ONNX Runtime Web using WebAssembly with latencies at ~200ms. 

The encoder is more compute-intensive, taking ~45sec in WebAssembly, which is not practical. However, by using WebGPU, we can speed up the encoder, making it feasible to run it inside the browser, even on an integrated GPU.

## Getting Started

### Prerequisites

Ensure that you have [Node.js](https://nodejs.org/) installed on your machine.

### The ONNX Model

The model used in this project is hosted on [Hugging Face](https://huggingface.co/schmuell/sam-b-fp16). It was created using [samexporter](https://github.com/vietanhdev/samexporter).

### Running the Project

Start a web server to serve the current folder at http://localhost:8888/. To start the server, run:

```sh
npx http-server .
```
