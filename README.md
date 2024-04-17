# ONNX Runtime Web Demo

ONNX Runtime Web demo is an interactive demo portal showing real use cases running [ONNX Runtime Web](https://github.com/microsoft/onnxruntime/tree/master/js/web#readme). It currently supports four examples for you to quickly experience the power of ONNX Runtime Web.

The demo is available here [ONNX Runtime Web demo website](https://microsoft.github.io/onnxruntime-web-demo/).

_NOTE: Currently, the supported platforms are Edge/Chrome (support for other platforms is coming soon)._

## Use Cases

The demo provides four scenarios based on four different ONNX pre-trained deep learning models.

### 1. Stable Diffusion 1.5

[Stable Diffusion](https://huggingface.co/onnxruntime-web-temp/demo/tree/main/stable-diffusion-1.5) is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.

### 2. SD-Turbo

[SD-Turbo](https://huggingface.co/onnxruntime-web-temp/demo/tree/main/sd-turbo) is a fast generative text-to-image model that can synthesize photorealistic images from a text prompt in a single network evaluation. In the demo, you can generate an image in 2s on AI PC devices by leveraging WebNN API, a dedicated low-level API for neural network inference hardware acceleration.

### 3. Segment Anything

[Segment Anything](https://huggingface.co/onnxruntime-web-temp/demo/tree/main/segment-anything) is a new AI model from Meta AI that can "cut out" any object. In the demo, you can segment any object from your uploaded images.

### 4. Whisper Base

[Whisper Base](https://huggingface.co/onnxruntime-web-temp/demo/tree/main/whisper-base) is a pre-trained model for automatic speech recognition (ASR) and speech translation. In the demo, you can experience the speech to text feature by using on-device inference powered by WebNN API and DirectML, especially the NPU acceleration.

## Archived Demos

The code of demos below can be found under [v1](./v1/) folder.

### 1. MobileNet

[MobileNet](https://github.com/onnx/models/tree/master/vision/classification/mobilenet) models perform image classification - they take images as input and classify the major object in the image into a set of pre-defined classes. They are trained on ImageNet dataset which contains images from 1000 classes. MobileNet models are also very efficient in terms of speed and size and hence are ideal for embedded and mobile applications.

### 2. SqueezeNet

[SqueezeNet](https://github.com/onnx/models/tree/master/vision/classification/squeezenet) is a light-weight convolutional network for image classification. In the demo, you can select or upload an image and see which category it's from in miliseconds.

### 3. FER+ Emotion Recognition

[Emotion Ferplus](https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus)
is a deep convolutional neural network for emotion recognition in faces. In the demo, you can choose to either select an image with any human face or to start a webcam and see what emotion it's showing.

### 4. Yolo

[Yolo](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov2) is a real-time neural network for object detection. It can detect 20 different objects such as person, potted plant and chair. In the demo, you can choose to either select an image or start a webcam to see what objects are in it.

### 5. MNIST

[MNIST](https://github.com/onnx/models/tree/master/vision/classification/mnist) is a convolutional neural network that predicts handwritten digits. In the demo, you can draw any number on the canvas and the model will tell you what number it is!

## Run ONNX Runtime Web Demo

### Install Dependencies

```
cd onnxruntime-web-demo
openssl req -newkey rsa:2048 -new -nodes -x509 -days 3650 -keyout key.pem -out cert.pem
npm install
```
> The private and public keys are used for local https connection, WebNN and WebGPU can only run in secured contexts. This step is optional if running with http://localhost since it's considered a secure context and will behave like https.

### Run the demo

**WebNN Installation Guides**

WebNN requires a compatible browser to run, and Windows* 11 v21H2 (DML 1.6.0) or higher.

1. Download the latest [Microsoft Edge Canary](https://www.microsoft.com/edge/download/insider) or [Google Chrome Canary](https://google.com/chrome/canary) browser
2. To enable WebNN, in your browser address bar, enter `chrome://flags`, and then press `Enter`. An Experiments page opens
3. In the Search flags box, enter `webnn`. Enables WebNN API appears
4. In the drop-down menu, select `Enabled`
5. Relaunch your browser

**Run the demo in localhost**

```
npm run dev
```

This will start a dev server and run ONNX Runtime Web demo with the WebNN enabled browser on your localhost.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
