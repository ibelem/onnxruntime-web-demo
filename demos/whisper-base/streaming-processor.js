class StreamingProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();

        this.sampleRate = 0;
        this.publishInterval = 0;
        this.index = 0; // index of this._streamingBuffer data
        if (options && options.processorOptions) {
            const {
                sampleRate,
                chunkLength,
            } = options.processorOptions;

            this.sampleRate = sampleRate;
            // We will use a timer to gate our messages; this one will publish at 60hz
            this.publishInterval = chunkLength * sampleRate;
        }
        this.stopProcessing = false;
        this._streamingBuffer = new Float32Array(this.publishInterval);

        this.port.onmessage = e => {
            if (e.data.message === 'STOP_PROCESSING') {
                this.stopProcessing = e.data.data;
            }
        };
    }

    process(inputs, outputs, params) {
        if (this.stopProcessing) {
            // Do nothing, suspend the audio processing
        } else {
            // inputs[0][0]'s length is 128
            for (let sample = 0; sample < inputs[0][0].length; sample++) {
                const currentSample = inputs[0][0][sample];

                // Copy data to streaming buffer.
                this._streamingBuffer[this.index] = currentSample;
                this.index++;
                // Should publish, clear this._streamingBuffer and this.index
                if (this.index == this.publishInterval) {
                    this.port.postMessage({
                        message: 'START_TRANSCRIBE',
                        buffer: this._streamingBuffer,
                    }, [this._streamingBuffer.buffer.slice()]);
                    this._streamingBuffer.fill(0);
                    this.index = 0;
                }
            }
        }
        return true;
    }
}

registerProcessor('streaming-processor', StreamingProcessor);