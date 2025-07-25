class SilenceDetector extends AudioWorkletProcessor {
  constructor() {
    super();
    this.silenceThreshold = 0.1; // Amplitude threshold for silence detection
    this.silenceDurationThreshold = 44100; //44,100  Corresponds to 500 ms at a sample rate of 44.1 kHz
    this.silenceCounter = 0;

    // Listening to messages from the main thread to update thresholds
    this.port.onmessage = (event) => {
      if (event.data.silenceThreshold !== undefined) {
        this.silentThreshold = event.data.silenceThreshold;
      }
      if (event.data.silenceDurationThreshold !== undefined) {
        this.silenceDurationThreshold = event.data.silenceDurationThreshold;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];

    if (input.length === 0) return true; // No input, nothing to process

    const samples = input[0];
    let isSilent = true;

    // Checking if any sample exceeds the silence threshold
    for (let i = 0; i < samples.length; i++) {
      if (Math.abs(samples[i]) >= this.silenceThreshold) {
        isSilent = false;
        break;
      }
    }

    // Updating the silence counter based on detection
    if (isSilent) {
      this.silenceCounter += samples.length;
    } else {
      this.silenceCounter = 0;
    }

    // Checking if the accumulated silent samples exceed the duration threshold
    if (this.silenceCounter >= this.silenceDurationThreshold) {
      this.port.postMessage({ isSilent: true });
    } else {
      this.port.postMessage({ isSilent: false });
    }

    return true; // Keep processor alive
  }
}

registerProcessor("silence-detector", SilenceDetector);
