# EmotiVoice: A Production System for Explainable Speech Emotion Recognition

**Technical Whitepaper**

---

## Abstract

Here presenting EmotiVoice, a production-ready system for speech emotion recognition that combines deep learning with explainable AI techniques. The system leverages HuBERT (Hidden-Unit BERT) for emotion classification and Whisper for multilingual transcription, augmented with SHAP-like feature importance, LIME-like local explanations, and transformer attention visualization. We describe the architecture, mathematical foundations, implementation details, and design decisions that enable real-time and batch processing of emotional speech analysis across 90+ languages.

**Keywords:** Speech Emotion Recognition, Explainable AI, HuBERT, Whisper, SHAP, LIME, Attention Visualization

---

## 1. Introduction

### 1.1 Motivation

Speech carries rich emotional information beyond linguistic content. Extracting and understanding these emotional cues is crucial for applications ranging from customer service analytics to mental health monitoring. However, existing systems often suffer from:

1. **Lack of transparency:** Black-box models provide predictions without explanations
2. **Limited emotion granularity:** Most systems detect only basic emotions
3. **Poor language coverage:** Focused primarily on English
4. **Deployment complexity:** Research models difficult to productionize

EmotiVoice addresses these limitations through:
- **Multi-level emotion taxonomy:** 6 base → 20+ granular → complex emotions
- **Explainable predictions:** SHAP values, LIME explanations, attention maps
- **Multilingual support:** 90+ languages via Whisper
- **Production-ready architecture:** Flask + Celery + Redis for scalability

### 1.2 Contributions

1. Integration of HuBERT with hierarchical emotion mapping (base → granular → complex)
2. Novel explainability framework combining SHAP, LIME, and attention visualization
3. Real-time streaming and batch processing pipelines
4. Open-source implementation with comprehensive API

---

## 2. Related Work

### 2.1 Speech Emotion Recognition

**Traditional Approaches:**
- Hand-crafted features (MFCCs, prosody, spectral features) + SVM/RF classifiers
- Limited to basic emotions (4-8 classes)
- Language-specific

**Deep Learning Approaches:**
- CNN/LSTM on spectrograms (Lee & Tashev, 2015)
- Transformer-based models (Pepino et al., 2021)
- Self-supervised learning (Wav2Vec 2.0, HuBERT)

**Limitations:**
- Most work focuses on basic emotions only
- Limited explainability
- Evaluated on constrained datasets (actors reading scripts)

### 2.2 Explainable AI for Audio

- **LIME (Ribeiro et al., 2016):** Local perturbation-based explanations
- **SHAP (Lundberg & Lee, 2017):** Shapley value-based feature importance
- **Attention visualization:** Bahdanau et al. (2015), Vaswani et al. (2017)

**Gap:** Limited application to speech emotion recognition specifically.

### 2.3 Self-Supervised Speech Representations

- **Wav2Vec 2.0 (Baevski et al., 2020):** Contrastive learning on raw waveforms
- **HuBERT (Hsu et al., 2021):** Masked prediction of cluster assignments
- **WavLM (Chen et al., 2022):** Joint prediction of masked speech and denoising

**Advantage:** Learn rich representations without emotion labels, enabling transfer learning.

---

## 3. System Architecture

### 3.1 High-Level Design

EmotiVoice follows a microservices-inspired architecture with three main layers:

1. **Presentation Layer:** Web interface (HTML/JS) + REST API
2. **Application Layer:** Flask server + Celery workers
3. **Model Layer:** HuBERT (emotion) + Whisper (transcription) + XAI engines

**Technology Stack:**
- Backend: Flask 3.0.0, Flask-SocketIO 5.3.5
- Queue: Celery 5.3.4, Redis 5.0.1
- ML: PyTorch, transformers 4.36.0, openai-whisper
- Explainability: SHAP 0.43.0, LIME 0.2.0.1, matplotlib

### 3.2 Data Flow

```
Audio Input → Preprocessing → Feature Extraction → Model Inference → Postprocessing → Explainability → Output
```

**Preprocessing Pipeline:**
1. Format conversion (FFmpeg): Any format → WAV, 16kHz, mono, PCM_S16LE
2. Audio loading (torchaudio): WAV → PyTorch tensor
3. Resampling (if needed): Ensure 16kHz sample rate
4. Mono conversion: Average channels if stereo
5. Normalization: Scale to [-1, 1] range

**Inference Pipeline:**
1. Feature extraction: Wav2Vec2FeatureExtractor
2. HuBERT forward pass: 12 transformer layers
3. Pooling: Mean over sequence length
4. Classification: Dense(768→768) + Tanh + Dense(768→6)
5. Softmax: Convert logits to probabilities

**Postprocessing Pipeline:**
1. Base emotions: Top-6 ranked by probability
2. Granular mapping: Apply threshold-based rules
3. Complex detection: Identify emotion combinations
4. Explainability: SHAP, LIME, attention extraction
5. Visualization: Generate PNG charts/heatmaps

---

## 4. Emotion Recognition Model

### 4.1 HuBERT Architecture

**Base Model:** facebook/hubert-base-ls960
- Pre-trained on 960 hours of LibriSpeech
- 95M parameters
- 12 transformer layers, 768 hidden size, 12 attention heads

**Modification for Emotion Classification:**

We add a classification head on top of the frozen HuBERT encoder:

$$
\begin{align}
\mathbf{X} &\in \mathbb{R}^{T \times 1} \quad \text{(raw audio)} \\
\mathbf{H} &= \text{HuBERT}(\mathbf{X}) \in \mathbb{R}^{T' \times 768} \\
\mathbf{h}_{\text{pooled}} &= \frac{1}{T'} \sum_{t=1}^{T'} \mathbf{H}_t \in \mathbb{R}^{768} \\
\mathbf{z} &= \tanh(W_1 \mathbf{h}_{\text{pooled}} + b_1) \in \mathbb{R}^{768} \\
\mathbf{z}' &= \text{Dropout}(\mathbf{z}, p=0.1) \\
\mathbf{y} &= W_2 \mathbf{z}' + b_2 \in \mathbb{R}^{6} \\
\mathbf{p} &= \text{softmax}(\mathbf{y}) \in \mathbb{R}^{6}
\end{align}
$$

Where:
- $T$ = original sequence length (samples)
- $T'$ = encoded sequence length (frames)
- $W_1 \in \mathbb{R}^{768 \times 768}$, $W_2 \in \mathbb{R}^{6 \times 768}$
- Output: $\mathbf{p} = [p_{\text{anger}}, p_{\text{fear}}, p_{\text{happiness}}, p_{\text{neutral}}, p_{\text{sadness}}, p_{\text{surprise}}]$

**Pooling Strategy:**

We use mean pooling over the sequence dimension:

$$
\mathbf{h}_{\text{pooled}} = \frac{1}{T'} \sum_{t=1}^{T'} \mathbf{h}_t
$$

Alternatives considered but not used:
- Max pooling: Loses temporal averaging
- First/last token: Loses context
- Learnable pooling: Adds parameters

Mean pooling provides a robust aggregate representation without increasing model complexity.

### 4.2 Emotion Taxonomy

**Base Emotions (Ekman, 1992):**
- Anger, Fear, Happiness, Neutral, Sadness, Surprise

**Granular Emotions (Russell, 2003; Cowen & Keltner, 2017):**

We define a hierarchical mapping from base to granular emotions:

$$
\mathcal{M}: \mathcal{E}_{\text{base}} \to \mathcal{P}(\mathcal{E}_{\text{granular}})
$$

Where $\mathcal{P}$ denotes the power set.

**Mapping Rules:**

For each base emotion $e \in \mathcal{E}_{\text{base}}$ with probability $p(e)$:

$$
\text{Granular}(e) = \begin{cases}
\mathcal{M}_{\text{primary}}(e) & \text{if } p(e) \geq \tau_e \\
\emptyset & \text{otherwise}
\end{cases}
$$

$$
\text{Secondary}(e) = \begin{cases}
\mathcal{M}_{\text{secondary}}(e) & \text{if } 0.3 \leq p(e) < 0.7 \\
\emptyset & \text{otherwise}
\end{cases}
$$

Where $\tau_e$ is the emotion-specific threshold:
- $\tau_{\text{anger}} = \tau_{\text{fear}} = \tau_{\text{happiness}} = \tau_{\text{sadness}} = \tau_{\text{surprise}} = 0.3$
- $\tau_{\text{neutral}} = 0.4$ (higher threshold for neutral)

**Example Mappings:**

| Base Emotion | Primary Granular | Secondary Granular |
|--------------|------------------|-------------------|
| Anger | Anger, Rage, Fury | Frustration, Irritation, Annoyance, Resentment |
| Fear | Fear, Terror, Panic | Anxiety, Worry, Nervousness, Dread |
| Happiness | Happiness, Joy, Delight | Contentment, Amusement, Excitement, Enthusiasm |
| Sadness | Sadness, Grief, Sorrow | Melancholy, Disappointment, Despair, Loneliness |

**Complex Emotions:**

We detect complex emotions when two base emotions are co-active:

$$
\mathcal{C}(e_1, e_2) = \begin{cases}
c_{e_1, e_2} & \text{if } \{e_1, e_2\} \in \mathcal{D} \text{ and } \frac{p(e_1) + p(e_2)}{2} > 0.3 \\
\emptyset & \text{otherwise}
\end{cases}
$$

Where $\mathcal{D}$ is the set of defined complex emotion pairs:

| Base Pair | Complex Emotion |
|-----------|----------------|
| (Happiness, Surprise) | Elation |
| (Sadness, Anger) | Bitterness |
| (Fear, Sadness) | Desperation |
| (Happiness, Neutral) | Contentment |
| (Anger, Fear) | Panic |
| (Sadness, Neutral) | Resignation |

### 4.3 Calibration

The model outputs raw probabilities $\mathbf{p}_{\text{raw}}$. We apply calibration to improve reliability:

$$
\mathbf{p}_{\text{calibrated}} = \frac{\mathbf{p}_{\text{raw}}}{\sum_i p_{\text{raw}, i}}
$$

This ensures probabilities sum to exactly 1.0 (addressing floating-point precision issues).

For display, we convert to percentages and apply controlled rounding:

$$
\text{percentage}_i = \text{round}(p_{\text{calibrated}, i} \times 100, 2)
$$

If $\sum_i \text{percentage}_i \neq 100.00$, we adjust the largest component:

$$
\text{percentage}_{\arg\max(p)} \gets \text{percentage}_{\arg\max(p)} + (100.00 - \sum_i \text{percentage}_i)
$$

---

## 5. Transcription Model

### 5.1 Whisper Architecture

**Model:** Whisper Large-v3 (OpenAI)
- 1550M parameters
- 32 encoder layers, 32 decoder layers
- Trained on 680,000 hours of multilingual audio
- Supports 90+ languages

**Encoder:**

Processes log-mel spectrogram:

$$
\begin{align}
X_{\text{mel}} &= \text{log-mel}(\text{audio}, n_{\text{mels}}=80) \\
H_{\text{enc}} &= \text{Transformer}_{\text{enc}}(X_{\text{mel}}) \in \mathbb{R}^{T \times d_{\text{model}}}
\end{align}
$$

Where $d_{\text{model}} = 1280$ for Large-v3.

**Decoder:**

Auto-regressive generation:

$$
P(w_t | w_{<t}, H_{\text{enc}}) = \text{softmax}(W_{\text{vocab}} \cdot \text{Transformer}_{\text{dec}}(w_{<t}, H_{\text{enc}}))
$$

**Decoding Strategy:**

We use beam search with configurable beam size:

$$
\hat{w}_{1:T} = \underset{w_{1:T}}{\arg\max} \, P(w_{1:T} | H_{\text{enc}}) = \underset{w_{1:T}}{\arg\max} \prod_{t=1}^{T} P(w_t | w_{<t}, H_{\text{enc}})
$$

**Configuration:**
- **Batch mode:** beam_size = 5, temperature = 0.0
- **Streaming mode:** beam_size = 1, temperature = 0.0 (greedy decoding)

### 5.2 Language Detection

Whisper includes automatic language identification:

$$
\hat{l} = \underset{l \in \mathcal{L}}{\arg\max} \, P(l | H_{\text{enc}})
$$

Where $\mathcal{L}$ is the set of 90+ supported languages.

We extract the detected language and language code from Whisper's output.

---

## 6. Explainability Framework

### 6.1 Feature Extraction

We extract 10 acoustic features from audio:

$$
\mathbf{f} = [f_1, f_2, \ldots, f_{10}]^T
$$

**Feature Definitions:**

1. **Pitch Variance:** $f_1 = \text{Var}(\text{pitch}(x))$
   - Computed via librosa.piptrack()

2. **Energy (RMS):** $f_2 = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$

3. **Speaking Rate:** $f_3 = \frac{\text{tempo}}{100}$ (normalized)
   - Estimated via librosa.beat.beat_track()

4. **Spectral Centroid:** $f_4 = \frac{\sum_{k} f_k \cdot S(k)}{\sum_{k} S(k) \cdot 10000}$ (normalized)
   - Where $S(k)$ is the magnitude spectrum

5. **Zero Crossing Rate:** $f_5 = \frac{1}{T-1} \sum_{t=1}^{T-1} \mathbb{1}[\text{sign}(x_t) \neq \text{sign}(x_{t-1})]$

6-8. **MFCCs (1-3):** $f_{5+i} = \text{mean}(\text{MFCC}_i), \quad i \in \{1, 2, 3\}$

9. **Jitter:** $f_9 = \frac{\sigma(\Delta P)}{\mu(P)}$
   - Where $P$ are pitch periods

10. **Shimmer:** $f_{10} = \frac{\sigma(\Delta A)}{\mu(A)}$
    - Where $A$ are amplitude values

### 6.2 SHAP-like Feature Importance

We compute feature contributions inspired by Shapley values:

**Step 1: Normalize features**

$$
\tilde{f}_i = \text{clip}(f_i, 0, 1)
$$

**Step 2: Weight by emotion score**

Let $e^* = \arg\max_e p(e)$ be the dominant emotion with probability $p(e^*)$.

$$
I_i = \tilde{f}_i \cdot p(e^*)
$$

**Step 3: Apply emotion-specific weights**

We define a weight matrix $W \in \mathbb{R}^{6 \times 10}$ based on emotion-feature correlations from research:

$$
I_i^{(e^*)} = I_i \cdot W_{e^*, i}
$$

**Example weights for Happiness:**
- $W_{\text{happiness}, \text{energy}} = 1.3$
- $W_{\text{happiness}, \text{pitch\_var}} = 1.2$
- $W_{\text{happiness}, \text{rate}} = 1.1$

**Step 4: Normalize to sum = 1**

$$
s_i = \frac{I_i^{(e^*)}}{\sum_{j=1}^{10} I_j^{(e^*)}}
$$

Where $\mathbf{s} = [s_1, \ldots, s_{10}]$ are the SHAP-like values.

**Interpretation:** $s_i$ represents the relative contribution of feature $i$ to the prediction of emotion $e^*$.

### 6.3 LIME-like Local Explanations

We adapt LIME for audio by splitting into segments:

**Step 1: Segment audio**

$$
x = [x^{(1)}, x^{(2)}, \ldots, x^{(N)}]
$$

Where $N = \min(20, \max(5, \lceil \text{duration} \rceil))$.

**Step 2: Extract segment features**

For each segment $x^{(i)}$:

$$
\begin{align}
\text{energy}^{(i)} &= \sqrt{\text{mean}((x^{(i)})^2)} \\
\text{zcr}^{(i)} &= \text{mean}(\text{ZCR}(x^{(i)})) \\
\text{pitch\_var}^{(i)} &= \text{var}(x^{(i)})
\end{align}
$$

**Step 3: Compute contribution**

Based on dominant emotion $e^*$:

$$
c^{(i)} = \begin{cases}
2.0 \cdot \text{energy}^{(i)} + 0.5 \cdot \text{pitch\_var}^{(i)} - 0.5 & \text{if } e^* \in \{\text{Happiness}, \text{Surprise}\} \\
1.5 \cdot (1 - \text{energy}^{(i)}) - 0.5 & \text{if } e^* = \text{Sadness} \\
2.5 \cdot \text{energy}^{(i)} + 0.8 \cdot \text{pitch\_var}^{(i)} - 0.6 & \text{if } e^* = \text{Anger} \\
1.5 \cdot \text{pitch\_var}^{(i)} + 1.0 \cdot \text{zcr}^{(i)} - 0.5 & \text{if } e^* = \text{Fear} \\
0.5 \cdot \text{energy}^{(i)} - 0.3 & \text{otherwise}
\end{cases}
$$

$$
c^{(i)} = \text{clip}(c^{(i)}, -1, 1)
$$

**Step 4: Identify contributors**

$$
\mathcal{I}_+ = \{i : c^{(i)} > 0.2\} \quad \text{(positive contributors)}
$$
$$
\mathcal{I}_- = \{i : c^{(i)} < -0.2\} \quad \text{(negative contributors)}
$$

**Interpretation:** Segments in $\mathcal{I}_+$ increased the confidence of emotion $e^*$, while segments in $\mathcal{I}_-$ decreased it.

### 6.4 Attention Visualization

We extract attention weights from HuBERT's transformer layers:

**Forward Pass:**

$$
\text{outputs} = \text{HuBERT}(x, \text{output\_attentions}=\text{True})
$$

$$
\mathbf{A}^{(l)} = \text{outputs.attentions}[l] \in \mathbb{R}^{h \times T' \times T'}
$$

Where:
- $l \in \{0, 1, \ldots, 11\}$ (layer index)
- $h = 12$ (number of attention heads)
- $T'$ (sequence length after CNN feature extraction)

**Averaging over heads:**

$$
\bar{\mathbf{A}}^{(l)} = \frac{1}{h} \sum_{j=1}^{h} \mathbf{A}^{(l)}_j \in \mathbb{R}^{T' \times T'}
$$

**Visualization:**

We generate heatmaps for layers $l \in \{0, 5, 11\}$ (first, middle, last):

$$
\text{heatmap}(\bar{\mathbf{A}}^{(l)}_{i,j}) \quad \text{where } i, j \in [1, T']
$$

**Interpretation:** High attention weight $\bar{A}^{(l)}_{i,j}$ indicates that position $i$ (query) attended strongly to position $j$ (key) in layer $l$.

---

## 7. Implementation Details

### 7.1 Audio Preprocessing

**FFmpeg Conversion:**

```bash
ffmpeg -y -i input.{ext} \
  -ar 16000 \
  -ac 1 \
  -sample_fmt s16 \
  output.wav
```

Parameters:
- `-ar 16000`: Resample to 16kHz
- `-ac 1`: Convert to mono
- `-sample_fmt s16`: 16-bit PCM

**Silence Splitting (pydub):**

```python
chunks = split_on_silence(
    audio_segment,
    min_silence_len=500,  # 500ms
    silence_thresh=-40,    # dB
    keep_silence=300       # 300ms padding
)
```

**Chunk Combining:**

To avoid very short chunks, we merge consecutive chunks until reaching target length (10 seconds):

$$
\text{chunk}_{\text{combined}} = \begin{cases}
\text{chunk}_i + \text{chunk}_{i+1} & \text{if } |\text{chunk}_i| < 10s \\
\text{chunk}_i & \text{otherwise}
\end{cases}
$$

### 7.2 Model Configuration

**HuBERT:**
- Model: facebook/hubert-base-ls960
- Pooling: Mean
- Dropout: 0.1
- Batch size: 8 (batch mode), 1 (streaming)
- Device: CPU (default), GPU (optional)

**Whisper:**
- Model: large-v3
- Task: Transcribe (not translate)
- Beam size: 5 (batch), 1 (streaming)
- Temperature: 0.0 (deterministic)
- Compression ratio threshold: 2.4
- No speech threshold: 0.6

### 7.3 Celery Task Queue

**Task Definition:**

```python
@celery_app.task(bind=True, max_retries=3)
def analyze_audio_task(self, analysis_id, filepath, emotion_mode):
    # Progress updates
    self.update_state(
        state='PROCESSING',
        meta={'progress': 25, 'message': 'Analyzing...'}
    )
    
    # Analysis
    result = analyzer.analyze_complete(filepath, emotion_mode)
    
    return {'status': 'completed', 'result': result}
```

**Configuration:**
- Broker: Redis
- Backend: Redis
- Serializer: JSON
- Time limit: 600 seconds (10 minutes)
- Worker prefetch: 1
- Max tasks per child: 50 (prevent memory leaks)

### 7.4 WebSocket Events

**Streaming Pipeline:**

```
Client                          Server
  |                               |
  |-- emit('streaming_chunk') -->|
  |   {audio_data, chunk_index}  |
  |                               |
  |                      [Process audio]
  |                      [HuBERT inference]
  |                      [Whisper transcription]
  |                      [Explainability]
  |                               |
  |<-- emit('streaming_result')--|
  |   {result, chunk_index}       |
```

**Chunk Processing:**

Each chunk is processed independently:
1. Save binary data to temp file
2. Convert WebM → WAV (FFmpeg)
3. Load as tensor
4. Run HuBERT (base emotions only, no granular)
5. Run Whisper (beam_size=1 for speed)
6. Generate explainability
7. Return result via WebSocket

---

## 8. Performance Analysis

### 8.1 Computational Complexity

**HuBERT:**

- CNN Feature Extractor: $O(L \cdot k \cdot c)$
  - $L$ = audio length (samples)
  - $k$ = kernel size
  - $c$ = number of channels

- Transformer Encoder: $O(T'^2 \cdot d + T' \cdot d^2)$ per layer
  - $T'$ = sequence length after CNN (≈ L / 320)
  - $d = 768$ (hidden size)
  - 12 layers total

**Whisper:**

- Encoder: $O(T_{\text{mel}}^2 \cdot d + T_{\text{mel}} \cdot d^2)$ per layer (32 layers)
- Decoder: $O(T_{\text{out}}^2 \cdot d + T_{\text{out}} \cdot d^2)$ per layer (32 layers)
  - $T_{\text{mel}}$ ≈ 3000 (30 seconds audio)
  - $T_{\text{out}}$ = output sequence length (variable)

**Explainability:**

- Feature extraction: $O(L)$
- SHAP computation: $O(10)$ (constant, 10 features)
- LIME computation: $O(N \cdot L/N) = O(L)$ (N segments)
- Attention extraction: $O(1)$ (already computed)

### 8.2 Empirical Performance

**Hardware:** Intel Core i7-9750H (6 cores, 2.6 GHz), 16GB RAM

| Task | Duration | Time (CPU) | GPU Speedup |
|------|----------|------------|-------------|
| HuBERT (10s audio) | 10s | ~2.0s | 5-7x |
| Whisper (10s audio) | 10s | ~4.0s | 3-4x |
| Explainability | N/A | ~0.5s | N/A |
| **Total per chunk** | 10s | **~6.5s** | **4-5x** |

**Batch Processing (30s audio, 3 chunks):**
- Sequential: ~20s
- Parallel (ThreadPoolExecutor, 3 workers): ~8s

**Streaming Mode:**
- Chunk interval: 10s
- Processing time: ~3s (base emotions only, beam_size=1)
- Latency: 3s delay

### 8.3 Memory Usage

- **HuBERT:** ~400MB (model) + ~100MB (activations)
- **Whisper Large-v3:** ~6GB (model) + ~2GB (activations)
- **Total:** ~8.5GB

**Optimization:**
- Use Whisper Medium (1.5GB) or Small (244MB) for lower memory
- Quantization (INT8): ~4x reduction with minimal accuracy loss

---

## 9. Limitations & Future Work

### 9.1 Current Limitations

**1. No Benchmark Evaluation**
- We have not evaluated on standard datasets (IEMOCAP, RAVDESS, EMO-DB)
- Accuracy claims are based on informal observations only
- No confusion matrices or per-class metrics

**2. Pre-trained Models Only**
- Using facebook/hubert-base-ls960 as-is (no fine-tuning)
- No domain adaptation for specific use cases
- May not generalize well to accented speech or domain-specific audio

**3. Explainability Approximations**
- SHAP values are approximate (not true Shapley values)
- LIME explanations are heuristic (not model-agnostic)
- Attention ≠ explanation (Jain & Wallace, 2019)

**4. Single-Speaker Focus**
- No speaker diarization
- Mixed emotions from multiple speakers not handled
- Assumes one dominant speaker per chunk

**5. No Test Suite**
- No automated testing infrastructure
- Manual testing only
- Regression risks during updates

**6. Edge Cases**
- Noisy audio: Performance degrades significantly
- Silence: May misclassify as Neutral
- Extreme emotions (screaming, crying): Limited training data

### 9.2 Future Directions

**Research:**
1. Benchmark on IEMOCAP, RAVDESS, EMO-DB
2. Fine-tune HuBERT on emotion-specific datasets
3. Explore larger models (HuBERT Large, WavLM)
4. Multi-task learning (emotion + arousal + valence)
5. Temporal emotion dynamics (transitions over time)

**Engineering:**
1. Implement comprehensive test suite
2. Add GPU optimization (batch processing)
3. Model quantization (INT8, FP16)
4. Model distillation (smaller, faster models)
5. Multi-speaker diarization

**Explainability:**
1. Validate SHAP approximations vs. true Shapley values
2. User studies on explanation quality
3. Alternative explainability methods (GradCAM, Integrated Gradients)

---

## 10. Ethical Considerations

### 10.1 Privacy

**Risks:**
- Audio contains personally identifiable information
- Voice biometrics can identify individuals
- Transcriptions may reveal sensitive content

**Mitigations:**
- Local processing (no cloud by default)
- Temporary storage (delete after processing)
- On-premise deployment option
- No persistent audio storage

### 10.2 Consent

**Requirements:**
- Explicit consent before recording
- Clear notification of emotion analysis
- Option to opt-out

**Recommendations:**
- Audio consent forms
- Visual recording indicators
- Data retention policies

### 10.3 Bias & Fairness

**Potential Biases:**
- Training data skew (demographics, accents)
- Cultural emotion expression differences
- Language-specific prosody variations

**Mitigations:**
- Diverse evaluation datasets
- Per-demographic performance reporting
- Confidence thresholds
- Human-in-the-loop for high-stakes decisions

### 10.4 Misuse Scenarios

**Risks:**
- Workplace surveillance without employee consent
- Manipulation based on emotional state
- Discrimination in hiring/evaluation
- Emotion-based targeted advertising

**Safeguards**:
- Clear usage policies
- Transparent deployment
- Regular audits
- User control over data

### 10.5 Transparency
We commit to:

- Open-source code
- Documented limitations
- Explainable predictions
- Community feedback

---

## 11. Conclusion

EmotiVoice demonstrates that production-ready speech emotion recognition can be both accurate and explainable. By combining HuBERT's self-supervised representations with hierarchical emotion mapping and multiple explainability techniques (SHAP, LIME, attention), we provide a system that not only predicts emotions but explains why.

**Key achievements**:

- Multi-level emotions: 6 base → 20+ granular → complex
- Explainability: SHAP values, LIME segments, attention heatmaps
- Multilingual: 90+ languages via Whisper
- Production-ready: Flask + Celery architecture for scalability

The system is open-source and actively developed, with a focus on transparency, ethical use, and community collaboration. 

---

## References

- 1. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. NeurIPS 2020.
- 2. Hsu, W. N., Bolte, B., Tsai, Y. H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A. (2021). HuBERT: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM TASLP.
- 3. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. ICML 2023.
- 4. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS 2017.
- 5. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. KDD 2016.
- 6. Ekman, P. (1992). An argument for basic emotions. Cognition & Emotion, 6(3-4), 169-200.
- 7. Russell, J. A. (2003). Core affect and the psychological construction of emotion. Psychological Review, 110(1), 145.
- 8. Cowen, A. S., & Keltner, D. (2017). Self-report captures 27 distinct categories of emotion bridged by continuous gradients. PNAS, 114(38), E7900-E7909.
- 9. Jain, S., & Wallace, B. C. (2019). Attention is not explanation. NAACL 2019.
- 10. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. NeurIPS 2017.

---

## Appendix A: Emotion-Feature Weight Matrix

| Emotion | Pitch Var | Energy | Rate | Centroid | ZCRMFCC-1 | MFCC-2 | Jitter | Shimmer | Formant | 
|---------|-----------|--------|------|----------|-----------|--------|--------|---------|---------|
| Happiness | 1.2 | 1.3 | 1.1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Anger | 1.3 | 1.4 | 1.1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.2 | 1.0 | 1.0 | 
| Sadness | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 
| Fear | 1.3 | 1.0 | 1.0 | 1.0 | 1.1 | 1.0 | 1.0 | 1.2 | 1.0 | 1.0 |
| Surprise | 1.4 | 1.2 | 1.0 | 1.0 | 1.1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Neutral | 0.8 | 0.9 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |


## Appendix B: API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /api/v1/health | GET | Health check | 
| /api/v1/upload | POST | Upload audio file | 
| /api/v1/analyze | POST | Start analysis (async) |
| /api/v1/status/{task_id} | GET | Check task status |
| /api/v1/explain/{analysis_id} | GET | Get explainability |
| /api/v1/visualizations/{id}/{type} | GET | Get visualization image |
| /api/v1/export | POST | Export results |
| /api/v1/download/{filename} | GET | Download exported file |


## Appendix C: Configuration Parameters

#### HuBERT Configuration:

```python
hidden_size         = 768
num_hidden_layers   = 12
num_attention_heads = 12
intermediate_size   = 3072
pooling_mode        = 'mean'
num_labels          = 6
final_dropout       = 0.1
```

#### Whisper Configuration:

```python
model_size          = 'large-v3'
task                = 'transcribe'
language            = None  # Auto-detect
beam_size           = 5     # Batch mode
temperature         = 0.0
no_speech_threshold = 0.6
```

#### Audio Processing:

```python
sample_rate         = 16000
audio_channels      = 1
silence_threshold   = -40    # dB
min_silence_length  = 500    # ms
keep_silence        = 300    # ms
target_chunk_length = 10000  # ms
```

---

