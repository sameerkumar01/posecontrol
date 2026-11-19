# Pose-Controlled Fighting Game (MediaPipe & LSTM)

A real-time, vision-driven fighting game built with **Python, TensorFlow, MediaPipe BlazePose, OpenCV, NumPy, and Scikit-learn**. The system extracts 33 3D skeletal landmarks from a commodity webcam, applies invariant biomechanical normalizations, classifies dynamic temporal gestures using a causal 2-layer LSTM neural network, and executes deterministic 60 Hz fighting game combat against an intelligent AI opponent.

---

## 🥊 Key System Features

1. **MediaPipe BlazePose 3D Tracking:**
   - Tracks 33 full-body landmarks $(x, y, z, v)$ at 30 FPS.
   - Low-confidence filtering and first-order kinematic forward imputation.

2. **3-Stage Biomechanical Normalization Pipeline:**
   - **Translation Invariance:** Centers skeletal origin to the mid-hip root anchor:
     $$\mathbf{P}'_i = \mathbf{P}_i - \frac{\mathbf{P}_{23} + \mathbf{P}_{24}}{2}$$
   - **Scale Invariance:** Normalizes by Euclidean torso length ($S_{\text{torso}} = \|\mathbf{P}_{\text{mid\_shoulder}} - \mathbf{P}_{\text{mid\_hip}}\|_2$).
   - **Coronal Yaw Rotation Invariance:** Evaluates shoulder angle $\theta_{\text{yaw}} = \operatorname{atan2}(\Delta z, \Delta x)$ and applies $R_y(-\theta_{\text{yaw}})$ to align body orientation.

3. **138-Dimensional Spatio-Temporal Feature Vector:**
   - **99 Positional Features:** Normalized $(x, y, z)$ coordinates for all 33 landmarks.
   - **33 First-Order Velocities:** Finite-difference momentum $\frac{\mathbf{P}(t) - \mathbf{P}(t-1)}{\Delta t}$ for 11 key strike landmarks (wrists, elbows, shoulders, hips, nose).
   - **6 Biomechanical Joint Angles:** Interior flexion angles for left/right elbows, shoulders, and hips.

4. **Causal 2-Layer LSTM Sequence Classifier (~190k parameters):**
   - Strictly causal and unidirectional: Input `(30, 138)` $\to$ Masking $\to$ LSTM(128) $\to$ LayerNorm $\to$ LSTM(64) $\to$ BatchNorm $\to$ Dense(64, ReLU) $\to$ Softmax(4).
   - Multi-class **Focal Loss** ($\gamma = 2.0$) with cost-sensitive inverse class weighting to balance dominant idle frames against active strikes.
   - Sub-5ms CPU forward pass.

5. **Real-Time Conditioning & Refractory Debounce Engine:**
   - **Dual-Rate EMA Smoothing:** Fast tracking ($\alpha = 0.65$) for strikes, heavy smoothing ($\alpha = 0.40$) for steady guard/idle.
   - **Class-Specific Thresholds:** JAB ($0.80$), UPPERCUT ($0.85$), BLOCK ($0.75$).
   - **Multi-Frame Persistence Hysteresis:** 2-3 frame consecutive activation confirmation.
   - **Refractory Mechanical Cooldown:** 450 ms lockout for Jab, 650 ms for Uppercut.

6. **Deterministic 60 Hz Combat Engine & Arcade HUD:**
   - Standard fighting game frame data: Startup $\to$ Active $\to$ Recovery $\to$ Neutral.
   - Counter-hit damage multiplier ($1.35\times$) when catching opponent during windup.
   - Guard mitigation ($80\%$ damage reduction on block).
   - Dynamic neon skeleton wireframe, glowing hitboxes, procedural impact sparks, screen shake, and combo counters.
   - Heuristic AI opponent with 3 difficulty tiers (Easy, Medium, Hard).

---

## 📁 Repository Structure

```
.
├── SRS.md                              # Formal ISO/IEC/IEEE 29148 / IEEE 830 Specification
├── docs/
│   └── technical_architecture.md       # Complete Deep System Architecture Document
├── pose_control/
│   ├── config.py                       # Global constants, thresholds, frame data, colors
│   ├── vision/
│   │   ├── normalizer.py               # 3-step invariant biomechanical normalizer
│   │   └── pose_detector.py            # MediaPipe BlazePose wrapper
│   ├── features/
│   │   ├── extractor.py                # 138-dim kinematic feature extractor
│   │   └── buffer.py                   # 30-frame circular FIFO temporal buffer
│   ├── model/
│   │   └── network.py                  # Keras LSTM topology, Focal Loss, NumPy evaluator
│   ├── dataset/
│   │   ├── collector.py                # Interactive webcam gesture recording CLI
│   │   ├── augmentor.py                # Spatial, temporal, and noise data augmentation
│   │   └── generator.py                # Synthetic pose generator & dataset builder
│   ├── inference/
│   │   ├── postprocessor.py            # EMA filter, hysteresis, and debounce gate
│   │   └── engine.py                   # Real-time gesture inference engine
│   ├── game/
│   │   ├── fsm.py                      # Fighter Finite State Machine (Startup/Active/Recovery)
│   │   ├── ai_opponent.py              # AI Opponent with 3 difficulty levels
│   │   └── combat_engine.py            # 60 Hz combat loop, collisions & damage
│   └── ui/
│       └── hud.py                      # Arcade OpenCV HUD, wireframe & particle sparks
├── tests/
│   ├── test_normalizer.py              # Normalizer invariant tests
│   ├── test_features.py                # 138-dim feature and buffer tests
│   └── test_model_and_combat.py        # LSTM, FSM, and combat collision tests
├── train.py                            # Model training and validation script
├── main.py                             # Main application entrypoint
├── requirements.txt                    # Project dependency specification
└── README.md
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- macOS (Apple Silicon / Intel), Linux, or Windows
- Commodity USB / Built-in Webcam (or run in Synthetic Demo Mode without a webcam)

### 2. Environment Setup
```bash
# Clone or navigate to the repository
cd "pose control"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎮 Running the Game

### Live Webcam Combat Mode
```bash
python main.py --camera 0 --model saved_models/combat_lstm_best.keras
```

### Synthetic Demo Mode (No Webcam Required!)
Run the game immediately using the built-in biomechanical pose stream simulator:
```bash
python main.py --demo --model saved_models/combat_lstm_best.keras
```

### In-Game Controls
| Key | Action |
|---|---|
| `[SPACE]` | Reset / Start Next Round |
| `[1]` | Set AI Difficulty to **Easy** |
| `[2]` | Set AI Difficulty to **Medium** |
| `[3]` | Set AI Difficulty to **Hard** |
| `[J]` | Manual Jab strike |
| `[U]` | Manual Uppercut strike |
| `[B]` | Manual High Guard block |
| `[D]` | Toggle Synthetic Demo Mode / Live Webcam |
| `[Q]` or `[ESC]` | Exit Game |

---

## 🧪 Training & Verification

### Run Automated Unit Tests
```bash
pytest -v tests/
```

### Train the LSTM Sequence Model
To bootstrap training on the balanced augmented synthetic dataset:
```bash
python train.py --epochs 30 --batch-size 32
```
Trained weights and `.keras` models are automatically saved into `saved_models/`.

### Record Custom Gesture Data via Webcam
To record custom user sequences (e.g. 20 samples of JAB):
```bash
python -m pose_control.dataset.collector --action JAB --samples 20 --subject player_01
```

---

## 📊 Requirements Compliance
This implementation strictly satisfies all functional, interface, and performance specifications established in:
- [SRS.md](file:///Users/sameerkumar/Desktop/pose%20control/SRS.md)
- [docs/technical_architecture.md](file:///Users/sameerkumar/Desktop/pose%20control/docs/technical_architecture.md)
