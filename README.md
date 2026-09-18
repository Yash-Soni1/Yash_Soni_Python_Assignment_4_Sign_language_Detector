# Sign Language Detector (TensorFlow + MediaPipe)

This package contains scripts to capture a dataset, train a TensorFlow classifier, and run real-time inference using MediaPipe hand detection.

## Structure
```
sign-language-detector/
├── README.md
├── requirements.txt
├── capture_dataset.py
├── train.py
├── realtime_infer.py
├── utils/
│   ├── label_utils.py
│   └── mediapipe_hand_crop.py
├── sample_dataset_structure.txt
└── models/
    └── (empty initially)
```

## Image size
This package uses a balanced image size of **160x160** for training and inference.

## Quick setup
1. Create virtual env and activate:

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows PowerShell
# .\venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Capture dataset:
```bash
python capture_dataset.py --out_dir dataset --img_size 160
```
- Press keys 0..6 to switch label (mapping shown in UI)
- Press `c` to capture a frame for current label
- Press `q` to quit

4. Train:
```bash
python train.py --data_dir dataset --img_size 160 --epochs 12 --batch 32
```

5. Run real-time inference:
```bash
python realtime_infer.py --model models/signclassifier.h5 --img_size 160
```

## Notes
- `models/` will contain trained model(s) after training.
- `utils/label_utils.py` provides label saving/loading to ensure consistent mapping.
- The capture script uses MediaPipe to crop hand regions, so dataset images focus on hands.

Have fun!