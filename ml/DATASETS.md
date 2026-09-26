# Models and datasets

Nothing listed here is committed to the repository. Scripts download into `data/` (gitignored):
`make model` for YAMNet, `make datasets` for the training data. Licences were checked on
2026-09-26 against each source's own page.

## Models

| Item | Source | Licence | Commercial use | Used for |
|---|---|---|---|---|
| YAMNet float32 TFLite | `storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite` (Google MediaPipe) | Apache-2.0 | Yes | Step A class scores; Step B embeddings |
| YAMNet class map | `tensorflow/models` `research/audioset/yamnet/yamnet_class_map.csv` | Apache-2.0 | Yes | Class names in output order |
| Step B head (`data/models/drone_head.onnx`) | trained locally by `make train` | inherits the training data's terms (below) | **No** (see ESC-50) | Step B classifier |

Despite its name, the MediaPipe "float32" YAMNet is int8-quantised inside; only its input and
output are float. Step B reads the 1024-d `layer28/reduce_mean` embedding and dequantises it
(`(q - zero_point) * scale`); this reproduces the model's own 521 class scores to within 0.04.

## Training and evaluation data (Step B)

| Dataset | Source | Licence | Commercial use | Used for |
|---|---|---|---|---|
| DroneNoise Database (Ramos-Romero, Green, Asensio, Torija Martinez; Univ. of Salford, v3 2024) | figshare article 22133411 | CC BY 4.0 | Yes, with attribution | Drone positives: outdoor overflights of four small drones (`3p`, `Fp`, `M3`, `Yn`) in Edzell, Scotland, each flight recorded by up to nine ground microphones, 50 kHz |
| ESC-50 (K. J. Piczak, 2015) | github.com/karolpiczak/ESC-50 | CC BY-NC 3.0 | **No** | Negatives: 2000 five-second clips in 50 classes; hard negatives reported separately: chainsaw, engine, airplane, helicopter, hand saw, vacuum cleaner, insects |

**Consequence:** a head trained with ESC-50 is for research and this demo only. Before any
commercial use it must be retrained on commercially licensed negatives (for example, own
recordings or CC BY / CC0 subsets of FSD50K).

Attribution: *DroneNoise Database*, C. Ramos-Romero, N. Green, C. Asensio, A. J. Torija Martinez,
University of Salford, https://doi.org/10.17866/rd.salford.22133411.v3, CC BY 4.0.
*ESC: Dataset for Environmental Sound Classification*, K. J. Piczak, ACM MM 2015,
https://doi.org/10.7910/DVN/YDEPUT, CC BY-NC 3.0.

### Considered and rejected

| Dataset | Why not |
|---|---|
| DroneAudioDataset (Al-Emadi, GitHub) | No licence is stated, which means all rights are reserved. |
| DADS (Hugging Face `geronimobasso/drone-audio-detection-samples`) | Labelled MIT, but it repackages the unlicensed DroneAudioDataset and non-commercial sets (ESC-50, UrbanSound8K), and its own card says to verify each source's licence. It also has no recording ids, so a split free of leakage is impossible. |
| Zenodo 7779574 (drone fault classification, CC BY 4.0) | Licence is fine, but it is 7.6 GB of close-range lab recordings of faulty rotors: little value for distant outdoor detection. A candidate for later. |

No YouTube or other scraped audio is used.

## Splits (no leakage)

- DroneNoise: one flight is heard by up to nine microphones at once, so every recording of a
  flight lands in the same split. Per drone type, flights are sorted by name and assigned in turn
  to test, validation, train, train. Every drone type appears in every split.
- ESC-50: fold 5 is test, fold 4 is validation, folds 1–3 are training. The folds ship grouped by
  source recording.
- Only the training split is augmented (distance, background mixing at −5…+20 dB SNR, reverb,
  ±3 % pitch drift, gain and time shift). Validation and test are clean.
