# cascad-ai
The goal of this project is to develop a system that automatically determines the score of a single player's Cascadia game board from a photograph of the completed environment.

## Creating a Dataset

I want to automaically create and annotate a dataset to train a YOLO model.

[How to Create a Dataset for a YOLO model](https://docs.ultralytics.com/yolov5/tutorials/train-custom-data#2-select-a-model)

## Object Detection Strategies
- DNN (YOLO)
- Circle Detection 
    - [Hough Circle Transform](https://docs.opencv.org/3.4.20/d4/d70/tutorial_hough_circle.html)    
- KMeans with color and Distance

## Token Detector Model

Trained on 2000 Artifical images made with the dataset generator tool. Model was trained within 5 epochs. Trained with imgsz param set to 800,

Best model is saved at: `src/cascadai/models/token_detector_yolo11n/TokenNet.pt`

The model has the yolo11n architecture.

Notes on Perfomance:
- really good performance for image size = 1024 and confidence threshold > 0.7


## Run Scoring Script

The `src/cascadai/game_scorer.py` script detects animal tokens in a photo of a Cascadia board, builds their adjacency graph, and prints the score for each animal (using the selected scoring card) plus the total.

Run it from the repo root with the project virtual environment (it imports the installed `cascadai` package):

```bash
.venv/bin/python src/cascadai/game_scorer.py -i <path-to-image>
```

### Options

| Flag | Long        | Default    | Description                              |
|------|-------------|------------|------------------------------------------|
| `-i` | `--image`   | *(required)* | Path to the input board photo          |
| `-o` | `--output`  | `output`   | Output directory (see note below)        |
| `-br`| `--bear`    | `A`        | Bear scoring card (A/B/C/D)              |
| `-ek`| `--elk`     | `A`        | Elk scoring card (A/B/C/D)               |
| `-sm`| `--salmon`  | `A`        | Salmon scoring card (A/B/C/D)            |
| `-hk`| `--hawk`    | `A`        | Hawk scoring card (A/B/C/D)              |
| `-fx`| `--fox`     | `A`        | Fox scoring card (A/B/C/D)               |

### Example

```bash
.venv/bin/python src/cascadai/game_scorer.py \
  -i datasets/real_games/1.jpg \
  -br B -ek A -sm D -hk C -fx B
```

### Output

The script prints a score line per animal and the total, for example `Score for Bears [B]: 4` and `Total Score for 22`. It also writes a token environment-graph plot to `output/env_graph_tokens.png`.

Notes:
- The `-o/--output` flag is currently unused by the script; the environment-graph plot always saves to `output/`.
- Scoring depends on the trained YOLO token detector. Some score-card functions (e.g. Elk A/B/D, Hawks B/C/D) are still placeholders that return `0`, so results for those entries are not yet complete.

## Ideas 
- remove background to prevent token incorrect token detection
- use hough circle transformation to find locations of tokens
- use a classifier (Auto encoder?) to classify regions where tokens were found 
- use detected tokens to build a graph
- use the detected tokens to sample local hexagons
- train a classifier to classify a hexagon based on local sample region, which is occluded/coverd by a token / or not.