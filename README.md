<div align="center">
  <img src="https://github.com/user-attachments/assets/22cc8c54-f3c7-4900-a9db-3e37fffac5ad" alt="Background Image" width="95%" />
</div>

<h1 align="center">Group Activity Recognition</h1>

<p align="center">
  A modern implementation of the CVPR 2016 paper, <em>"A Hierarchical Deep Temporal Model for Group Activity Recognition."</em> 
  This model employs a two-stage LSTM architecture to recognize group activities by capturing both individual and group-level temporal dynamics.
</p>

<p align="rigth"><strong>Key updates include:</strong></p>
<ul align="rigth">
  <li>ResNet50 for feature extraction (replacing AlexNet).</li>
  <li>Ablation studies to analyze model components.</li>
  <li>Implementation of an end-to-end version (Baseline 9).</li>
  <li>Achieve higher performance across every model baseline compared to the original paper.</li>
  <li>Full implementation in Python (original used Caffe).</li>
</ul>

-----
## Usage

---

### 1. Clone the Repository
```bash
git clone https://github.com/Sh-31/Group-Activity-Recognition.git
```

### 2. Install the Required Dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Download the Model Checkpoint
This is a manual step that involves downloading the model checkpoint files.

#### Option 1: Use Python Code
Replace the `modeling` folder with the downloaded folder:
```python
import kagglehub

# Download the latest version
path = kagglehub.model_download("sherif31/group-activity-recognition/pyTorch/v1")

print("Path to model files:", path)
```

#### Option 2: Download Directly
Browse and download the specific checkpoint from Kaggle:  
[Group Activity Recognition - PyTorch Checkpoint](https://www.kaggle.com/models/sherif31/group-activity-recognition/pyTorch/v1/1)

-----
## Dataset Overview

The dataset was created using publicly available YouTube volleyball videos. The authors annotated 4,830 frames from 55 videos, categorizing player actions into 9 labels and team activities into 8 labels. 

### Example Annotations

![image](https://github.com/user-attachments/assets/50f906ad-c68c-4882-b9cf-9200f5a380c7)

- **Figure**: A frame labeled as "Left Spike," with bounding boxes around each player, demonstrating team activity annotations.

![image](https://github.com/user-attachments/assets/cca9447a-8b40-4330-a11d-dbc0feb230ff)

### Train-Test Split

- **Training Set**: 3,493 frames
- **Testing Set**: 1,337 frames

### Dataset Statistics

#### Group Activity Labels
| Group Activity Class | Instances |
|-----------------------|-----------|
| Right set            | 644       |
| Right spike          | 623       |
| Right pass           | 801       |
| Right winpoint       | 295       |
| Left winpoint        | 367       |
| Left pass            | 826       |
| Left spike           | 642       |
| Left set             | 633       |

#### Player Action Labels
| Action Class | Instances |
|--------------|-----------|
| Waiting      | 3,601     |
| Setting      | 1,332     |
| Digging      | 2,333     |
| Falling      | 1,241     |
| Spiking      | 1,216     |
| Blocking     | 2,458     |
| Jumping      | 341       |
| Moving       | 5,121     |
| Standing     | 38,696    |

### Dataset Organization

- **Videos**: 55, each assigned a unique ID (0–54).
- **Train Videos**: 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54.
- **Validation Videos**: 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51.
- **Test Videos**: 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47.

### Dataset Download Instructions

1. Enable Kaggle's public API. Follow the guide here: [Kaggle API Documentation](https://www.kaggle.com/docs/api).  
2. Use the provided shell script:
```bash
  chmod 600 .kaggle/kaggle.json 
  chmod +x script/script_download_volleball_dataset.sh
  .script/script_download_volleball_dataset.sh
```
For further information about dataset, you can check out the paper author's repository:  
[link](https://github.com/mostafa-saad/deep-activity-rec)

-----
## [Ablation Study](https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)#:~:text=In%20artificial%20intelligence%20(AI)%2C,resultant%20performance%20of%20the%20system)

### Baselines

1. **Image Classification:**  
   A straightforward image classifier based on ResNet-50, fine-tuned to classify group activities using a single frame from a video clip.

3. **Fine-tuned Person Classification:**  
   The ResNet-50 CNN model is deployed on each person. Feature extraction for each crop 2048 features are pooled over all people and then fed to a softmax classifier to recognize group activities in a single frame.

4. **Temporal Model with Image Features:**  
   A temporal model that uses image features per clip. Each clip consists of 9 frames, and an LSTM is trained on sequences of 9 steps for each clip.

5. **Temporal Model with Person Features:**  
   A temporal extension of the previous baseline (B3), where person-specific features pooled over all individuals in each frame are fed to an LSTM model to capture group dynamics.

6. **Temporal Model with Person Features (B5):**  
  Individual features pooled over all people are fed into an LSTM model to recognize group activities.

7. **Two-stage Model without LSTM 1:**  
   The full model (V1) trains an LSTM on crop-level data (LSTM on a player level). Clips are extracted: sequences of 9 steps per player for each frame. A max-pooling operation is applied to the players, and LSTM 2 is trained on the frame level.

8. **Two-stage Model without LSTM 2:**  
   The full model (V2) trains an LSTM on crop-level data (LSTM on a player). Clips are extracted as sequences of 9 steps per player for each frame. A max-pooling operation is applied to each player's team in a dependent way. Features from both teams are concatenated along the feature dimension, and the result is fed to LSTM 2 at the frame level.

9. **Unified Two-stage Model (Baseline 9):**  
   In previous baselines, person-level and group-level activity losses were handled independently, resulting in a two-stage model. Baseline 9 combines these processes into a unified, end-to-end training pipeline. This approach allows for the simultaneous optimization of both person-level and group-level activity classification using a shared gradient flow.

---
## Performance comparison

### Original Paper Baselines Score

![Original Paper Scores](https://github.com/user-attachments/assets/591cd10d-d767-4868-847b-727c9292f435)

### My Scores (Accuracy and F1 Scores)

| **Baseline** | **Accuracy** | **F1 Score** |
|--------------|--------------|--------------|
| Baseline 1   | 72.66%       | 72.63%       |
| Baseline 3   | 80.25%       | 80.24%       |
| Baseline 4   | 73.45%       | 73.27%       |
| Baseline 5   | 77.04%       | 77.07%       |
| Baseline 6   | 84.52%       | 83.99%       |
| Baseline 7   | 88.71%       | 88.77%       |
| Baseline 8   | 91.40%       | 91.39%       |
| Baseline 9   | 98.47%       | 98.47%       |

---

## Interesting Observations

### Effect of Team Independent Pooling

The following confusion matrices from Baseline 5 and Baseline 6 reveal some interesting insights:

#### Baseline 5 Confusion Matrix
<img src="modeling/baseline%205/outputs/Group_Activity_Baseline_5_eval_on_testset_confusion_matrix.png" alt="Baseline 5 confusion matrix" width="60%">

#### Baseline 6 Confusion Matrix
<img src="modeling/baseline%206/outputs/Group_Activity_Baseline_6_eval_on_testset_confusion_matrix.png" alt="Baseline 6 confusion matrix" width="60%">

- The most frequent confusions occur between:
  - Right winpoint vs. left winpoint
  - Right pass vs. left pass
  - Right set vs. left set
  - Right spike vs. left spike

This behavior is likely due to the pooling of the 12 players from both teams when transitioning from the individual/personal level to the frame/group level. By grouping all players into one unit, the model loses valuable geometric information regarding player positions. 

When the teams are grouped and processed individually before concatenation, the player position information is retained. This suggests that a more careful handling of player positions could improve model performance, as observed in Baseline 8 and Baseline 9.

#### Baseline 8 Confusion Matrix
<img src="modeling/baseline%208/outputs/Group_Activity_Baseline_8_eval_on_testset_confusion_matrix.png" alt="Baseline 8 confusion matrix" width="60%">

#### Baseline 9 Confusion Matrix
<img src="modeling/baseline 9 (end to end)/outputs/Group_Activity_Baseline_9_eval_on_testset_confusion_matrix.png" alt="Baseline 9 confusion matrix" width="60%">

--- 
