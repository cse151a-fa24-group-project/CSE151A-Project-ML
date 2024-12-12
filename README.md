x<div align="center">
<img width="100%" src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/assets/banner_cse151a.png" alt="Main Banner">
</div>

# Milestone 5: Final Submission

## Table of Contents
<details>
<summary> Click here to view full Table of Contents </summary>

- [Milestone 5: Final Submission](#milestone-5-final-submission)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
  - [2. Data Prep](#2-data-prep)
    - [2.1 - Extraction Method](#21---extraction-method)
    - [2.2 - Number of Classes and Example Classes](#22---number-of-classes-and-example-classes)
    - [2.3 - Size and Number of Images](#23---size-and-number-of-images)
      - [2.3.1 - Size of Images](#231---size-of-images)
      - [2.3.2 - Number of Images](#232---number-of-images)
  - [3. Models Methods](#3-models-methods)
    - [3.1 - 1st Model (Simple CNN)](#31---1st-model-simple-cnn)
      - [3.1.1 - Preprocessing](#311---preprocessing)
      - [3.1.2 - Training](#312---training)
    - [3.2 - 2nd Model (ResNet50\_v1)](#32---2nd-model-resnet50_v1)
      - [3.2.1 - Preprocessing](#321---preprocessing)
      - [3.2.2 - Training](#322---training)
    - [3.3 - 3rd Model (ResNet50\_v2)](#33---3rd-model-resnet50_v2)
      - [3.3.1 - Preprocessing](#331---preprocessing)
      - [3.3.2 - Training](#332---training)
    - [3.4 - 4th Model (VGG16)](#34---4th-model-vgg16)
      - [3.4.1 - Preprocessing](#341---preprocessing)
      - [3.4.2 - Training](#342---training)
    - [3.5 - 5th Model (EfficientNet)](#35---5th-model-efficientnet)
      - [3.5.1 - Preprocessing](#351---preprocessing)
      - [3.5.2 - Training](#352---training)
    - [3.6 - 6th Model (YOLOv11)](#36---6th-model-yolov11)
      - [3.6.1 - Preprocessing](#361---preprocessing)
      - [3.6.2 - Training](#362---training)
    - [3.7 - Model Flow Summary](#37---model-flow-summary)
  - [4. Models Results (Best)](#4-models-results-best)
    - [4.1 - 1st Model (Simple CNN)](#41---1st-model-simple-cnn)
    - [4.2 - 2nd Model (ResNet50\_v1)](#42---2nd-model-resnet50_v1)
    - [4.3 - 3rd Model (ResNet50\_v2)](#43---3rd-model-resnet50_v2)
      - [4.3.1 - New Codes to Predict S2E18 (Milestone 5)](#431---new-codes-to-predict-s2e18-milestone-5)
      - [4.3.2 - Initial Model (Milestone 4)](#432---initial-model-milestone-4)
      - [4.3.3 - Model Using Dataset After Duplicated Images Removal (Milestone 5)](#433---model-using-dataset-after-duplicated-images-removal-milestone-5)
    - [4.4 - 4th Model (VGG16)](#44---4th-model-vgg16)
      - [4.4.1 - Initial Model  (Milestone 4)](#441---initial-model--milestone-4)
      - [4.4.2 - Model using Dataset After Duplicated Images Removal (Milestone 5)](#442---model-using-dataset-after-duplicated-images-removal-milestone-5)
    - [4.5 - 5th Model (EfficientNet)](#45---5th-model-efficientnet)
      - [4.3.1 - Initial Model (Milestone 5)](#431---initial-model-milestone-5)
      - [4.3.2 - Model Using Dataset Afer Duplicated Images Removal (Milestone 5)](#432---model-using-dataset-afer-duplicated-images-removal-milestone-5)
    - [4.6 - 6th Model (YOLOv11)](#46---6th-model-yolov11)
      - [4.6.1 - F1, PR, P, R curves](#461---f1-pr-p-r-curves)
      - [4.6.2 - Confusion Matrix](#462---confusion-matrix)
      - [4.6.3 - Model Training Result](#463---model-training-result)
      - [4.6.4 - Model's Prediciton on Validation Set and unseen data](#464---models-prediciton-on-validation-set-and-unseen-data)
      - [4.6.5 - Model's Prediction on Video (S2E18 unseen data)](#465---models-prediction-on-video-s2e18-unseen-data)
  - [5. Discussion on Model's Methods \& Results](#5-discussion-on-models-methods--results)
    - [5.0 - Preprocessing Related](#50---preprocessing-related)
    - [5.1 - 1st Model (Simple CNN) to 2nd Model (ResNet)](#51---1st-model-simple-cnn-to-2nd-model-resnet)
      - [5.1.1 - Simple CNN](#511---simple-cnn)
      - [5.1.2 - ResNet50\_v1](#512---resnet50_v1)
    - [5.2 - 3rd Model (ResNet\_v2)](#52---3rd-model-resnet_v2)
      - [5.2.1 - ResNet\_v2 with Unmodified Dataset](#521---resnet_v2-with-unmodified-dataset)
      - [5.2.2 - ResNet\_v2 with Modified Dataset](#522---resnet_v2-with-modified-dataset)
    - [5.3 - 4th Model (VGG16)](#53---4th-model-vgg16)
      - [5.3.1 - VGG16 with Unmodified Dataset](#531---vgg16-with-unmodified-dataset)
      - [5.3.2 - VGG16 with Modified Dataset](#532---vgg16-with-modified-dataset)
    - [5.4 - 5th Model (EfficientNet)](#54---5th-model-efficientnet)
      - [5.4.1 - EfficientNet with Unmodified Dataset](#541---efficientnet-with-unmodified-dataset)
      - [5.4.2 - EfficientNet with Modified Dataset](#542---efficientnet-with-modified-dataset)
    - [5.5 - 6th Model (YOLOv11)](#55---6th-model-yolov11)
  - [6. Conclusion](#6-conclusion)
  - [7. Statement of Collaboration](#7-statement-of-collaboration)
- [Milestone 4: Second Model](#milestone-4-second-model)
  - [Second Model](#second-model)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Fitting Model](#fitting-model)
  - [Next Model](#next-model)
  - [Conclusion](#conclusion)
- [Milestone 3: Pre-Processing](#milestone-3-pre-processing)
  - [Pre-Processing](#pre-processing)
  - [First Model](#first-model)
    - [Training](#training-1)
    - [Evaluating](#evaluating)
    - [Fitting Model](#fitting-model-1)
  - [Conclusion](#conclusion-1)
- [Milestone 2: Data Exploration \& Initial Preprocessing](#milestone-2-data-exploration--initial-preprocessing)
  - [Data Exploration and Plot](#data-exploration-and-plot)
    - [Overview](#overview)
    - [Extraction method](#extraction-method)
    - [Number of Classes and Example Classes](#number-of-classes-and-example-classes)
    - [Size of Image](#size-of-image)
    - [Image Naming Convention](#image-naming-convention)
    - [Plot Our Data](#plot-our-data)
  - [Preprocessing](#preprocessing)
    - [Points to Consider](#points-to-consider)
    - [Initial Approach](#initial-approach)

</details>

## 1. Introduction
Our model aims to recognize whether Peter Griffin is on screen during any given moment of a Family Guy episode. We are looking to explore image classification problems and hope that we can learn them from solving this predictive task. Ultimately, we hope that our model will allow those affected by prosopagnosia to more easily enjoy television shows, especially when applied to live-action formats. Our model might also prove useful in helping people create compilations featuring a character of their choosing.


## 2. Data Prep
The basic preparation for our model includes the following steps:

### 2.1 - Extraction Method
To check which episode includes Peter Griffin, we used **[Family Guy Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/family-guy-dataset/data)** from Kaggle. This dataset includes various information about each Episode/Season of Family Guy. We performed data exploration on this dataset as the **[Family_Guy_Episode_Extract.ipynb](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/Data_Prep/Family_Guy_Episode_Extract.ipynb)**:
<details>
<summary>Click to collapse code</summary>

```python
import pandas as pd
import kagglehub
import os

path = kagglehub.dataset_download("iamsouravbanerjee/family-guy-dataset")
files = os.listdir(path)
print(files)

file_path = os.path.join(path, 'Family Guy Dataset.csv')
df = pd.read_csv(file_path)
filtered_episodes = df[df['Featuring'].str.contains(r'\bPeter\b', case=False, na=False)]
filered_Season = filtered_episodes[['Season', 'No. of Episode (Season)']]
season1_filtered_episodes = filered_Season[filered_Season['Season'] == 1]
season2_filtered_episodes = filered_Season[filered_Season['Season'] == 2]
season3_filtered_episodes = filered_Season[filered_Season['Season'] == 3]
season4_filtered_episodes = filered_Season[filered_Season['Season'] == 4]
season5_filtered_episodes = filered_Season[filered_Season['Season'] == 5]
```
**Example Result**:
```plaintext
| Season | No. of Episode (Season) |
|--------|-------------------------|
| 1      |           1             |
| 1      |           2             |
| 1      |           3             |
| 1      |           4             |
| 1      |           5             |
| 1      |           6             |
| 1      |           7             |
```
</details><br>

Then we used following two methods to extract frames from Family Guy episodes that include Peter Griffin:
1. **[Video Frame Extractor](https://frame-extractor.com/)**: We set distance between frames as 500ms so that we can get 2 frames per second. We experimeneted with few distances and found that 2fps result in fastest clear image extraction. 
2. **[Frame_Extractor.ipynb](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/Data_Prep/Frame_Extractor.ipynb)**: Some group members encountered error while using frame extractor above, so we used cv2 library to extract by ourselves as the below:
    <details>
    <summary>Click to collapse code</summary>
    
    ```python
    import cv2
    import os
    import zipfile

    def extract_frames(video_path, output_folder, frame_rate=1):
        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps / frame_rate)  # Capture every 'interval' frame

        frame_count = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save frames at the specified interval
            if frame_count % interval == 0:
                frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"Extracted {saved_count} frames to {output_folder}")

    video_path = 's5e4.mp4'
    output_folder = 'extracted_frames'
    extract_frames(video_path, output_folder, frame_rate=2)

    def zip_extracted_frames(folder_path, zip_filename="extracted_frames.zip"):
        # Create a Zip file
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            # Loop through all files in the folder
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))
        
        print(f"Frames from '{folder_path}' have been zipped into '{zip_filename}'")

    folder_path = 'extracted_frames'  # Folder with extracted frames
    zip_filename = 'extracted_frames.zip'
    zip_extracted_frames(folder_path, zip_filename)
    ```
    </details>

### 2.2 - Number of Classes and Example Classes
There are 2 classes in total:
1. Presence of Peter Griffin
   <br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/assets/with_peter_example.png" alt="With Peter" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/assets/with_peter_example2.png" alt="With Peter2" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/assets/with_peter_example3.png" alt="With Peter3" width="400"/>
        </tr>
   </table>
   
   Note: We classified the scenes where only small portion of Peter Griffin appears as presence of Peter Griffin. 
2. Absence of Peter Griffin
   <br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/assets/without_peter_example.jpg" alt="Without Peter" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/assets/without_peter_example2.jpg" alt="Without Peter2" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/assets/without_peter_example3.jpg" alt="Without Peter3" width="400"/>
        </tr>
   </table>

### 2.3 - Size and Number of Images
#### 2.3.1 - Size of Images
Depending on the extraction methods that each group member used, the image size varies.
1. 1440x1080 png (from Video Frame Extractor)
2. 320x240 jpg (from Frame_Extractor.ipynb*)

#### 2.3.2 - Number of Images
1. Number of images in each episode (With_Peter vs Without_Peter)
   <br><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/Data_Prep/Dataset_number_of_Images_per_Episode.png" alt="Number of images in each episode" width="800"/>
2. Total number of images across all episodes
   <br><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/Data_Prep/Dataset_total_number_of_Images.png" alt="Number of images total" width="300"/>
3. Total number of images across all episodes after removing duplicated images (Milestone 5)
   <br><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/Data_Prep/Dataset_non_Duplicated_total_number_of_Images.png" alt="Number of images total" width="300"/>





## 3. Models Methods
### 3.1 - 1st Model (Simple CNN)
#### 3.1.1 - Preprocessing
We restricted our dataset to images from a singular episode (S5E04) with downscaling to 320x240 from 1440x1080 for faster downloading and training speed. We splitted training and validation set in 9:1 ratio. 
#### 3.1.2 - Training 
We applied Geek-for-Geek's **[cat-vs-dog model](https://www.geeksforgeeks.org/cat-dog-classification-using-convolutional-neural-network-in-python/)** and fit it for our image data. Running **[1st model](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/1st_Model/1st_model_CNN.ipynb)** yielded promising signs, yet was clearly not in tune for the specific problem we are tackling:
```python
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(320, 240, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])
```

More details for our first model can be found in [Milestone 3](#milestone-3-pre-processing) section

### 3.2 - 2nd Model (ResNet50_v1)
#### 3.2.1 - Preprocessing
Unlike our 1st model, we decided to use all images (S1E05, S2E01, S3E10, S5E04, S5E05, S5E16) by combinining all image folders into one folder in our google drive. We also splitted training and validation set in 9:1 ratio. 
#### 3.2.2 - Training
We used pretrained model (ResNet50) as our base model and left other model setup same as our 1st model. **[Second Model](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/2nd_Model/2nd_model_ResNet_v1.ipynb)** had model layers as the following:
```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(320, 240, 3))
model = tf.keras.models.Sequential([
    base_model,
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])
```
More details for our Second model can be found in [Milestone 4](#milestone-4-second-model) section

### 3.3 - 3rd Model (ResNet50_v2)
#### 3.3.1 - Preprocessing
With the same setup as our 2nd model, we added one more preprocessing step using **["preprocess_input"](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/preprocess_input)** function:

```python
from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess_dataset(image, label):
    # Apply ResNet50 preprocessing
    image = preprocess_input(image)
    return image, label

# Preprocess datasets using the custom function
train_datagen = train_datagen.map(preprocess_dataset)
test_datagen = test_datagen.map(preprocess_dataset)
```
Later, during Milestone 5, we removed duplicated images (frames that have similar background and character setups) from our dataset. These, duplicated images are created since we set distance between frames as 500ms so that we can get 2 frames per second during extraction process mentioned at [1.1 - Extraction Method](#11---extraction-method)

#### 3.3.2 - Training 
We came up with simpler model with ResNet50 base model from our 3rd model. In addition, instead of using MaxPooling, we chose to use GlobalAveragePooling2D. **[Thrid Model](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/3rd_Model/3rd_model_ResNet_v2.ipynb)** had model layers as the following:
```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(320, 240, 3))
base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])
```
The Third Model that used dataset where duplicated images are removed can be found **[here](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/3rd_Model/3rd_model_ResNet_v2(dataset_fixed).ipynb)**; this version of model was created during Milestone 5.

More details for our Third model can be found in [Milestone 4](#milestone-4-second-model) section

### 3.4 - 4th Model (VGG16)
#### 3.4.1 - Preprocessing
Instead of using resnet50's preprocess_input function, we adopted **["preprocess_input"](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/preprocess_input)** function from vgg16:
```python
from tensorflow.keras.applications.vgg16 import preprocess_input

def preprocess_dataset(image, label):
    # Apply ResNet50 preprocessing
    image = preprocess_input(image)
    return image, label

# Preprocess datasets using the custom function
train_datagen = train_datagen.map(preprocess_dataset)
test_datagen = test_datagen.map(preprocess_dataset)
```
Similar to ResNet_v2, later during Milestone 5, we removed duplicated images from our dataset. 

#### 3.4.2 - Training
With the same setup as our ResNet model, we changed the base model to VGG16 from ResNet50. **[Fourth Model](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/4th_Model/4th_model_VGG16.ipynb)** had model layers as the following:
```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(320, 240, 3))
base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
```
The Fourth Model that used dataset where duplicated images are removed can be found **[here](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/4th_Model/4th_model_VGG16%20(dataset_fixed).ipynb)**; this version of model was created during Milestone 5.

More details for our Fourth model can be found in [Milestone 4](#milestone-4-second-model) section

### 3.5 - 5th Model (EfficientNet)
This model was created during Milestone 5. 
#### 3.5.1 - Preprocessing
First, we tested with the dataset we used for ResNet50_v1. Then, we removed duplicated images (frames that have similar background and character setups) from our dataset like ResNet_v2 and VGG16.

In addition, we utilized **["preprocess_input](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/preprocess_input)** function from resnet:
```python
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_dataset(image, label):
    image = preprocess_input(image)
    return image, label

# Preprocess datasets using the custom function
train_datagen = train_datagen.map(preprocess_dataset)
test_datagen = test_datagen.map(preprocess_dataset)
```
#### 3.5.2 - Training
With the same setup as our ResNet model, we changed the base model to EfficientNet. **[Fifth Model](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/5th_Model/5th_model_efficientNet.ipynb)** had model layers as the following:
```python
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(320, 240, 3))
base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])
```
The Fourth Model that used dataset where duplicated images are removed can be found **[here](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/5th_Model/5th_model_efficientNet(dataset_fixed).ipynb)**; this version of model was also created during Milestone 5.

### 3.6 - 6th Model (YOLOv11)
This model was created during Milestone 5 for preliminary model for future usage. 
#### 3.6.1 - Preprocessing
We added an labeling annotation for Peter using **[CVAT](https://www.cvat.ai/)** to the images from the dataset where we removed the duplicated images. Below two images are showing how we labeled/annotated Peter using CVAT:
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model/CVAT_example1.png" alt="CVAT Example 1" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model/CVAT_example2.png" alt="CVAT Example 2" width="400"/>
        </tr>
   </table>

In addition, to run the YOLOv11 model, we setup our working directory as the following:

```
Working Directory/
├── train/
│   ├── images     # used ~80% of images 
│   ├── labels     # used ~80% of labels 
├── val/
│   ├── images     # used ~20% of images 
│   ├── labels     # used ~20% of labels 
```

We had total number of images and labels for Train and Val as the following:
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model/6th_model_YOLOv11_total_number_of_images.png" alt="Total Number of Images for Train and Val" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model/6th_model_YOLOv11_total_number_of_labels.png" alt="Total Number of Labels for Train and Val" width="400"/>
        </tr>
   </table>

Lastly, we created **[config.yaml](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model/config.yaml)** for YOLOv11 model training:
```yaml
train: /train
val: /val

nc: 1 # number of classes

names: ["Peter"] # name of class
```

#### 3.6.2 - Training 
Since this YOLOv11 model was created for preliminary purpose, we didn't do hyperparameter tuning or experimenting with different YOLO variants. The training is done through **[6th_model_YOLOv11.ipynb](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model/6th_model_YOLOv11.ipynb)**:
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt") # yolo v11 nano

train_results = model.train(
    data="config.yaml",  
    epochs=10,  
)
```


### 3.7 - Model Flow Summary
| Model Name | Dataset | Milestone | Misc. |
|:-------:|:---------:|:---------:|:---------:|
|  [Simple CNN](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/1st_Model/1st_model_CNN.ipynb)    | Duplicated O   | Milestone 3   | No fine tuning                  |
|  [ResNet_v1](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/2nd_Model/2nd_model_ResNet_v1.ipynb)     | Duplicated O   | Milestone 4   | Different lr, optimization      |
|  [ResNet_v2](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/3rd_Model/3rd_model_ResNet_v2.ipynb)     | Duplicated O   | Milestone 4   | Different lr, optimization      |
|  [VGG16](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/4th_Model/4th_model_VGG16.ipynb)         | Duplicated O   | Milestone 4   | Different lr, optimization      |
|  [ResNet_v2](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/3rd_Model/3rd_model_ResNet_v2(dataset_fixed).ipynb)     | Duplicated X   | Milestone 5   | Used best ResNet_v2 version     |
|  [VGG16](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/4th_Model/4th_model_VGG16%20(dataset_fixed).ipynb)         | Duplicated X   | Milestone 5   | Used best VGG16 version         |
|  [EfficientNet](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/5th_Model/5th_model_efficientNet.ipynb)  | Duplicated O   | Milestone 5   | Different lr, optimization      |
|  [EfficientNet](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/5th_Model/5th_model_efficientNet(dataset_fixed).ipynb)  | Duplicated X   | Milestone 5   | Used best EfficientNet version  |
|  [YOLOv11](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model/6th_model_YOLOv11.ipynb)       | Smaller Duplicated X   | Milestone 5   | No fine tuning, used smaller dataset. This is NOT our final Model since this was for exploring/experiencing new Model which is popular among ML/DL programmer rather than using it for our improved version of model from previous Milstones; more details on last model will be covered in [Discussion](#5-discussion-on-models-methods--results) |

## 4. Models Results (Best)
### 4.1 - 1st Model (Simple CNN)
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/1st_Model/1st_model_CNN_accuracy_plot.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/1st_Model/1st_model_CNN_loss_plot.png" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>
Our first model (CNN Model) achieved 99.59% of training accuracy and 85.87% of validation accuracy. 

### 4.2 - 2nd Model (ResNet50_v1)
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/2nd_Model/2nd_model_ResNet_v1_accuracy_plot.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/2nd_Model/2nd_model_ResNet_v1_loss_plot.png" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>
Our ResNet50_v1 Model achieved 99.53% of training accuracy and 97.26% of validation accuracy. 


### 4.3 - 3rd Model (ResNet50_v2)
#### 4.3.1 - New Codes to Predict S2E18 (Milestone 5)
During Milestone 5, we tried to check the model performance on unseen data (not validation datset) from **S2E18**. We first set two folders: "frames_resized" and "frames_classified".
1. "frames_resized" includes all the frames from S2E18 without classifying it into with_peter and without_peter.
2. "frames_classified" includes two folders inside: "With_peter" and "Without_peter" which are classfied manually. 

Then, using the code below, we calculated classified list which includes 1s for Peter and 0s for Non-Peter from our model prediciton on S2E18:
```python
directory = "frames_resized"

files = [f for f in os.listdir(directory) if f.startswith("out-") and f.endswith(".jpg")]

sorted_files = sorted(files, key=lambda x: int(x.split('-')[1].split('.')[0]))

from tensorflow.keras.preprocessing.image import load_img, img_to_array

classified = []

index = 0
for file in sorted_files:
    file_path = os.path.join(directory, file)
    test_image = load_img(file_path,target_size=(320,240))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    result = model.predict(test_image,verbose=0)
    if(result<0.5):
        peter = 1
    else:
        peter = 0
    classified.append(peter)
    index += 1
```
After that, using the following code, we calculated actual classified list using "With_Peter" images from "frames_classified" to compare with above classified list from our model:
```python
directory = "frames_classified/With_Peter"

files = [f for f in os.listdir(directory) if f.startswith("out-") and f.endswith(".jpg")]

classified_actual = [0 for _ in range(len(classified))] 

for file in files:
    integer_part = int(file.split('-')[1].split('.')[0])
    integer_part = int(integer_part/10 -1)
    classified_actual[integer_part] = 1

print(classified_actual)
```
Finally, we calcualted how many images from S2E18 that the model correctly predicted as the following:
```python
T = np.sum(np.array(classified) == np.array(classified_actual))
F = np.sum(np.array(classified) != np.array(classified_actual))
accuracy = T / (T+F)
print(T+F)
print(accuracy)

```

In addition to observing accuracy of model's prediction on unseen data using classified lists, we wrote java code - **[App.java](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/App.java)** - to create a bash script that can be used to generate the video. We drew text using [java.io.PrintWriter](https://docs.oracle.com/javase/8/docs/api/java/io/PrintWriter.html), which indicates the presence of Peter according to the Model's result on live video with the bash script created.

 **[App_csv.java](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/App_csv.java)** is diferent version of code which uses csv file of classified list instead of manually typing them inside the code. Example screenshots of video generated by bash script from **App.java** are at below:

<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/assets/App_java_example_1.png" alt="App.java Example 1" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/assets/App_java_example_2.png" alt="App.java Example 2" width="400"/>
        </tr>
   </table>



#### 4.3.2 - Initial Model (Milestone 4)
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/3rd_Model/3rd_model_ResNet_v2_accuracy_plot.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/3rd_Model/3rd_model_ResNet_v2_loss_plot.png" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>
Our Initial ResNet50_v2 model achieved 95.59% of training accuracy and 95.85% of validation accuracy. 

In addition, using the code snippet below:

<details>
<summary>Click here to view code snippet</summary>

```python
tp = 0 
fp = 0 
tn = 0 
fn = 0  

for images, labels in test_datagen:
    preds = model.predict(images)
    # threshold=0.5
    binary_preds = (preds > 0.5).astype(int).flatten()
    labels = labels.numpy().flatten()
    
    for pred, true_label in zip(binary_preds, labels):
        if pred == 1 and true_label == 1:
            tp += 1  # Correctly predicted 'With Peter'
        elif pred == 1 and true_label == 0:
            fp += 1  # Incorrectly predicted 'With Peter'
        elif pred == 0 and true_label == 0:
            tn += 1  # Correctly predicted 'Without Peter'
        elif pred == 0 and true_label == 1:
            fn += 1  # Incorrectly predicted 'Without Peter'

# Print results
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")
```
</details>
<br>
We got our TP, TN, FP, FN values for Initial ResNet50_v2 model as the following:

```
True Positives (TP): 671
False Positives (FP): 40
True Negatives (TN): 552
False Negatives (FN): 13
```

Lastly, we got 75.30% of accuracy on predicting S2E18 unseen episode. 

#### 4.3.3 - Model Using Dataset After Duplicated Images Removal (Milestone 5)
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/3rd_Model/3rd_model_ResNet_v2(dataset_fixed)_accuracy_plot.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/3rd_Model/3rd_model_ResNet_v2(dataset_fixed)_loss_plot.png" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>
Using the same model but with the dataset that doesn't include duplicated images, our ResNet50_v2 model achieved 91.38% of training accuracy and 81.11% of validation accuracy.

In addition, we got our TP, TN, FP, FN values as the following:
```
True Positives (TP): 93
False Positives (FP): 17
True Negatives (TN): 53
False Negatives (FN): 17
```

Lastly, we got 74.05% of accuracy on predicting S2E18 unseen episode. 


### 4.4 - 4th Model (VGG16)
#### 4.4.1 - Initial Model  (Milestone 4)
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/4th_Model/4th_model_VGG16_accuracy_plot.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/4th_Model/4th_model_VGG16_loss_plot.png" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>
Our Initial VGG16 model achieved 90.73% of training accuracy and 91.61% of validation accuracy.

#### 4.4.2 - Model using Dataset After Duplicated Images Removal (Milestone 5)
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/4th_Model/4th_model_VGG16%20(dataset_fixed)_accuracy_plot.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/4th_Model/4th_model_VGG16%20(dataset_fixed)_loss_plot.png" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>
Using the same model but with the dataset that doesn't include duplicated images, our VGG16 model achieved 88.67% of training accuracy and 82.22% of validation accuracy.

In addition, we got our TP, TN, FP, FN values as the following:
```
True Positives (TP): 94
False Positives (FP): 16
True Negatives (TN): 54
False Negatives (FN): 16
```

Lastly, we got 78.22% of accuracy on predicting S2E18 unseen episode. 

### 4.5 - 5th Model (EfficientNet)
#### 4.3.1 - Initial Model (Milestone 5)
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/5th_Model/5th_model_efficientNet_accuracy_plot.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/5th_Model/5th_model_efficientNet_loss_plot.png" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>
Our Initial EfficientNet model achieved 95% of training accuracy and 95.22% of validation accuracy. 

In addition, we got our TP, TN, FP, FN values as the following:
```
True Positives (TP): 651
False Positives (FP): 28
True Negatives (TN): 564
False Negatives (FN): 33
```

Lastly, we got 78.40% of accuracy on predicting S2E18 unseen episode. 


#### 4.3.2 - Model Using Dataset Afer Duplicated Images Removal (Milestone 5)
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/5th_Model/5th_model_efficientNet(dataset_fixed)_accuracy_plot.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/5th_Model/5th_model_efficientNet(dataset_fixed)_loss_plot.png" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>
Using the same model but with the dataset that doesn't include duplicated images, our EfficientNet model achieved 92.76% of training accuracy and 84.44% of validation accuracy. 

In addition, we got our TP, TN, FP, FN values as the following:
```
True Positives (TP): 102
False Positives (FP): 20
True Negatives (TN): 50
False Negatives (FN): 8
```
Lastly, we got 86.23% of accuracy on predicting S2E18 unseen episode. 

### 4.6 - 6th Model (YOLOv11)
#### 4.6.1 - F1, PR, P, R curves
After running YOLOv11 model, it created F1, PR, P, and R curve for the model as the following:

<br>
    <table>
    <tr>
        <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model_Plots/F1_curve.png" alt="F1 curve" width="400"></td>
        <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model_Plots/PR_curve.png" alt="PR curve" width="400"></td>
    </tr>
    <tr>
        <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model_Plots/P_curve.png" alt="P curve" width="400"></td>
        <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model_Plots/R_curve.png" alt="R curve" width="400"></td>
    </tr>
    </table>

#### 4.6.2 - Confusion Matrix
We got confusion matrix (normalized & unnormalized) as the following:
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model_Plots/confusion_matrix.png" alt="Confusion Matrix" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model_Plots/confusion_matrix_normalized.png" alt="Normalized Confusion Matrix" width="400"/>
        </tr>
   </table>

#### 4.6.3 - Model Training Result
The model's loss, precision, recall, and other metrics are in **[results.png](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model/results.png)**:
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model_Plots/results.png" alt="Results" width="600"/>
        </tr>
   </table>

#### 4.6.4 - Model's Prediciton on Validation Set and unseen data
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model/val_batch2_labels.jpg" alt="val_batch2_labels" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model/val_batch2_pred.jpg" alt="val_batch2_pred" width="400"/>
        </tr>
   </table>
Left plot is displaying the label for few validation images. Right plot is displaying the prediction on each label for those validation images. 
<br>

In addition, using **["6th_model_YOLOv11_4prediction.ipynb"](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model/6th_model_YOLOv11_4prediction.ipynb)**, we predicted unseen images with the YOLOv11 model. Below is one of the prediction on unseen images:
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model/6th_model_YOLOv11_prediction_example.jpg" alt="prediction example" width="400"/>
        </tr>
   </table>
It is considering Peter in images as Peter with 0.96 probability(confidence).

#### 4.6.5 - Model's Prediction on Video (S2E18 unseen data)
Lastly, using **["6th_model_YOLOv11_video_prediction.ipynb"](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/6th_Model/6th_model_YOLOv11_video_prediction.ipynb)**, we drew green box with Peter label to the frame where model predicts presence of Peter by higher than 0.5 threshold:
```python
...code snippets from  "6th_model_YOLOv11_video_prediction.ipynb"...

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()
```

Then, we merged all the frames to convert to video back.

Unlike, the video that we created using **"App.java"** for ResNet model, the model predicts Peter more accurately with green box moving along as Peter moves. The short video (full video is 22mins 27secs) can be found in [here](https://drive.google.com/file/d/1AUsivR5dPQ3cNXlRmMZGwXlVv_yBfJCS/view?usp=drive_link)(google drive) 

> 🕒 **"Running '6th_model_YOLOv11_video_prediction.ipynb' took 108 minutes.**  
> But, great things take time — Rome wasn't YOLO-ed in a day!" 😄
> <br> - ChatGPT - 


## 5. Discussion on Model's Methods & Results
### 5.0 - Preprocessing Related
Before starting to build our models, we needed to manually create our own datasets. To do this, we scraped one frame every 500 ms between 6 episodes (S1E05, S2E01, S3E10, S5E04, S5E05, S5E16), then decided if that frame contained Peter Griffin or not. This resulted in a dataset of 13939 images! To help the model run more efficiently, we decided to downscale all the images from a crisp 1444 × 1080 to a grainy 320 x 240. Despite this, we believe that the model’s ability to classify the images would not be hindered much if at all. Due to the nature of being an animated sitcom, every frame tended to be quite basic. Such, compressing the images does not lose much data.

However, while working on a later milestone, we were frustrated by the disconnect between the validation accuracy and the accuracy on the unseen data. We figured that the validation accuracy was so high due to the characteristics of our train and validation datasets. Because of how we originally acquired the data, we had many frames that were identical or near-identical to each other. This unfairly benefits the validation dataset accuracy and would not be indicative of how the model would perform on unseen data. In other words, our train and validation datasets had a lot of unintentional overlap. To remedy this issue, we went back and removed these identical/near-identical frames from our dataset, leaving us with 1806 images, just 13% of the original dataset! The resulting models took a hit on validation accuracy, but had noticeably more similar accuracy between the validation and unseen data, hopefully indicating that our model was doing a better job at generalizing. A useful side effect of the large decrease of dataset size was the much-improved speed of the model.

### 5.1 - 1st Model (Simple CNN) to 2nd Model (ResNet)
#### 5.1.1 - Simple CNN
As with most image classification models, we decided to use binary cross-entropy as our loss function. We also decided to use accuracy as a metric, since true positives, false positives, true negatives, and false negatives are all very important to our goals.

Our first model (CNN Model) achieved 99.59% of training accuracy and 85.87% of validation accuracy. Then, after manually testing several frames from this episode if it is correctly classified, we earned some idea about where the model may be inaccurate. 

Looking at plots of validation metrics in [4.1 Section](#41---1st-model-simple-cnn), we confirmed that our model fits in the overfitting region of the fitting graph. The model achieved relatively low training loss (0.0155) and high training accuracy (0.9959). On the other hand, the model achieved relatively high training loss (0.5099) and low validation accuracy (0.8587). In addition, both validation loss and accuracy fluctuated significantly; validation loss spiked twice up to 7.8551 and 2.8553 before reaching 0.5099 at the final epoch. 

Therefore, our conclusion was that our model is too complex for our dataset, which led poor generalization to the validation set and overfitting issue (even though worked well on the training data). However, it showed some promise. The model, with its validation accuracy of ~86%, performed much better than random guessing, suggesting that CNNs were up to the task. 

#### 5.1.2 - ResNet50_v1
To fix this issue, we tried to improve our model by changing base model from simple CNN to ResNet50 pretrained model. We hoped to utilize the advantage of pretrained model as they provide a set of initial weights and biases that can be fine-tuned for a specific task in our own way.

In the end, we decided on using ResNet50, a model in the ResNet family that was widely used for image classification, as the base of our second model. Rather than using ResNet with other depths (e.g. ResNet18, ResNet101, ResNet152) we chose ResNet50 specifically since it is the most intermediate one so that we can avoid risk of having too small or too large models. Plots in [4.2 Section](#42---2nd-model-resnet50_v1) demonstrates that our ResNet model achieved 99.53% of training accuracy and 97.26% of validation accuracy. 

Comparing to the simple 1st model (Simple CNN), we were able to confirm that our new model improved in large scale. First of all, the validation accuracy increased from 0.8587 to 0.9726. In addition, validation accuracy remains more stable and less fluctuated comparing to our previous model which had intense fluctuation, especially at epoch 3 and 7. However, this model hasn't solved the overfitting and generalization issue perfectly yet. In addition, the model size was pretty big (21,500,929 trainable parameters), so we decided to decrease the model size by alternating the layer structure since an overcomplicated structure may result in overfitting later.


### 5.2 - 3rd Model (ResNet_v2)
#### 5.2.1 - ResNet_v2 with Unmodified Dataset
In ResNet50_v2, we decided to decrease the number of additional layers to improve generalizability. Additionally, by switching from using MaxPooling to GlobalAveragePooling2D, we hoped to reduce the tendency of overfitting and to help insulate our model from outliers.

Training and testing ResNet50_v2 showed that this model retained the effectiveness of ResNet50_v1, with training and validation accuracies at 95.59% and 95.85% respectively. Despite this, it performed much more poorly on the unseen episode, with an accuracy of 75.3%.To test the accuracy of the unseen data, we manually classified every tenth frame of the episode S2E18 for a grand total of 6540 images. Then, we have the model classify the images, and compare the two results. It is at this point where we realized that having identical or nearly-identical images in the datasets can result in unintended “cheating” of our model. Since we just randomly assigned training and validation dataset, it’s possible that both dataset have very similar images, which will result in higher accuracy than what the actual model’s capacity can result in.

#### 5.2.2 - ResNet_v2 with Modified Dataset
However, removing duplicated images did not directly increase the model’s performance. As expected, the model accuracy decreased to 91.38% and 81.11% after removal. However, it was a good sign that the accuracy on the unseen data remained nearly the same, at 74.1%.

### 5.3 - 4th Model (VGG16)
#### 5.3.1 - VGG16 with Unmodified Dataset
We also decided to test the effectiveness of VGG16 as our base. VGG16, like ResNet50, is a pretrained model that is commonly used for image classification. A key difference between the two is the number of layers; VGG16 has 16 layers while ResNet50 has 50 layers. We wanted to see if the less complex model that is VGG16 would help our model generalize better. The resulting accuracy of our fourth model was 90.73% for training and 91.61% for validation. It is important to note that the validation accuracy was higher than the training accuracy. This may have two explanations. The first is underfitting, meaning that the model isn’t trained enough. We found this to be unlikely since we trained it for 10 epochs. The second is unlucky train and test splits. Since we did not remove the duplicated images in the original dataset, if the validation dataset contained many duplicates of images in the training dataset, it becomes possible for the validation accuracy to be higher than the training accuracy.

#### 5.3.2 - VGG16 with Modified Dataset
As we saw in our third model, the accuracy of the model decreased after removing the duplicated images, with a result of 88.67% and 82.22%. The unseen accuracy rose to 78.2%, which suggests that VGG16 would be better than the ResNet models moving forward.

### 5.4 - 5th Model (EfficientNet)
#### 5.4.1 - EfficientNet with Unmodified Dataset
This time, we decided to utilize the newest pretrained model so far: EfficientNet. As the name suggests, EfficientNet promised to increase efficiency while maintaining accuracy. As our base model, EfficientNet delivered. The accuracy of our fifth model dramatically improved compared to our previous models, sitting at 95% for training and validation. At 78.4%, this model had the highest accuracy of any model before making changes to the dataset.

#### 5.4.2 - EfficientNet with Modified Dataset
Despite the dataset change, our EfficientNet-based model still maintained pretty high values (92.76% & 84.44%). Most importantly, among the models trained on the new dataset, this model resulted in the highest accuracy for the unseen episode by far, at a scorching 86.2%. This is almost the same as the validation accuracy, which means that the dataset is well organized after removal. After compiling and watching the video with these predictions, it passes the eye test. The noticeable mistakes were few, with one being a scene in the dark where Peter’s face is only mostly visible.

### 5.5 - 6th Model (YOLOv11)
First of all, even though we explored YOLOv11 model at the end of this project, this doesn't mean that we are trying to settle down at YOLOv11 as our final model. Since we employed YOLOv11 with minimal configuration, relying on default parameters and pretrained weights without extensive hyperparameter tuning or architectural adjustments. This "out-of-the-box" approach provided quick insights but did not fully leverage YOLOv11's potential for optimization and performance enhancement. We are considering this model to be "preliminary trial" and to be our next step to further explore after this class ends; experience with YOLO model was really valuable and fun though. We still wanted to analyze the results from YOLOv11 model and discuss about it (to more review our learnings from this class and apply them to this project).

YOLO (You Only Look Once) models are specifically designed for real-time object detection. This can process frames with high detection accuracy quickly. In addition, it is easy relatively easy to deploy and has well-documented [instructions](https://docs.ultralytics.com/ko/models/yolo11/) on how to use this model with active community. Lastly, YOLO is still evolving and has gained popularity in industry since it has been released. Thus, we chose to explore on YOLOv11 before this class finishes not only for testing out better model performance, but also for future work. 

Looking at plots in [4.6.1 Section](#461---f1-pr-p-r-curves), we can get an insight of which threshold(confidence) we need to use for different purposes:
1. **F1-Confidence Curve**: F1 score peaked at approximately 0.87 when the confidence threshold is around 0.289. At lower threshold greater than approximately 0.1 less than 0.289 remained at similar F1 score as the peak. However, at higher threshold greater than 0.289, drops exponentially. Thus, we can conclude that, if we need situation where equal emphasis on precision and recall is required, we need to use threshold around 0.2~0.3.
2. **Precision-Confidence Curve**: Precision reached 1.0 maximum value at a confidence threshold of 0.93. Precision increased as the confidence threshold increased. This is desired shape of the curve as higher thresholds make the model more selective, reducing the number of false positives. For the situation where false positives are costly, we need to look for higher confidnence threshold to keep high precision.
3. **Recall-Confidence Curve**: Recall decreased as the confidence threshold increases; it is also an expected resut as described above. At 0 threshold (very low), recall approaches 1.0. Under the situation where missing true positives is costly, we need to look for lower confidence threshold (closer to 0 as possible). 
4. **Precision-Recall Curve**: Precision was close to 1 for lower recall values, indicating that the model started with very few false positives. Then, as recall increases, precision slightly decreases due to getting more false positives. The curve shaped round to top-right corner with large inner area, meaning that our mean average precision was high as desired. The mean average precision (mAP) was 0.853, which proves strong performance. 

Looking at plots in [4.6.2 Section](#462---confusion-matrix), we can get an insight of how model performed on different types of images. Confusion matrix in this plot demonstrates that model correctly identifies 90% of the frames containing Peter; in other words, 10% of the frames containing Peter are misclassified as background. The model never misclassified background frames as containing Peter. Again, this plot is suggesting that model performed quite well, but still we need to improve recall.

The plots in [4.6.3 Section](#463---model-training-result) wraps up and prove the analysis to be valid by showing the losses, mAP50, mAP50-95, and other metrics per epoch during training.

Overall, the model performed well on detecting/classifying Peter in each frame from Family Guy episodes. Yet, we can improve more by fine-tuning hyperparameters, adopting better thresholds according to this result, making better generalization, etc. We are looking forward to continue this project focusing on YOLOv11 model even after this course to explore 💻 ML/DL Worlds!

Again, the video including green boxes for peter detecton generated by bash script can be found in [here](https://drive.google.com/file/d/1AUsivR5dPQ3cNXlRmMZGwXlVv_yBfJCS/view?usp=sharing) - 20 seconds clips from full video. Full video can be found in [here](https://drive.google.com/file/d/1g4039k4WD_WzaZ3LCVzKiOtMuVWTuE1Q/view?usp=sharing).

## 6. Conclusion
We were quite happy with the results of our EfficientNet-based model. The high accuracy on the unseen data was very encouraging. However, there would be more work to be done before it could be a viable deliverable. We noticed from watching the classified video that there are three main mistakes that the model can make. The first is long scenes where the model is outright wrong. Here, the model seems to believe that Bill Clinton is Peter Griffin (?), perhaps it is thrown off by Bill Clinton’s size and butt-chin.


https://github.com/user-attachments/assets/11a220b3-340b-412a-92f8-82280ea2f443


Issues like these are very tough to fix easily.

The second common mistake is when the model constantly switches back and forth on whether Peter is there or not, when Peter is definitely present. Here is an example of that happening.


https://github.com/user-attachments/assets/07b30a8d-b27f-45fd-a32e-71670e044cce


This suggests that the model is right on the verge in predicting either way (0.5… vs 0.4…) during the scene. Perhaps by further tuning our model, we could improve our model during such scenes with some difficulty.

The last common issue is flickers to wrong classifications. These are the easiest issues to correct. Simply by smoothing the output, these issues are easily eliminated.


https://github.com/user-attachments/assets/90ac94de-4927-48a3-930d-2b063f3968f0


As for our Model 6, it seems promising. If we can improve upon it, it can have many uses, such as integrating into video-editing tools for easy Peter access or into a real-time application to detect Peter.

We can further expand model 6 to detect other characters in Family Guy; or apply our model to  other animation. In that way, our ultimate goal to allow those affected by prosopagnosia to better enjoy television shows, especially when applied to live-action formats, will be another step closer!


## 7. Statement of Collaboration
**🌟 Team Contributions: The Magnificent Six 🌟**
👥 Contributors
Jacob Zhang jaz007@ucsd.edu
Jaewon Han jah034@ucsd.edu
Patric Sanchez p3sanchez@ucsd.edu
Yoongon Kim ygk001@ucsd.edu
Ruhua Pan oriiii932@gmail.com
Hou Wan hwan@ucsd.edu

<a href="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/graphs/contributors">
<img width="500" src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone5/milestone5/assets/contributor_image.png">
</a>

# Milestone 4: Second Model
## Second Model
### Training
Unlike our first model from Milestone 3, we decided to use all images (S1E05, S2E01, S3E10, S5E04, S5E05, S5E16) by combinining all image folders into one folder in our google drive. In addition, instead of using simple CNN, we came up with 2 different models that use different types of pretrained model: ResNet50 and VGG16; later on, we're trying to also compare them with deeper CNN model which contains deeper layers in it. 

*Note: Since we had to test different models with different parameters, we have two types of codes: a code for Google Colab which downloads dataset from google drive and a code for local in which we manually downloaded dataset from google drive.  

1. [ResNet50](#)
    ```python
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(320, 240, 3))
    model = tf.keras.models.Sequential([
        base_model,
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.1),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])
    ```
    Instead of using simple CNN structure, we used ResNet50 as our base model. In addition, after observing our loss/accuracy graph to be fluctuated, we tried more simpler model which still uses ResNet50 as the following:
    ```python
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(320, 240, 3))
    model = tf.keras.models.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    ``` 
    We also experimented with different types of pooling: GlobalAveragePooling2D and MaxPooling2D/ 
2. [VGG16](#)
    ```python
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(320, 240, 3))
    model = tf.keras.models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    ```
    Instead of using simple CNN, we used VGG16 as our base model. Similar to ResNet50, we tried with different parameters and layer setups, but the above code was our base code for different variations of VGG16 model. 

Lastly, after observing some overfitting patterns, we experimented our model with learning rate scheduler as the following:
```python
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3, # adjusted
    decay_steps=10000, # adjusted
    decay_rate=0.9 # adjusted
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```
### Evaluation
1. **[ResNet v1](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/ResNet_v1.ipynb)**
   <br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/resnet_v1_accuracy.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/resnet_v1_loss.png" alt="Training Loss and Validation Loss" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>
   Comparing to the simple CNN model from Milestone 3, we can check that our new model improved in large scale. First of all, the validation accuracy increased from 0.8587 to 0.9726. In addition, validation accuracy remians more stable and less fluctuated comparing to our previous model which had intense fluctuation, especially at epoch 3 and 7; validation accuracy dropped exponentially. 
2. **[ResNet v2](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/ResNet_v2.ipynb)**
   <br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/resnet_v2_accuracy.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/resnet_v2_loss.png" alt="Training Loss and Validation Loss" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>
   Second version of Resnet showed lower accuracy and higher loss than first version (even though the differences were really small). However, there were huge improvements as we finally overcame (not 100% though) the fluctuation issue and overfitting problem. Our validation accuracy and loss aligns almost same with training ones; At the end, we are still experiencing with slight overfitting pattern, but we think that we can reduce that sign of overfitting, for example by making epoch to be not processed by lr scheduler.
3. **[VGG16 v1](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/VGG_v1.ipynb)**
   <br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/vgg_accuracy.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/vgg_loss.png" alt="Training Loss and Validation Loss" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>
   VGG model also resolves some fluctuating graphs and overfitting issue, but had significantly low accuracy and loss comparing to ResNets. In addition, it took way longer to train the VGG than ResNet or CNN. 

Lastly, similar to Milestone2 model, two ResNet models listed above correctly classified the unseen data (5 images); VGG classified one unseen data (out of 5) wrong. 
The unseen images in question are from S2E18:
   <br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/out-10000.jpg" alt="Image 1" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/out-20000.jpg" alt="Image 2" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/out-30000.jpg" alt="Image 3" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/out-40000.jpg" alt="Image 4" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/out-50000.jpg" alt="Image 5" width="400"/>
        </tr>
   </table>
For our v2 model, we calculated our TP, TN, FP, FN values.
   <br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/image.PNG" alt="TP, TN, FP, FN" width="400"/>
        </tr>
   </table>

```
tp = 0 
fp = 0 
tn = 0 
fn = 0

for images, labels in test_datagen:
    preds = model.predict(images)
    # threshold=0.5
    binary_preds = (preds > 0.5).astype(int).flatten()
    labels = labels.numpy().flatten()

    for pred, true_label in zip(binary_preds, labels):
        if pred == 1 and true_label == 1:
            tp += 1  # Correctly predicted 'With Peter'
        elif pred == 1 and true_label == 0:
            fp += 1  # Incorrectly predicted 'With Peter'
        elif pred == 0 and true_label == 0:
            tn += 1  # Correctly predicted 'Without Peter'
        elif pred == 0 and true_label == 1:
            fn += 1  # Incorrectly predicted 'Without Peter'

Print results
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")
```



### Fitting Model
Out of the three, we can see that ResNet50 v2 is most likely to be within the ideal range for model complexity on the Fitting Model. Although the val_accuracy for that model is less than v1, the closer accuracy vs val_accuracy and loss vs val_loss scores suggest that the v2 model is a better fit.

## Next Model
For our next model, we are considering data augmentation. By adding in layers that can flip, invert, or otherwise alter the image, it can help our model better generalize. We are also thinking about using Deeper Custom CNN to test its ability to classify the presence of Peter against the models we tested for this milestone.

## Conclusion
As we shifted into using a deeper neural network, out test accuracy has improved greatly. Theoretically, our model should be able to correctly identify Peter's presence on screen in more than 19 out of every 20 frames. This makes the model much more reliable, and thus proves our endevour to be a fruitful one. The model's success at classifying frames from models it have not seen shows its versatility and generality. However, there are still aspects where we can shore up the model's overall accuracy, which we will work on for the next milestone. For our next milestone, we will also try to apply our model to an entire unseen episode of Family Guy.

# Milestone 3: Pre-Processing
## Pre-Processing
To make our processing more efficient and to address the inconsistency between image sizes, we decided to downscale all of our images to be the same size, 320x240. Because of its nature as a animated TV sitcom series, Family Guy is composed of mostly simple frames. Thus, downsizing the individual frames will not lose much detail and should not have a too much of a detrimental effect on the model's ability to classify the frames as with Peter vs. without Peter. However, the decreased sizes should greatly increase the speeds at which the model's downloading and processing times. We used a simple **[python script](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone3/milestone3/Image%20resizing.ipynb)** to decrease the image sizes.

## First Model
### Training
In order to focus on testing the feasibility of our project, we decided to restrict our dataset to images from a singular episode (S5E04). We applied Geek-for-Geek's **[cat-vs-dog model](https://www.geeksforgeeks.org/cat-dog-classification-using-convolutional-neural-network-in-python/)** and fit it for our image data. Running **[our model](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone3/milestone3/Model_Initial.ipynb)** yielded promising signs, yet was clearly not in tune for the specific problem we are tackling. However, our model suggested that we were on the right path. And, after much tweaking, such a technique can most likely be successful for accomplishing the goals of our project.


### Evaluating
After ten epochs, our training accuracy settled at 99.59% and our testing accuracy at 85.87%. Manually testing several frames from this episode suggested that these numbers were accurate. In order to check for overfitting, we also gave the model a few frames from a different episode (S1E05) to classify, with the results shown below.

   <br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone3/milestone3/assets/S1E05_TEST_1.png" alt="With Peter" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone3/milestone3/assets/S1E05_TEST_2.png" alt="With Peter2" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone3/milestone3/assets/S1E05_TEST_3.png" alt="With Peter3" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone3/milestone3/assets/S1E05_TEST_4.png" alt="With Peter3" width="400"/>
        </tr>
   </table>

Although the accuracy of this quick test seems to align with our testing accuracy, it gives us some idea about where the model may be inaccurate.

### Fitting Model
Our model fits in the overfitting region of the fitting graph. It is well shown in the training and validation metrics (graphs) resulted in Model_initial.ipynb:
<br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone3/milestone3/assets/Training Accuracy and Validation Accuracy.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone3/milestone3/assets/Training Accuracy and Validation Accuracy.png" alt="Training Loss and Validation Loss" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>

The model achieves relatively low training loss (0.0155) and high training accuracy (0.9959). On the other hand, the model achieves relatively high training loss (0.5099) and low validation accuracy (0.8587). In addition, both validation loss and accuracy fluctuates significantly; validation loss spikes twice up to 7.8551 and 2.8553 before reaching 0.5099 at the final epoch. 

Therefore, our conclusion is that our model now is too complex for our dataset, which led poor generalization to the validation set and overfitting issue (even though worked well on the training data). 

To improve our models, we can try several options for our next model as the following:
- Reduce the number of convolutional and dense layers
- Decrease the filters/units in each layers
- Increase dropout
- Adding regularization to layers
- Adjusting learning rate
- Increase training dataset

## Conclusion
Our model has a testing accuracy of just ~85%, which does beat out random guessing, suggestig that we are on the right track. However, the accuracy is much too low for our objective, especially when considering that the testing and training data all come from the same episode. From testing with images from other episodes, we can see that details such as Peter wearing different clothes may trip up the model, suggesting overfitting. 

There are many methods that we are considering to improve upon our initial model. They include:
- Regularizing our data
- Modifying our train/test splits
- Reducing/changing the layers of the neural network
- Optimizing the number of epochs
- Using all our image data for training and testing

We hope that this course of action will greatly improve the reliability of our model.

# Milestone 2: Data Exploration & Initial Preprocessing

## Data Exploration and Plot
### Overview
Our model aims to recognize whether Peter Griffin is on screen during any given moment of a Family Guy episode. In order to train and test our model, we extracted frames of Family Guy episodes and manually classfied them by the presence or absence of Peter Griffin.

### Extraction method
To check which episode includes Peter Griffin, we used **[Family Guy Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/family-guy-dataset/data)** from Kaggle. This dataset includes various information about each Episode/Season of Family Guy. We performed data exploration on this dataset as the [following](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone2/milestone2/Family_Guy_Episode_Extract.ipynb):
<details>
<summary>Click to collapse code</summary>

```python
import pandas as pd
import kagglehub
import os

path = kagglehub.dataset_download("iamsouravbanerjee/family-guy-dataset")
files = os.listdir(path)
print(files)

file_path = os.path.join(path, 'Family Guy Dataset.csv')
df = pd.read_csv(file_path)
filtered_episodes = df[df['Featuring'].str.contains(r'\bPeter\b', case=False, na=False)]
filered_Season = filtered_episodes[['Season', 'No. of Episode (Season)']]
season1_filtered_episodes = filered_Season[filered_Season['Season'] == 1]
season2_filtered_episodes = filered_Season[filered_Season['Season'] == 2]
season3_filtered_episodes = filered_Season[filered_Season['Season'] == 3]
season4_filtered_episodes = filered_Season[filered_Season['Season'] == 4]
season5_filtered_episodes = filered_Season[filered_Season['Season'] == 5]
```
**Example Result**:
```plaintext
| Season | No. of Episode (Season) |
|--------|-------------------------|
| 1      |           1             |
| 1      |           2             |
| 1      |           3             |
| 1      |           4             |
| 1      |           5             |
| 1      |           6             |
| 1      |           7             |
```
</details><br>

Then we assigned each person one episode to extract frames for our dataset. We used following two methods to achieve it:

1. **[Video Frame Extractor](https://frame-extractor.com/)**: We set distance between frames as 500ms so that we can get 2 frames per second. We experimeneted with few distances and found that 2fps result in fastest clear image extraction. 
2. **[Frame_Extractor.ipynb](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone2/milestone2/Frame_Extractor.ipynb)**: Some group members encountered error while using frame extractor above, so we used cv2 library to extract by ourselves as the below:
    <details>
    <summary>Click to collapse code</summary>
    
    ```python
    import cv2
    import os
    import zipfile

    def extract_frames(video_path, output_folder, frame_rate=1):
        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps / frame_rate)  # Capture every 'interval' frame

        frame_count = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save frames at the specified interval
            if frame_count % interval == 0:
                frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"Extracted {saved_count} frames to {output_folder}")

    video_path = 's5e4.mp4'
    output_folder = 'extracted_frames'
    extract_frames(video_path, output_folder, frame_rate=2)

    def zip_extracted_frames(folder_path, zip_filename="extracted_frames.zip"):
        # Create a Zip file
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            # Loop through all files in the folder
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))
        
        print(f"Frames from '{folder_path}' have been zipped into '{zip_filename}'")

    folder_path = 'extracted_frames'  # Folder with extracted frames
    zip_filename = 'extracted_frames.zip'
    zip_extracted_frames(folder_path, zip_filename)
    ```
    </details>

### Number of Classes and Example Classes
1. Presence of Peter Griffin
   <br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone2/milestone2/assets/with_peter_example.png" alt="With Peter" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone2/milestone2/assets/with_peter_example2.png" alt="With Peter2" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone2/milestone2/assets/with_peter_example3.png" alt="With Peter3" width="400"/>
        </tr>
   </table>
   
   Note: We classified the scenes where only small portion of Peter Griffin appears as presence of Peter Griffin. However, this standard may be changed later as we perform our model training. 
2. Absence of Peter Griffin
   <br><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone2/milestone2/assets/without_peter_example.png" alt="Without Peter" width="400"/>

### Size of Image
Depending on the extraction methods that each group member used, the image size varies.
1. 1440x1080 png (from Video Frame Extractor)
2. 320x240 jpg (from Frame_Extractor.ipynb*)
<br><br>

### Image Naming Convention
Every image will be labeled as SAEBB_###. For example, the 12th extracted frame of Season 5, Episode 5 will be named as S5E05_12.

### Plot Our Data
To download the dataset into Jupyter Notebook from google drive, we used the following code snippet which uses "**gdown**" library:
<details>
  <summary>Click to Collapse Code</summary>

  ```python
  !pip install gdown

  import os
  import zipfile
  import gdown
  
  # List of file IDs (from google drive that stores our dataset)
  file_ids = [
      '1pF2dgzRh5bv7N6CtnNbxSdTY96-WtscQ',
      '1_AnxC-HX0cONqzx6iieZ-jVuUfeBGikv',
      '1NPUlNQ72xjnu0EYWWbThB72UFSsTtrDL',
      '1qdo1P-FaShwrf8uAtVganaOK1jPpproM',
      '1EypHEU0fZHWQECoNCxpmgRHWRlsB9_eA',
      '18-vCtuExrjNObIbwSoskgBf59vOK1Hhv'
  ]
  
  # Folder to store downloaded ZIP files
  zip_folder = 'zip_files'
  os.makedirs(zip_folder, exist_ok=True)
  
  # Folder to extract all ZIP files
  extract_to = 'extracted_images'
  os.makedirs(extract_to, exist_ok=True)
  
  # download and extract all ZIP files from our google drive
  for i, file_id in enumerate(file_ids, 1):
      zip_path = os.path.join(zip_folder, f'file_{i}.zip') # each ZIP file will be stored as file_#.zip
      
      # downloading file from google drive
      gdown.download(f'https://drive.google.com/uc?id={file_id}', zip_path, quiet=False)
  
      # Extract the downloaded ZIP file
      with zipfile.ZipFile(zip_path, 'r') as zip_ref:
          zip_ref.extractall(extract_to)
  
  print(f"Done, stored in {extract_to}")
  ```
</details>

Then to access the data, we can directly use the images under each episode folder (under extracted_images folder). Below is the example code snippet of how we access to the number of images in each folder:

<details>
  <summary>Click to Collapse Code</summary>
  
  ```python
  main_folder = 'extracted_images'

  image_extensions = {'.jpg', '.jpeg', '.png'}
  
  image_counts = {}
  
  # Loop through each episode folder (e.g., S1E5)
  for episode_folder in os.listdir(main_folder):
      episode_path = os.path.join(main_folder, episode_folder)
  
      if os.path.isdir(episode_path):
          image_counts[episode_folder] = {'With_Peter': 0, 'Without_Peter': 0}
  
          for subfolder in ['With_Peter', 'Without_Peter']:
              subfolder_path = os.path.join(episode_path, subfolder)
  
              if os.path.isdir(subfolder_path):
                  image_count = sum(
                      1 for file_name in os.listdir(subfolder_path)
                      if os.path.splitext(file_name)[1].lower() in image_extensions
                  )
  
                  image_counts[episode_folder][subfolder] = image_count
  
  for episode, counts in image_counts.items():
      print(f"Episode '{episode}':")
      print(f"  With_Peter: {counts['With_Peter']} images")
      print(f"  Without_Peter: {counts['Without_Peter']} images")
  ```
</details>

Finally, these are two plots that show the number of images in each episode and total number of images (with_peter vs. without_peter):
1. Number of Images in Each Episode (With_Peter vs Without_Peter)
   <br><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone2/milestone2/assets/number_of_images_each_episode.png" alt="Number of images in each episode" width="800"/>
2. Total Number of Images Across All Episodes
   <br><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone2/milestone2/assets/number_of_images_total.png" alt="Number of images total" width="300"/>
   
   Note: Since our dataset includes more Without_Peter images than With_Peter images, we may remove some Without_Peter images or add more With_Peter images to make both category in almost same size (i.e. number of images).

The Jupyter Notebook that includes data download and plotting information can be found at **[Data_Download.ipynb](https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone2/milestone2/Data_Download.ipynb)**

## Preprocessing
In our initial exploration of the data, we identified several key preprocessing considerations that will guide our approach for training a CNN model.

### Points to Consider
1. Image Size and Dimensions
   - Setting a standard input dimension is crucial for our CNN training process. To determine the optimal dimensions, we will experiment with different resolutions, balancing between higher dimensions, which offer more detail but require greater computational power, and lower dimensions, which are computationally efficient but capture less detail. This trade-off will allow us to find an effective compromise between detail and efficiency.
2. Background Masking
   - For potential future segmentation tasks, we could implement background masking to distinguish between pixels belonging to Peter and those that do not. However, as this approach requires pixel-level annotation, it will involve significant manual work. Therefore, we will consider this option only after we establish reliable frame-level detection of Peter in the scenes.
3. Cropped Face Image
   - Another possible enhancement involves detecting and cropping faces of various characters in each frame, followed by classifying whether a cropped character is Peter. Similar to background masking, this approach would require manual labeling and inspection, so we may pursue it in later stages of the project if it proves beneficial.
    
### Initial Approach
For our initial model training, we plan to use the raw frames and standardize them to a fixed 3D matrix format for color images. We will explore various image processing techniques to optimize the input, such as resizing with interpolation or anti-aliasing, based on their impact on performance. This straightforward approach will allow us to establish a baseline and later experiment with more advanced preprocessing techniques as needed.
