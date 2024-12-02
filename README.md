# Milestone 4: Second Model

## Table of Contents
- [Milestone 4: Second Model](#milestone-4-second-model)
  - [Table of Contents](#table-of-contents)
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
  - [Conclusion](#conclusion)
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
1. [ResNet50 v1](#)
   <br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/resnet_v1_accuracy.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/resnet_v1_loss.png" alt="Training Loss and Validation Loss" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>
   Comparing to the simple CNN model from Milestone 3, we can check that our new model improved in large scale. First of all, the validation accuracy increased from 0.8587 to 0.9726. In addition, validation accuracy remians more stable and less fluctuated comparing to our previous model which had intense fluctuation, especially at epoch 3 and 7; validation accuracy dropped exponentially. 
2. [ResNet50 v2](#)
   <br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/resnet_v2_accuracy.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/resnet_v2_loss.png" alt="Training Loss and Validation Loss" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>
   Second version of Resnet showed lower accuracy and higher loss than first version (even though the differences were really small). However, there were huge improvements as we finally overcame (not 100% though) the fluctuation issue and overfitting problem. Our validation accuracy and loss aligns almost same with training ones; At the end, we are still experiencing with slight overfitting pattern, but we think that we can reduce that sign of overfitting, for example by making epoch to be not processed by lr scheduler.
3. [VGG16 v1](#)
   <br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/vgg_accuracy.png" alt="Training Accuracy and Validation Accuracy" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/vgg_loss.png" alt="Training Loss and Validation Loss" alt="Training Loss and Validation Loss" width="400"/>
        </tr>
   </table>
   VGG model also resolves some fluctuating graphs and overfitting issue, but had significantly low accuracy and loss comparing to ResNets. In addition, it took way longer to train the VGG than ResNet or CNN. 

Lastly, similar to Milestone2 model, two ResNet models listed above correctly classified the unseen data (5 images); VGG classified one unseen data (out of 5) wrong. 
The unseen images in question:
   <br>
   <table>
        <tr>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/out-10000.jpg" alt="Image 1" width="400"/>
            <td><img src="https://github.com/cse151a-fa24-group-project/CSE151A-Project-ML/blob/Milestone4/milestone4/assets/out-20000.jpg" alt="Image 2" width="400"/>
        </tr>
   </table>

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
