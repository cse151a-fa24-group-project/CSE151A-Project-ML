# Milestone 2: Data Exploration & Initial Preprocessing

## Table of Contents
- [Milestone 2: Data Exploration \& Initial Preprocessing](#milestone-2-data-exploration--initial-preprocessing)
  - [Table of Contents](#table-of-contents)
- [Data Exploration and Plot](#data-exploration-and-plot)
  - [Overview](#overview)
  - [Extraction method](#extraction-method)
  - [Number of Classes and Example Classes](#number-of-classes-and-example-classes)
  - [Size of Image](#size-of-image)
- [Preprocessing](#preprocessing)


# Data Exploration and Plot
## Overview
Our model aims to recognize whether Peter Griffin is on screen during any given moment of a Family Guy episode. In order to train and test our model, we extracted frames of Family Guy episodes and manually classfied them by the presence or absence of Peter Griffin.

## Extraction method
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

## Number of Classes and Example Classes
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

## Size of Image
Depending on the extraction methods that each group member used, the image size varies.
1. 1440x1080 png (from Video Frame Extractor)
2. 320x240 jpg (from Frame_Extractor.ipynb*)
<br><br>

# Preprocessing
We found few things to be considered after exploring our data.

1. Image Size
2. Background Masking
3. Cropped Face Image
4. etc



























------------------------
*previous README.md*

CSE151A FA24 group project 

Data Gathering:
1. Get the video file for a specific episode.

2. Feed through Handbrake to convert to a mp4 file if needed.

3. Feed through [frame-extractor](https://frame-extractor.com/), set "Distance between frames..." as 500 ( 500 ms, or 2 frames per second), and calculate the total number of frames based on episode length

4. Name a folder SAEBB, where A is season number and BB is episode number. Inside that folder, have two folders, named With_Peter and Without_Peter. Place the images into the corresponding folders.

5. Upload the SAEBB folder to the Google Drive folder.


Preprocessing steps:
1. Size
We will make sure that every image extracted will be of uniform size.

2. Blurry Frames
We will toss any frames too blurry to easily identify if Peter is present.
