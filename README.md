# Milestone 3: Pre-processing

## Table of Contents
- [Milestone 3: Pre-processing](#milestone-3-pre-processing)
    - [Table of Contents](#table-of-contents)
  - [Pre-Processing](#pre-processing)

- [Milestone 2: Data Exploration \& Initial Preprocessing](#milestone-2-data-exploration--initial-preprocessing)
  - [Data Exploration and Plot](#data-exploration-and-plot)
    - [Overview](#overview)
    - [Extraction method](#extraction-method)
    - [Number of Classes and Example Classes](#number-of-classes-and-example-classes)
    - [Size of Image](#size-of-image)
    - [Image Naming Convention](#image-naming-convention)
    - [Plot Our Data](#plot-our-data)
  - [Preprocessing](#preprocessing)


# Pre-Processing




# Milestone 2: Data Exploration & Initial Preprocessing

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
# Preprocessing
In our initial exploration of the data, we identified several key preprocessing considerations that will guide our approach for training a CNN model.

## Points to Consider
1. Image Size and Dimensions
   - Setting a standard input dimension is crucial for our CNN training process. To determine the optimal dimensions, we will experiment with different resolutions, balancing between higher dimensions, which offer more detail but require greater computational power, and lower dimensions, which are computationally efficient but capture less detail. This trade-off will allow us to find an effective compromise between detail and efficiency.
2. Background Masking
   - For potential future segmentation tasks, we could implement background masking to distinguish between pixels belonging to Peter and those that do not. However, as this approach requires pixel-level annotation, it will involve significant manual work. Therefore, we will consider this option only after we establish reliable frame-level detection of Peter in the scenes.
3. Cropped Face Image
   - Another possible enhancement involves detecting and cropping faces of various characters in each frame, followed by classifying whether a cropped character is Peter. Similar to background masking, this approach would require manual labeling and inspection, so we may pursue it in later stages of the project if it proves beneficial.
    
## Initial Approach
For our initial model training, we plan to use the raw frames and standardize them to a fixed 3D matrix format for color images. We will explore various image processing techniques to optimize the input, such as resizing with interpolation or anti-aliasing, based on their impact on performance. This straightforward approach will allow us to establish a baseline and later experiment with more advanced preprocessing techniques as needed.
