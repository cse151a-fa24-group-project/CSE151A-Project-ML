# CSE151A-Project-ML
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
