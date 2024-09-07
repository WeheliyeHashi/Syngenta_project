#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:35:56 2021

@author: Weheliye
"""

#%%
%matplotlib qt
import cv2
import matplotlib.pyplot as plt
plt.rcParams['image.cmap']='gray'
from pathlib import Path
#%%
# Path to the MP4 video file


#video_path = Path("calibration_board/GBJH006_01_08_1169168181_08_2024-08-01_13-50-24-895_001_Off_Centre.mp4")
video_path = Path("Final_code/Project_1/RawVideos/CTRL - untreated/time 0h/GBJH006_01_01_1169872637_08_2023-03-13_16-14-48-757_001_Off_Centre.mp4")
print(video_path.parent.joinpath('calibartion.jpeg'))
#%%
# Open the video file
cap = cv2.VideoCapture(str(video_path))

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read until the video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Reached end of video or cannot fetch the frame.")
        break

    # Display the resulting frame
    cv2.imshow("Video", frame)
    I =frame

    # Press 'q' on the keyboard to exit the playback
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# When everything is done, release the video capture object
cap.release()

# Close all the frames
cv2.destroyAllWindows()

# %%
plt.figure()
plt.imshow(I[:,:,0])
plt.axis('off')
plt.savefig(video_path.parent.joinpath('calibartion.jpeg'), dpi=300,bbox_inches='tight', pad_inches=0)
# %%
