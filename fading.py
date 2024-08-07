
#doing the transition video here where the starting hallway fades into the now hallway


import cv2
import numpy as np

# Load the new_hallway.mp4 video
new_hallway_cap = cv2.VideoCapture("new_hallway.mp4")

# Load the first frame of the character_transition_slow.mp4 video
character_transition_cap = cv2.VideoCapture("character_transition_slow.mp4")
ret, first_frame = character_transition_cap.read()

# Check if the first frame was read successfully
if not ret:
    print("Error: Unable to read the first frame of the character transition video.")
    exit()

# Resize the first frame to match the dimensions of the videos
first_frame = cv2.resize(first_frame, (600, 600))

# Define the output video file name
output_video_file = "transition_video.mp4"

# Define the number of frames for the transition
num_frames = 150

# Define the frame rate of the output video
fps = 200

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (600, 600))

# Morphing in 'num_frames' steps from new_hallway.mp4 to the first frame of character_transition_slow.mp4
for i in range(num_frames):
    percentage = float(i / num_frames)
    print(percentage)
    
    # Read a frame from new_hallway.mp4
    ret, frame = new_hallway_cap.read()
    if not ret:
        break
    
    # Resize the frame to match the dimensions of the videos
    frame = cv2.resize(frame, (600, 600))
    
    # Blend the frame with the first frame of character_transition_slow.mp4
    img_blend = cv2.addWeighted(frame, 1 - percentage, first_frame, percentage, 0)
    
    # Write the blended frame to the output video
    video_writer.write(img_blend)

# Write the remaining frames of character_transition_slow.mp4
while True:
    ret, frame = character_transition_cap.read()
    if not ret:
        break
    
    # Resize the frame to match the dimensions of the output video
    frame = cv2.resize(frame, (600, 600))
    
    # Write the frame to the output video
    video_writer.write(frame)

# Release VideoCapture objects
new_hallway_cap.release()
character_transition_cap.release()

# Release the VideoWriter object
video_writer.release()

print("Video saved successfully.")


# Displaying the video

video_path = "transition_video.mp4"

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    # Read and display each frame of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            # Break the loop if there are no more frames to read
            break
        
        # Display the frame
        cv2.imshow("Transition Video", frame)
        
        # Wait for a short duration (25 milliseconds)
        # Press 'q' to exit the video display
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Release the VideoCapture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()











