
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import os



#editing the first stance
image1 = Image.open('character.png')


# Resize the image to a new width and height
new_width = 750
new_height = 700
resized_image = image1.resize((new_width, new_height))
rotated_image = resized_image.rotate(0)

brightness_factor = 0.4  # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)

background = Image.open("hallway images/final_image.png")

background.paste(dark_image, (30,380), dark_image.convert('RGBA'))


background.save('character_in_hallway.png')






# #editing the second stance

image3 = Image.open('character2.png')


# Resize the image to a new width and height
new_width = 750
new_height = 700
resized_image = image3.resize((new_width, new_height))
rotated_image = resized_image.rotate(0)

brightness_factor = 0.4  # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)

background = Image.open("hallway images/final_image.png")

background.paste(dark_image, (30,380), dark_image.convert('RGBA'))
background.save('character2_in_hallway.png')








#animating the movement of the character using the two character in the hallway images 

# Open the character images with different stances
character1 = Image.open("character_in_hallway.png")
character2 = Image.open("character2_in_hallway.png")

# Resize the character images to a common size
width, height = 750, 790
character1_resized = character1.resize((width, height))
character2_resized = character2.resize((width, height))

# Convert character images to RGB mode if they are not already
character1_resized = character1_resized.convert("RGB")
character2_resized = character2_resized.convert("RGB")

# Define the number of frames for each stage of the transition
num_frames_turning = 100  # More frames for turning
num_frames_other = 50    # Fewer frames for other stages

# Define the frame rate of the output video (adjust for slower movement)
fps = 1000

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video_file = "character_transition_slow.mp4"
video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

# Blend the two character images with a smooth transition and show in a window
for i in range(num_frames_turning + num_frames_other):
    if i < num_frames_turning:
        # Turning stage: More frames
        alpha = i / num_frames_turning
    else:
        # Other stages: Fewer frames
        alpha = 1.0
    
    # Increase the number of frames gradually towards the end
    if i >= num_frames_turning - 5:  
        num_frames_other += 2

    # Blend the character images
    blended_character = Image.blend(character1_resized, character2_resized, alpha)

    # Convert the blended character image to a numpy array
    frame = np.array(blended_character)

    # Convert the frame from RGB to BGR format (required by OpenCV)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Show the frame in a window
    cv2.imshow("Character Animation", frame_bgr)

    # Write the frame to the video file
    video_writer.write(frame_bgr)

    # Wait for a key press
    key = cv2.waitKey(1000 // fps)  # Wait for the frame duration

    # Check if the second image is displayed and pause the program
    if i == num_frames_turning + 1:
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed to resume

    # Check if the user pressed 'q' to quit
    if key == ord('q'):
        break

# Release the VideoWriter object
video_writer.release()

# Close the OpenCV window
cv2.destroyAllWindows()

print("Video saved successfully.")
