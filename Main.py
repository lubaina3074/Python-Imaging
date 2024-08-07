import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import cv2
import numpy as np
import random


#editing the first original hallway image which then later will have background charactwrs added to it



# Load the image
image = Image.open('hallway images/clean hallway.jpg')

filtered_image = image.filter(ImageFilter.MedianFilter(size=3))


out = filtered_image.filter(ImageFilter.DETAIL)

# Enhance brightness and contrast
enhancer = ImageEnhance.Contrast(out)
bright_image = enhancer.enhance(1.1)  # Increase contrast by 20%

enhancer = ImageEnhance.Brightness(bright_image)
enhanced_image = enhancer.enhance(1.3)  

enhancer = ImageEnhance.Sharpness(enhanced_image)
sharpened_image = enhancer.enhance(1.5)

# Enhance saturation
saturation_factor = 1.1
saturation_enhancer = ImageEnhance.Color(sharpened_image)
final_image = saturation_enhancer.enhance(saturation_factor)

# Adjust color balance by manipulating RGB channels
def adjust_color_balance(image, red_factor, green_factor, blue_factor):
    r, g, b = image.split()
    r = r.point(lambda i: i * red_factor)
    g = g.point(lambda i: i * green_factor)
    b = b.point(lambda i: i * blue_factor)
    return Image.merge('RGB', (r, g, b))

# Adjust color balance (example: increase red, decrease blue)
adjusted_image = adjust_color_balance(final_image, red_factor=0.8, green_factor=0.9, blue_factor=0.9)
adjusted_image.save('adjusted_image.jpg')










#adding the background characters to the hallway image

image1 = Image.open('background1.png')
background = Image.open("adjusted_image.jpg")

# Resize the image to a new width and height
new_width = 750
new_height = 600
resized_image = image1.resize((new_width, new_height))
rotated_image = resized_image.rotate(0)

brightness_factor = 0.8  # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)



background.paste(dark_image, (100,300), dark_image.convert('RGBA'))




image2 = Image.open('background2.png')


# Resize the image to a new width and height
new_width = 750
new_height = 600
resized_image = image2.resize((new_width, new_height))
rotated_image = resized_image.rotate(0)

brightness_factor = 0.8  # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)



background.paste(dark_image, (300,300), dark_image.convert('RGBA'))




image3 = Image.open('background3.png')


# Resize the image to a new width and height
new_width = 750
new_height = 700
resized_image = image3.resize((new_width, new_height))
rotated_image = resized_image.rotate(0)

brightness_factor = 1.1 # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)



background.paste(dark_image, (200,400), dark_image.convert('RGBA'))


image4 = Image.open('background4.png')


# Resize the image to a new width and height
new_width = 300
new_height = 550
resized_image = image4.resize((new_width, new_height))
rotated_image = resized_image.rotate(0)

brightness_factor = 1.5  # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)



background.paste(dark_image, (120,500), dark_image.convert('RGBA'))




image5 = Image.open('background5.png')


# Resize the image to a new width and height
new_width = 200
new_height = 260
resized_image = image5.resize((new_width, new_height))
rotated_image = resized_image.rotate(0)

brightness_factor = 0.6  # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)




background.paste(dark_image, (340,510), dark_image.convert('RGBA'))






#editing the sword
imageObject = Image.open("sword1.png")

# Do a flip of left and right
flippedImage = imageObject.transpose(Image.FLIP_LEFT_RIGHT)

new_width = 300
new_height = 300
resized_image = flippedImage.resize((new_width, new_height))
rotated_image = resized_image.rotate(-20)


brightness_factor = 1.5 # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)

background.paste(dark_image, (0,700), dark_image.convert('RGBA'))

background.save('start_hallway.png')









#doing the final editing of the first image


#creating sun particles

def create_sunlight_particles(base_image_path, output_image_path, position, particle_count=100, particle_size=(5, 10), vertical_spread=50):
    # Open the base image
    base_image = Image.open(base_image_path).convert("RGBA")

    # Create a new image for the particles with the same size as the base image
    particles_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))

    # Create a draw object to draw on the particles layer
    draw = ImageDraw.Draw(particles_layer)

    # Draw particles (fairy lights)
    for _ in range(particle_count):
        # Random position around the given position with larger spread
        x = position[0] + random.randint(-100, 100)
        y = position[1] + random.randint(-vertical_spread, vertical_spread)
        
        # Random size for each particle
        particle_radius = random.randint(particle_size[0], particle_size[0])
        
        # Random opacity for a more natural look
        opacity = random.randint(100, 100)
        
        # Draw an ellipse (particle) with a less yellow color
        draw.ellipse(
            (x - particle_radius, y - particle_radius, x + particle_radius, y + particle_radius), 
            fill=(255, 255, 205, opacity)  # White color with varying opacity
        )

    # Apply a Gaussian blur to the particles layer for a glow effect
    particles_layer = particles_layer.filter(ImageFilter.GaussianBlur(3))

    # Composite the particles layer onto the base image
    result_image = Image.alpha_composite(base_image, particles_layer)

    # Save the result
    result_image.save(output_image_path)

    # # Optionally, show the result
    # result_image.show()

# Usage example
create_sunlight_particles(
    base_image_path="start_hallway.png",
    output_image_path="new_hallway.png",
    position=(700, 400),
    vertical_spread=60 # Replace with the desired position for the particles
)


# Read the image
image6 = cv2.imread('start_hallway.png')

# Apply Bilateral Filter
bilateral_filter = cv2.bilateralFilter(image6, 9, 15, 60)



# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image6, cv2.COLOR_BGR2HSV)

# Adjust Hue
hue_factor = -2 
hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_factor) % 180

yellow_hue_min =40
yellow_hue_max = 60
mask = cv2.inRange(hsv_image, (yellow_hue_min, 100, 100), (yellow_hue_max, 255, 255))
hsv_image[:, :, 1][mask != 0] = np.clip(hsv_image[:, :, 1][mask != 0] * 0.8, 0, 255)

# Adjust Saturation
saturation_factor = 0.9
hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)

# Adjust Value (Brightness)
value_factor = 1.1  
hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * value_factor, 0, 255)

# Convert back to BGR color space
enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

cv2.imwrite('new_hallway.jpg', enhanced_image)







#adding a sunlight glow to the image

def add_sunlight_glow(base_image_path, output_video_path, frames=30, fps=10):
    # Open the base image
    base_image = Image.open(base_image_path).convert("RGBA")
    
    # Get the size of the image
    width, height = base_image.size
    
    # Prepare a list to store the frames
    frame_list = []

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Create frames with changing sunlight glow
    for i in range(frames-1,-1,-1):
        # Create an overlay with a gradually changing yellow color
        overlay = Image.new("RGBA", (width, height), (255, 223, 186, int(30 * (i / frames))))
        
        # Composite the overlay onto the base image
        frame = Image.alpha_composite(base_image, overlay)
        
        # Optionally, enhance the brightness and color to simulate sunlight glow
        enhancer = ImageEnhance.Brightness(frame)
        frame = enhancer.enhance(1 + (i / frames) * 0.2)  # Gradually increase brightness
        
        enhancer = ImageEnhance.Color(frame)
        frame = enhancer.enhance(1 + (i / frames) * 0.1)  # Gradually increase color saturation
        
        # Convert frame to RGB
        frame = frame.convert("RGB")
        
        # Convert the frame to a format suitable for OpenCV
        frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video file
        video_writer.write(frame_cv)
    
    # Release the video writer
    video_writer.release()

# Usage example
add_sunlight_glow(
    base_image_path="new_hallway.png",
    output_video_path="new_hallway.mp4",
    frames=200,  # Adjust the number of frames for a smoother or faster transition
    fps=5 # Frames per second
)

# Display the video using OpenCV
cap = cv2.VideoCapture("new_hallway.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Sunlight Glow Animation', frame)

        # Press 'q' to exit the video display
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



