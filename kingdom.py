from PIL import Image, ImageEnhance, ImageFilter, ImageChops, ImageOps, ImageDraw
import cv2
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pygame

# editing the original kingdom picture

background = Image.open("kingdom.jpeg")

new_width = 900
new_height = 700
background2 = background.resize((new_width, new_height))

# Reduce brightness
enhancer = ImageEnhance.Brightness(background2)
background2 = enhancer.enhance(0.7)  # 0.5 is for reducing brightness, you can adjust this value

# Reduce saturation
enhancer = ImageEnhance.Color(background2)
background2 = enhancer.enhance(0.4)  # 0.5 is for reducing saturation, you can adjust this value

# Reduce contrast
enhancer = ImageEnhance.Contrast(background2)
background2 = enhancer.enhance(0.9)

blue_tint = Image.new('RGB', background2.size, (0, 0, 50))
background2 = Image.blend(background2, blue_tint, 0.3)  # Blend with a 20% blue tint


def add_noise(image, amount=0.2):
    np_image = np.array(image)
    noise = np.random.normal(0, 255 * amount, np_image.shape)
    noisy_image = np_image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)


def charred_edges(image, border_width=50):
    # Create a gradient mask for charred edges
    mask = Image.new('L', image.size, 0)
    gradient = Image.linear_gradient('L')
    gradient = gradient.resize((border_width, border_width))

    for i in range(0, mask.width, border_width):
        for j in range(0, mask.height, border_width):
            mask.paste(gradient, (i, j))

    charred_image = ImageChops.multiply(image, mask.convert('RGB'))
    return charred_image


def make_image_burnt(input_path, output_path, soot_texture_path=None):
    # Open an image file
    with Image.open(input_path) as img:
        # Ensure the image has an alpha channel
        img = img.convert('RGBA')

        # Separate the image into RGBA channels
        r, g, b, a = img.split()

        # Combine the RGB channels back into one image
        rgb_img = Image.merge('RGB', (r, g, b))

        # Step 1: Darken the image
        enhancer = ImageEnhance.Brightness(rgb_img)
        rgb_img = enhancer.enhance(0.5)  # Reduce brightness significantly

        # Step 2: Add noise to simulate soot
        rgb_img = add_noise(rgb_img, amount=0.1)  # Adjust amount for desired soot effect

        # Step 3: Apply a charred border
        rgb_img = charred_edges(rgb_img, border_width=5)  # Adjust border_width as needed

        # Step 4: Overlay soot texture (optional)
        if soot_texture_path:
            soot_texture = Image.open('/Users/lubainakhan/Downloads/soot.jpg').convert('L')
            soot_texture = soot_texture.resize(rgb_img.size)
            soot_texture = ImageOps.invert(soot_texture)
            rgb_img = ImageChops.multiply(rgb_img, soot_texture.convert('RGB'))

        # Recombine the modified RGB channels with the original alpha channel
        burnt_img = Image.merge('RGBA', (rgb_img.split()[0], rgb_img.split()[1], rgb_img.split()[2], a))

        # Save the burnt image
        burnt_img.save(output_path)


make_image_burnt('castle1.png', 'castle2.png')

# editing the castle

image = Image.open('castle2.png')

# Resize the image to a new width and height
new_width = 400
new_height = 390
resized_image = image.resize((new_width, new_height))
rotated_image = resized_image.rotate(0)

brightness_factor = 0.5  # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)

background2.paste(dark_image, (-40, 310), dark_image.convert('RGBA'))

# editing the crows

image = Image.open('crows.png')

# Resize the image to a new width and height
new_width = 600
new_height = 200
resized_image = image.resize((new_width, new_height))
rotated_image = resized_image.rotate(0)

brightness_factor = 0.5  # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)

background2.paste(dark_image, (70, 0), dark_image.convert('RGBA'))






# editing the clouds


background3 = Image.open("dark_kingdom.jpg")
image = Image.open('clouds2.png')

new_width = 400
new_height = 450
resized_image = image.resize((new_width, new_height))


brightness_factor = 0.35  # Adjust as needed (1.0 for no change)
enhancer = ImageEnhance.Brightness(resized_image)
dark_image = enhancer.enhance(brightness_factor)

dark_image = dark_image.filter(ImageFilter.GaussianBlur(radius=4))

blue_tint = Image.new('RGBA', dark_image.size, (0, 0, 50, 0))
dark_image = Image.alpha_composite(dark_image.convert('RGBA'), blue_tint)

background2.paste(dark_image, (50, -150), dark_image.convert('RGBA'))







#ediiting the cliff


image2 = Image.open('cliff.png')

new_width = 900
new_height = 990
resized_image = image2.resize((new_width, new_height))
rotated_image = resized_image.rotate(0)

brightness_factor = 0.3  # Adjust as needed (1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)

dark_image = dark_image.filter(ImageFilter.GaussianBlur(radius=1))

background2.paste(dark_image, (200, 70), dark_image.convert('RGBA'))








# editing the clouds

image2 = Image.open('clouds2.png')

new_width = 450
new_height = 500
resized_image = image2.resize((new_width, new_height))

brightness_factor = 0.35  # Adjust as needed (1.0 for no change)
enhancer = ImageEnhance.Brightness(resized_image)
dark_image = enhancer.enhance(brightness_factor)

dark_image = dark_image.filter(ImageFilter.GaussianBlur(radius=4))

blue_tint = Image.new('RGBA', dark_image.size, (0, 0, 50, 0))
dark_image = Image.alpha_composite(dark_image.convert('RGBA'), blue_tint)

background2.paste(dark_image, (450, -150), dark_image.convert('RGBA'))

background2.save("kingdom.jpg")












# editing the moon


def create_crescent_moon_overlay(size, moon_radius, moon_position):
    width, height = size
    moon_overlay = Image.new('RGBA', size, (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(moon_overlay)

    # Define the color for the moon (greyish)
    moon_color = (200, 200, 200, 255)  # Slightly greyish white

    # Draw the larger circle for the crescent moon
    moon_center = (moon_position[0], moon_position[1])
    draw.ellipse(
        (moon_center[0] - moon_radius, moon_center[1] - moon_radius,
         moon_center[0] + moon_radius, moon_center[1] + moon_radius),
        fill=moon_color  # Slightly greyish white moon
    )

    # Draw the smaller circle to create the crescent effect
    draw.ellipse(
        (moon_center[0] - moon_radius * 0.7, moon_center[1] - moon_radius,
         moon_center[0] + moon_radius * 1.3, moon_center[1] + moon_radius),
        fill=(0, 0, 0, 0)  # Transparent to create the crescent effect
    )

    # Add a glow around the crescent moon
    glow = Image.new('RGBA', size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_radius = moon_radius * 1.5
    for r in range(moon_radius, int(glow_radius)):
        alpha = int(255 * (1 - (r - moon_radius) / (glow_radius - moon_radius)) * 0.7)  # Increased alpha value
        glow_draw.ellipse(
            (moon_center[0] - r, moon_center[1] - r,
             moon_center[0] + r, moon_center[1] + r),
            fill=(200, 200, 200, alpha)  # Greyish glow with increased alpha
        )

    # Blur the glow to make it smoother
    glow = glow.filter(ImageFilter.GaussianBlur(radius=10))
    moon_overlay = Image.alpha_composite(glow, moon_overlay)

    return moon_overlay


# Parameters for the crescent moon
image_size = (800, 600)  # Set to the size of your dark image
moon_radius = 50
moon_position = (100, 80)  # Position of the moon in the upper center of the image

# Create crescent moon overlay
crescent_moon_overlay = create_crescent_moon_overlay(image_size, moon_radius, moon_position)

# Load the background image
background = Image.open("kingdom.jpg").convert('RGBA')

# Resize crescent moon overlay to match the background size
crescent_moon_overlay = crescent_moon_overlay.resize(background.size)

# Composite the crescent moon overlay onto the dark image
combined_image = Image.alpha_composite(background, crescent_moon_overlay)

# save the combined image
combined_image.save("kingdom.png")










# editing fog


def create_fog_overlay(size, opacity=100):
    width, height = size
    fog = Image.new('RGBA', size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(fog)

    for y in range(height):
        alpha = int(opacity * (y / height))  # Adjust opacity based on y-coordinate
        draw.line((0, y, width, y), fill=(255, 255, 255, alpha))

    fog = fog.filter(ImageFilter.GaussianBlur(radius=5))  # Add a blur to the fog
    return fog


# Load the background image
background3 = Image.open("kingdom.png").convert('RGBA')

# Create the fog overlay
fog_overlay = create_fog_overlay(background3.size, opacity=200)  # Adjust opacity value

# Prepare frames for animation
frames = []
num_frames = 300  # Number of frames in the animation
offset_increment = 7  # Increment value for offset in each frame

for frame in range(num_frames):
    # Calculate the offset for moving fog effect
    offset = frame * offset_increment

    # Create a new image by shifting the fog overlay upwards
    moving_fog = Image.new('RGBA', background3.size)
    if offset < background3.height:
        moving_fog.paste(fog_overlay, (0, background3.height - offset), fog_overlay)
    else:
        # When offset exceeds the height, start over
        moving_fog.paste(fog_overlay, (0, 0), fog_overlay)

    # Composite the moving fog with the background image
    animated_frame = Image.alpha_composite(background3, moving_fog)

    # Convert the frame to OpenCV format (BGR)
    cv2_frame = cv2.cvtColor(np.array(animated_frame), cv2.COLOR_RGBA2BGR)

    # Append the frame to the list of frames
    frames.append(cv2_frame)

# Get the shape (width, height) of the frames
height, width, _ = frames[0].shape

# Define the video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4 format
out = cv2.VideoWriter('kingdom.mp4', fourcc, 10.0, (width, height))

# Write frames to video
for frame in frames:
    out.write(frame)

# Release VideoWriter object
out.release()

print("Video saved successfully!")






#play the animation with the sound



# Initialize pygame mixer
pygame.mixer.init()

# Load the sound file
pygame.mixer.music.load("windsound.mp3")

# Start playing the sound
pygame.mixer.music.play()

# Open the video file
cap = cv2.VideoCapture('kingdom.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Fog Animation', frame)

        # Press 'q' to exit the video display
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Ensure the program doesn't exit until the sound finishes playing
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)