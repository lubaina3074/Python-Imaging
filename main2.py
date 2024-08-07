import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance




#editing the hallway image to make it look gloomy and overgrown


# Load the image
image = cv2.imread('hallway images/clean hallway.jpg')

# Darken the image
value = 60  # Adjust the value to control darkness
darkened_image = cv2.subtract(image, np.ones(image.shape, dtype='uint8') * value)
cv2.imwrite('darkened_image.jpg', darkened_image)

# # Display the darkened image
# plt.figure(figsize=(5, 7))  
# plt.imshow(cv2.cvtColor(darkened_image, cv2.COLOR_BGR2RGB), aspect='auto')
# plt.axis('off')  
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# plt.show()



#editing of the first ivy image

image = Image.open('plant images/ivy.png')


# Resize the image to a new width and height
new_width = 300
new_height = 200
resized_image = image.resize((new_width, new_height))
rotated_image = resized_image.rotate(90)

brightness_factor = 0.5  # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)


dark_image.save('plant images/image2.png')


background = Image.open("darkened_image.jpg")
foreground = Image.open("plant images/image2.png")

background.paste(foreground, (10, 600), foreground.convert('RGBA'))





#using plant 5 image
image = Image.open('plant images/plant6.png')


# Resize the image to a new width and height
new_width = 100
new_height = 100
resized_image = image.resize((new_width, new_height))
rotated_image = resized_image.rotate(190)

brightness_factor = 0.6 # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)


background.paste(dark_image, (70,780), dark_image.convert('RGBA'))






#editing of the shrub image

image = Image.open('plant images/shrub.png')


# Resize the image to a new width and height
new_width = 200
new_height = 100
resized_image = image.resize((new_width, new_height))
rotated_image = resized_image.rotate(200)

brightness_factor = 0.5  # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)


dark_image.save("plant images/shrub2.png")


Shrub3 = Image.open("plant images/shrub2.png")
rotated_shrub = Shrub3.rotate(-10, expand=True)
background.paste(rotated_shrub, (700, 800), rotated_shrub.convert('RGBA'))





#using the same ivy image a different way

image = Image.open('plant images/cropivy.png')


# Resize the image to a new width and height
new_width = 50
new_height = 400
resized_image = image.resize((new_width, new_height))
flipped_image = resized_image.transpose(Image.FLIP_LEFT_RIGHT)

brightness_factor = 0.5  # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(flipped_image)
dark_image = enhancer.enhance(brightness_factor)

dark_image.save("plant images/ivy3.png")
ivy = Image.open("plant images/ivy3.png")

background.paste(ivy, (-30,-10), ivy.convert('RGBA'))





#using the plant image

image = Image.open('plant images/plant.png')


# Resize the image to a new width and height
new_width = 300
new_height = 400
resized_image = image.resize((new_width, new_height))
rotated_image = resized_image.rotate(0)

brightness_factor = 0.5  # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)

background.paste(dark_image, (600,100), dark_image.convert('RGBA'))





#using yet another plant image
image = Image.open('plant images/plant3.png')


# Resize the image to a new width and height
new_width = 50
new_height = 70
resized_image = image.resize((new_width, new_height))
rotated_image = resized_image.rotate(0)

brightness_factor = 0.3  # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)


background.paste(dark_image, (700,550), dark_image.convert('RGBA'))





#using another plant image
image = Image.open('plant images/plant4.png')


# Resize the image to a new width and height
new_width = 110
new_height = 300
resized_image = image.resize((new_width, new_height))
rotated_image = resized_image.rotate(0)

brightness_factor = 0.5 # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)

background.paste(dark_image, (500,300), dark_image.convert('RGBA'))




#using plant3 again
image = Image.open('plant images/plant3.png')


# Resize the image to a new width and height
new_width = 50
new_height = 70
resized_image = image.resize((new_width, new_height))
flipped_image = resized_image.transpose(Image.FLIP_LEFT_RIGHT)
rotated_image = flipped_image.rotate(0)

brightness_factor = 0.3  # Adjust as needed (0.0 for black, 1.0 for no change)
enhancer = ImageEnhance.Brightness(rotated_image)
dark_image = enhancer.enhance(brightness_factor)


background.paste(dark_image, (280,550), dark_image.convert('RGBA'))
background.show()

background.save('hallway images/final_image.png')

