import albumentations as A
import cv2
import os
import matplotlib.pyplot as plt

PATH = "C:/Users/orlov/Angelina/AI/GAN/Darcy_GAN/Gano-Cat-Breeds-V1_1/Ragdoll/"
RES_PATH = "C:/Users/orlov/Angelina/AI/GAN/Darcy_GAN/ds_augm/"
AUGMENTATION = 5

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Blur(blur_limit=3),
    A.ShiftScaleRotate(p=0.2, rotate_limit=10),
])

def visualize(image):
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(image)


photo_list = os.listdir(PATH)
image = cv2.imread(PATH + photo_list[0])
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image = cv2.resize(image, (640, 640))

visualize(image)

for img in photo_list:
    image = cv2.imread(PATH + str(img))
    image = cv2.resize(image, (350, 350))
    for i in range(AUGMENTATION):
        augmented_image = transform(image=image)['image']
        cv2.imwrite(RES_PATH + str(img).split(".")[0] + str(i) + ".jpg", augmented_image)