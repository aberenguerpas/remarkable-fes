import os
import albumentations as A
import cv2
import math

def augment_image(image_path):
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    augmentation_pipeline = A.Compose([
    A.Rotate(limit=45, p=0.8),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=15, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
    A.MotionBlur(blur_limit=5, p=0.1),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
    A.Perspective(scale=(0.05, 0.1), p=0.5),
    A.Resize(height=height, width=width)
    ])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = augmentation_pipeline(image=image)
    return augmented["image"]

issues_type = ["Healthy","MF1","MF3","MF4","R1","R3","R4","R6"]

for issue in issues_type:
    total = len(os.listdir(f"dataset/{issue}/"))
    for file_name in os.listdir(f"dataset/{issue}/"):
       
        augmentations= int(math.ceil((200-total)/total))
        for i in range(augmentations):
            augmented_image = augment_image(f"dataset/{issue}/{file_name}")
            augment_image_path = f"dataset/{issue}/ext_{i}_{file_name}"
            cv2.imwrite(augment_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))