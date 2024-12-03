import os
import albumentations as A
import cv2

augmentation_pipeline = A.Compose([
    A.Rotate(limit=45, p=0.8),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.6),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.4),
    A.Perspective(scale=(0.05, 0.1), p=0.5),
    A.RandomCrop(width=500, height=500, p=0.5),
    A.Resize(height=1600, width=1200)
])


    # Aplicar la aumentación
def augment_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = augmentation_pipeline(image=image)
    return augmented["image"]

# Ejemplo de uso
for file_name in os.listdir("dataset/MF1/"):
    for i in range(3):
        augmented_image = augment_image(f"dataset/MF1/{file_name}")
        augment_image_path = f"dataset/MF1/ext_{i}_{file_name}"
        cv2.imwrite(augment_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))