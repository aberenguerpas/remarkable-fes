# **Disease Classification Task**

## Dataset description

The dataset used in this experiment consists of images of almond tree leaves and branches, showcasing the effects of various biotic and abiotic issues on the trees. The primary goal of this dataset is to support the classification and study of different diseases affecting almond trees.

#### Data Collection

The images in the dataset were collected from a local farm using various models of mobile phones, leading to a diverse set of resolutions and image qualities. The variability in image size is due to the different camera resolutions of the mobile devices used during data acquisition.

#### Dataset Structure

The dataset is categorized based on the type of disease affecting the almond trees. Two primary categories of diseases were identified:

1. **Fungal Diseases (MFx)** : Diseases caused by fungal infections.
2. **Insect-related Diseases (Rx)** : Diseases caused by insect infestations.

The dataset is further classified into the following specific disease classes:

* **MF1, MF3, MF4** : Representing different types of fungal diseases.
* **R1, R3, R5, R6** : Representing various stages and effects of insect-related diseases.

For each class, the dataset includes images that capture the progression of the issue across different stages, providing valuable insights into disease development and impact.

#### Dataset Challenges

1. **Class Imbalance** : One of the significant challenges encountered while working with this dataset is the class imbalance. Certain classes have a limited number of images, which poses difficulties in developing a robust and reliable classification system.
2. **Resolution Variability** : The varying resolutions of the images due to differences in mobile phone cameras introduce an additional layer of complexity in preprocessing and standardizing the dataset for model training.

Despite these challenges, the dataset represents a valuable resource for understanding and addressing almond tree diseases. Future work could include augmenting the dataset to balance the class distribution and standardizing image resolutions to improve the robustness of classification models.

## **DATASET PREPROCESSING**

Augmentation strategies

* **Rotations:** Between -45º and 45º degrees. With the aim to simulate the possible angles while taking the pictures
* **Horizontal/Vertical Flips:** Another way to simulate the position of the leaf while taking the picture.
* **Translations:** Between 100 and 200 px. Not always the picture will be centered.
* **Scale:** Vary the leaf size.
* **Crops:** Select a subregion, that is specilly useful when the issue is focused on one zone.
* **Bright Variations:** Simulate different bright conditions
* **Adjust hue:** Simulate differente camera qualities
* **Noise:** Add gaussian noise and blur to simulate camera issues.
* **Deformations:** In order to simulate different leaf persperctives

Clean, preprocess, and organize it to ensure it’s ready for further analysis and modeling.

Carefully document your process, including challenges faced and how they were addressed

**CLASSIFICATION MODEL RESULTS**

Experiment with different classification algorithms, evaluate their performance, and document the results.

**Insights and results from the classification model.**

The first try of classification model is composed by a simple neuronal network. That follows this strcuture:

The results were not so good:

Loss: 0.903, Accuracy: 0.667
