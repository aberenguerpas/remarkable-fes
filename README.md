# **Almond disease Classification Task**

## Dataset description

The dataset used in this experiment consists of images of almond tree leaves and branches, showcasing the effects of various biotic and abiotic issues on the trees. The primary goal of this dataset is to support the classification and study of different diseases affecting almond trees.

#### Data Collection

The images in the dataset were collected from a local farm using various models of mobile phones, leading to a diverse set of resolutions and image qualities. The variability in image size is due to the different camera resolutions of the mobile devices used during data acquisition. 

#### Dataset Structure

The dataset is categorized based on the type of disease affecting the almond trees. Two primary categories of diseases were identified:

1. **Fungal Diseases (MFx)** : Diseases caused by fungal infections.
2. **Insect-related Diseases (Rx)** : Diseases caused by insect infestations.

The dataset is further classified into the following specific disease classes:

* **MF1:** Light green spots on the leaves, turning brown in more advanced stages. 70 Samples availables
* **MF3:** Leaves appear drier, with a rust-like texture and brown spots. 22 Samples availables
* **MF4:** Leaves have a pale coloration and are dry, affecting an entire branch. 10 Samples availables
* **R1:** White spots concentrated in specific areas of the leaf, with insects or larvae visible on the underside. 23 Samples availables
* **R3:** Small insects on the leaves, white specks, and pale-colored leaves. 47 Samples availables
* **R5:** Tiny black and white spots on the leaves. Leaves are significantly dry in advanced stages. 49 Samples availables
* **R6:** Leaves with black spots and signs of insect bites across the surface. 4 Samples availables

For each class, the dataset includes images that capture the progression of the issue across different stages, providing valuable insights into disease development and impact.

#### Dataset Challenges

1. **Class Imbalance**: One of the significant challenges encountered while working with this dataset is the class imbalance. Certain classes have a limited number of images, which poses difficulties in developing a robust and reliable classification system.
2. **Resolution Variability** : The varying resolutions of the images due to differences in mobile phone cameras introduce an additional layer of complexity in preprocessing and standardizing the dataset for model training.
3. **Diaseases mix:** We detect that some samples contains a mix of different diseases and this could be confusing for the model.

Despite these challenges, the dataset represents a valuable resource for understanding and addressing almond tree diseases. Future work could include augmenting the dataset to balance the class distribution and standardizing image resolutions to improve the robustness of classification models.

## **DATASET PREPROCESSING**

Using the Albumentations library, we implemented a pipeline to augment and diversify the sample dataset by applying various real-world variations. This process not only expanded the dataset but also made it more representative of real-life scenarios. Additionally, we tailored the level of augmentation for each class to achieve better balance among the classes.

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

## **CLASSIFICATION MODEL RESULTS**

**Insights and results from the classification model.**

#### NEURONAL NETWORK

The first try of classification model is composed by a simple neuronal network. The results were not so good:

|                        | precision | recall | f1-score | support |
| :--------------------: | :-------: | :----: | :------: | :-----: |
|   **Healthy**   |   0.12   |  0.20  |   0.15   |   40   |
|     **MF1**     |   0.14   |  0.52  |   0.23   |   42   |
|     **MF3**     |   0.06   |  0.16  |   0.09   |   44   |
|     **MF4**     |   0.00   |  0.00  |   0.00   |   40   |
|      **R1**      |   0.00   |  0.00  |   0.00   |   41   |
|      **R3**      |   0.00   |  0.00  |   0.00   |   47   |
|      **R4**      |   0.00   |  0.00  |   0.00   |   49   |
|      **R6**      |   0.12   |  0.05  |   0.07   |   40   |
|                        |          |        |          |        |
|   **accuracy**   |          |        |   0.11   |   343   |
|  **macro avg**  |   0.06   |  0.12  |   0.07   |   343   |
| **weighted avg** |   0.05   |  0.11  |   0.07   |   343   |

As we can see in the classification report, the results are quite bad, with an accuracy of 0.11. Because of this, we will try to find a solution to the problem by finetuning on previously trained models.

#### ResNet

The next network model we have tested is a ResNet. The results were as follows:

|                        | precision | recall | f1-score | support |
| :--------------------: | :-------: | :----: | :------: | :-----: |
|   **Healthy**   |   0.15   |  0.93  |   0.25   |   40   |
|     **MF1**     |   0.00   |  0.00  |   0.00   |   42   |
|     **MF3**     |   1.00   |  0.41  |   0.58   |   44   |
|     **MF4**     |   0.25   |  0.33  |   0.28   |   40   |
|      **R1**      |   0.14   |  0.02  |   0.04   |   41   |
|      **R3**      |   0.00   |  0.00  |   0.00   |   47   |
|      **R4**      |   0.00   |  0.00  |   0.00   |   49   |
|      **R6**      |   0.50   |  0.03  |   0.05   |   40   |
|                        |          |        |          |        |
|   **accuracy**   |          |        |   0.20   |   343   |
|  **macro avg**  |   0.25   |  0.21  |   0.15   |   343   |
| **weighted avg** |   0.25   |  0.20  |   0.15   |   343   |

As we can see, the result with respect to the first network improves, with an accuracy of 0.20. This result is not sufficient and there is still room for improvement.

#### ConvNext

We have also tested the ConvNextLarge. Some extra layers were added to it to further adjust it to the task we want to solve. Specifically, the following layers were added:

1. **GlobalAveragePooling2D()**: helps to reduce the number of parameters and to avoid over-adjustment.
2. **Dense(256,activation='relu'):** the ReLU activation function introduces non-linearities and helps the model to learn more complex patterns.
3. **Dropout(0.5):** randomly deactivate 50% of the neurons in the dense layer during training, to prevent overadjustment of dependence on any particular neuron.
4. **Dense(NUM_CLASSES,activation='softmax'):** prepare the output according to the number of classes we have, namely 8.

This model gave us the following results:

|                        | precision | recall | f1-score | support |
| :--------------------: | :-------: | :----: | :------: | :-----: |
|   **Healthy**   |   0.95   |  0.93  |   0.94   |   40   |
|     **MF1**     |   0.68   |  0.67  |   0.67   |   42   |
|     **MF3**     |   0.77   |  0.75  |   0.76   |   44   |
|     **MF4**     |   0.80   |  0.90  |   0.85   |   40   |
|      **R1**      |   0.92   |  0.85  |   0.89   |   41   |
|      **R3**      |   0.76   |  0.87  |   0.81   |   47   |
|      **R4**      |   0.87   |  0.84  |   0.85   |   49   |
|      **R6**      |   0.97   |  0.88  |   0.92   |   40   |
|                        |          |        |          |        |
|   **accuracy**   |          |        |   0.83   |   343   |
|  **macro avg**  |   0.84   |  0.83  |   0.84   |   343   |
| **weighted avg** |   0.84   |  0.83  |   0.83   |   343   |

As we can see, results have improved considerably, with an accuracy of 0.83. We are still testing some models in order to get better results.

#### Inception

The next model we are going to test is InceptionV3, with the addition of the same extra layers we added to the previous model. This model gave us the following results:

|                        | precision | recall | f1-score | support |
| :--------------------: | :-------: | :----: | :------: | :-----: |
|   **Healthy**   |   1.00   |  0.95  |   0.97   |   40   |
|     **MF1**     |   0.97   |  0.76  |   0.85   |   42   |
|     **MF3**     |   1.00   |  0.98  |   0.99   |   44   |
|     **MF4**     |   0.83   |  1.00  |   0.91   |   40   |
|      **R1**      |   0.85   |  0.95  |   0.90   |   41   |
|      **R3**      |   0.84   |  0.87  |   0.85   |   47   |
|      **R4**      |   0.81   |  0.78  |   0.79   |   49   |
|      **R6**      |   1.00   |  0.97  |   0.99   |   40   |
|                        |          |        |          |        |
|   **accuracy**   |          |        |   0.90   |   343   |
|  **macro avg**  |   0.91   |  0.91  |   0.91   |   343   |
| **weighted avg** |   0.91   |  0.90  |   0.90   |   343   |

As we can see, this model gives better results than the ConvNextLarge finetuning, with an accuracy of 0.90.

#### MobileNet

Ultimately, we tested the MobileNetV2 model with the addition of the same extra layers, with the following results:

|                        | precision | recall | f1-score | support |
| :--------------------: | --------- | ------ | -------- | ------- |
|   **Healthy**   | 0.97      | 0.97   | 0.97     | 40      |
|     **MF1**     | 1.00      | 0.98   | 0.99     | 42      |
|     **MF3**     | 0.98      | 1.00   | 0.99     | 44      |
|     **MF4**     | 0.83      | 1.00   | 0.91     | 40      |
|      **R1**      | 0.95      | 0.98   | 0.96     | 41      |
|      **R3**      | 0.98      | 0.91   | 0.95     | 47      |
|      **R4**      | 0.98      | 0.92   | 0.95     | 49      |
|      **R6**      | 1.00      | 0.93   | 0.96     | 40      |
|                        |           |        |          |         |
|   **accuracy**   |           |        | 0.96     | 343     |
|  **macro avg**  | 0.96      | 0.96   | 0.96     | 343     |
| **weighted avg** | 0.96      | 0.96   | 0.96     | 343     |

With MobileNet, we see that the results we had so far with Inception have been improved, bringing the accuracy up to 0.96.

## CONCLUSIONS AND FUTURE WORK

In this work, we developed an augmentation script and tested various architectures to identify the best solution for our primary objective. The augmentation process played a crucial role in generating sufficient samples and achieving a balanced distribution among the different classes.

After extensive testing, the best-performing architecture was MobileNet, achieving a precision of 0.96 . This model not only provided excellent results but also proved to be highly efficient for deployment in a mobile application, which is the core purpose of our project. Its lightweight architecture ensures that it operates seamlessly on resource-constrained devices, making it an ideal choice for mobile integration.

As future work, it would be valuable to develop methods for determining the current state and progression of the disease in real time. This would enable farmers to take timely and targeted actions to mitigate its impact, reducing the negative effects on crop yield and quality.
