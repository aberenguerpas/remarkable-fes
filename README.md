# Documentation


**Disease Classification Task:**

**DATASET DESCRIPTION**

source, features, challenges, and size

**DATASET PREPROCESSING**

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
* 

Clean, preprocess, and organize it to ensure it’s ready for further analysis and modeling.

Carefully document your process, including challenges faced and how they were addressed

**CLASSIFICATION MODEL RESULTS**

Utilize the prepared dataset to develop a disease classification model.

Experiment with different classification algorithms, evaluate their performance, and document the results.

**Insights and results from the classification model.**

Results for the first version of the Neural Network:

Loss: 0.903, Accuracy: 0.667
