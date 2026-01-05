# Rice Leaf Disease Detection using Deep Learning

##  Project Overview

Rice leaf diseases significantly affect crop yield and productivity. This project focuses on building and evaluating deep learning models to classify three major rice leaf diseases using image data:

* **Leaf Blast**
* **Bacterial Blight**
* **Brown Spot**

The project analyzes multiple modeling approaches and selects the most suitable model for real-world deployment.

---

##  Objectives

* Perform exploratory data analysis (EDA) on the rice leaf image dataset.
* Build and evaluate deep learning models for disease classification.
* Analyze techniques such as data augmentation, regularization, and transfer learning.
* Compare model performances and select the best model for production use.
* Evaluate the final model using predictions on unseen rice leaf images.

---

##  Dataset Information

* **Type:** Image Classification Dataset
* **Total Images:** 120
* **Classes:** 3 (Leaf Blast, Bacterial Blight, Brown Spot)
* **Image Format:** RGB images

---

##  Exploratory Data Analysis (EDA)

* Analyzed class distribution and image structure.
* Visualized sample images from each disease category.
* Identified dataset limitations such as small dataset size and potential overfitting risk.

---

##  Models Implemented

### 1️ Baseline CNN

* Custom CNN architecture built from scratch.
* Achieved **73.91% validation accuracy**.
* Showed signs of overfitting due to limited data.

### 2️ CNN with Data Augmentation

* Applied random flipping, rotation, and zooming.
* Improved training stability but validation accuracy remained **73.91%**.

### 3️ CNN with Regularization

* Used Dropout and L2 regularization.
* Resulted in **underfitting** with validation accuracy of **60.87%**.

### 4️ Transfer Learning using MobileNetV2

* Pre-trained MobileNetV2 model used.
* Achieved the **highest validation accuracy of 82.61%**.
* Demonstrated better generalization and faster convergence.

---

##  Model Comparison Summary

| Model                   | Validation Accuracy |
| ----------------------- | ------------------- |
| Baseline CNN            | 73.91%              |
| CNN + Data Augmentation | 73.91%              |
| CNN + Regularization    | 60.87%              |
| **MobileNetV2**         | **82.61%**          |

**Selected Model:** MobileNetV2

---

##  Prediction on New Rice Leaf Images

* Predictions were performed on **three unseen rice leaf images**.
* The MobileNetV2 model correctly classified all test images.
* This confirms the model’s real-world applicability and generalization capability.

---

##  Challenges and Mitigation

* **Limited Dataset Size:** Addressed using data augmentation and transfer learning.
* **Overfitting in CNN Models:** Controlled using early stopping and advanced models.
* **Visual Similarity Between Diseases:** Mitigated using deep feature extraction from MobileNetV2.

---

##  Final Conclusion

MobileNetV2 outperformed all CNN-based approaches by achieving the highest validation accuracy and better generalization on limited data. Prediction results on unseen images further confirmed its practical effectiveness, making it the most suitable model for rice leaf disease detection.

---

##  Technologies Used

* Python
* TensorFlow / Keras
* NumPy, Matplotlib
* Jupyter Notebook

---

##  Submission Note

All tasks, experiments, evaluations, and conclusions were completed within a **single Jupyter Notebook**, as per institute guidelines.

---

##  Author

**Juju**
Data Science Trainee

---

 *This project demonstrates an end-to-end deep learning workflow for agricultural disease classification using limited image data.*
