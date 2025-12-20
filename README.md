

# Advanced Machine Learning
# 🩺 Breast Cancer Detection & Classification 

## 📌 Project Overview

Breast cancer is one of the most common and life-threatening cancers worldwide. Early detection and accurate classification are critical to improving patient outcomes and supporting clinical decision-making.

This project focuses on building **Machine Learning and Deep Learning models** for **breast cancer detection, classification, and risk analysis**, using **three complementary datasets**:

* Clinical and morphological features,
* Anthropometric and blood analysis data,
* Histopathological image patches.

The project is conducted following the **CRISP-DM methodology** and is inspired by a scientific article. Our objective is not only to reproduce the results presented in the article, but also to **achieve better performance**, particularly in the **DSO1 phase**.

---

## 📊 Datasets Description

### 1️⃣ Base Dataset – Clinical & Morphological Features

This dataset contains quantitative features computed from digitized images of breast mass biopsies.

**Target variable**

* `diagnosis`:

  * **M** → Malignant
  * **B** → Benign

**Example features**

* Radius, texture, perimeter, area
* Smoothness, compactness, concavity
* Symmetry and fractal dimension
  (all provided as mean, standard error, and worst values)

**Sample extract**

```
id    diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  ...
842302  M        17.99        10.38          122.8           1001       ...
842517  M        20.57        17.77          132.9           1326       ...
```

📌 **Purpose**:
Used to **classify tumors as benign or malignant**.

---

### 2️⃣ Additional Dataset – Anthropometric & Blood Analysis Data

This dataset is composed of **10 quantitative predictors** collected from routine medical exams.

**Features**

* Age (years)
* BMI (kg/m²)
* Glucose (mg/dL)
* Insulin (µU/mL)
* HOMA
* Leptin (ng/mL)
* Adiponectin (µg/mL)
* Resistin (ng/mL)
* MCP-1 (pg/dL)

**Target variable**

* `Classification`:

  * **1** → Healthy controls
  * **2** → Patients with breast cancer

**Sample extract**

```
Age  BMI        Glucose  Insulin  HOMA  Leptin  Adiponectin  Resistin  MCP.1  Classification
54   35.20      103      5.642    1.37  65.66   9.73         31.17     197.66 1
52   22.97      132      6.054    1.14  47.54   3.62         23.03     423.36 2
```

📌 **Purpose**:
Used to **identify early risk factors** and support **preventive recommendations** for healthy patients.

---

### 3️⃣ Image Dataset – Histopathological Breast Cancer Images

This dataset consists of **histopathology image patches** extracted from whole slide images.

**Dataset characteristics**

* 162 whole mount slide images scanned at 40×
* 277,524 image patches (50×50 pixels):

  * 198,738 IDC negative
  * 78,786 IDC positive

**File naming format**

```
u_xX_yY_classC.png
Example: 10253_idx5_x1351_y1101_class0.png
```

Where:

* `u` → Patient ID
* `X`, `Y` → Patch coordinates
* `C` → Class label:

  * **0** → Non-IDC
  * **1** → IDC (Invasive Ductal Carcinoma)

📌 **Purpose**:
Used to **detect tumor presence from medical images** using Deep Learning models (CNNs).

---

## 🎯 Business Objectives

* **Confirm the presence of a breast tumor**
  Provide reliable decision support to physicians when assessing suspicious cases.

* **Characterize the tumor to guide clinical decisions**
  Clearly identify whether a tumor is benign or malignant.

* **Detect early risk factors in healthy patients**
  Identify patients at risk and recommend appropriate preventive actions.

---

## 🤖 Data Science Objectives

* **Tumor type classification**
  Build predictive models to distinguish between benign and malignant tumors using clinical features.
  *(Article dataset)*

* **Tumor presence detection**
  Develop image-based models capable of detecting cancerous regions in histopathological images.
  *(Image dataset)*

* **Risk factor identification**
  Analyze anthropometric and blood biomarkers to detect early breast cancer risk in healthy individuals.
  *(Clinic dataset)*

---

## 🔄 Methodology: CRISP-DM

This project strictly follows the **CRISP-DM** framework:

1. **Business Understanding**
   Define medical and clinical objectives.

2. **Data Understanding**
   Explore datasets, analyze distributions, correlations, and data quality.

3. **Data Preparation**
   Data cleaning, normalization, feature selection, augmentation (for images).

4. **Modeling**
   Apply Machine Learning and Deep Learning algorithms:

   * Classical ML (Logistic Regression, SVM, Random Forest, etc.)
   * Deep Learning (CNNs for image data)

5. **Evaluation**
   Performance analysis using:

   * Accuracy
   * Precision, Recall, F1-score
   * ROC-AUC
   * Confusion Matrix

6. **Deployment**
   The final stage of the project focuses on the deployment of the trained models through a web-based application.
   The developed web platform allows users (medical staff) to:
   * Upload patient clinical data or medical images,
   * Obtain real-time predictions regarding:
       - Tumor presence,
       - Tumor type (benign or malignant),
       - Potential breast cancer risk indicators.
   Key deployment aspects include:
   * Model persistence: trained models are saved and loaded to ensure consistency and reproducibility.
   * Reproducibility: fixed preprocessing pipelines and versioned models guarantee stable predictions.
   * Decision-support integration: the web application acts as a decision-support system, providing interpretable outputs to assist medical professionals.
   * Scalability and accessibility: a web-based interface ensures easy access without requiring advanced technical knowledge.
   => This deployment approach bridges the gap between data science and real-world clinical usage, transforming predictive models into a practical and usable tool.

---

## 🧪 Expected Outcomes

* High-performance models for breast cancer detection and classification.
* Improved results compared to the reference scientific article.
* Robust and interpretable models for clinical support.
* Actionable insights for early prevention strategies.

---

## ⚠️ Disclaimer

This project is intended **for academic and research purposes only**.
It **does not replace professional medical diagnosis** or clinical expertise.

---

## 👤 Authors

  - Ahmed Rayen Aloui
  - Mohamed Yassine Janfaoui
  - Ahmed Tounsi
  - Farah Derbel
  - Farah Boubaker
  - Mahdi Saoudi
