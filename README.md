# 🚨 An Explainable Approach for Network Intrusion Detection Using Machine Learning and Deep Neural Network

This repository contains the complete implementation, documentation, and results of our senior design project for CSE499. Our project focuses on developing a machine learning and deep neural network-based Intrusion Detection System (IDS) that is not only accurate but also explainable.

## 📌 Project Overview
With the growing rate of cyberattacks on web applications, a robust and interpretable IDS is crucial. This project investigates the impact of synthetic data on IDS performance and introduces a framework that combines multiple machine learning models with Explainable AI (XAI) techniques for improved accuracy and transparency.

### Key Features
- Utilizes **[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)**, a comprehensive intrusion detection dataset.
- Applies **SMOTE** for dataset balancing (small, medium, and large variants).
- Employs **XGBoost** for feature selection.
- Implements multiple ML algorithms: Decision Tree, Random Forest, Naïve Bayes, AdaBoost, MLP.
- Constructs a **weighted ensemble model** for robust classification.
- Integrates **SHAP** and **LIME** for explainable predictions.
- Includes **real-world testing** using captured traffic via Wireshark.

### System Design: 

<img src="https://github.com/sajan-sarker/web-attack-detection/blob/main/figures/System%20Design.png?raw=true" alt="System Design" width="1000"/>

### Dataset Statistics:
<img src="https://github.com/sajan-sarker/web-attack-detection/blob/main/figures/dataset.jpg?raw=true" alt="Model Performances" width="800"/>

## 🛠️ Technologies Used

| Tool               | Purpose                                                                       |
|--------------------|-------------------------------------------------------------------------------|
| Python3            | Core programming language                                                     |
| Scikit-learn       | ML model development & preprocessing                                          |
| PyTorch            | Deep neural network (MLP)                                                     |
| Google Colab       | Cloud-based GPU training                                                      |
| Flask              | Lightweight web deployment for real world data testing and capstone showcase  |
| CICFlowMeter       | Flow-based feature extraction                                                 |
| Wireshark          | Network packet capturing                                                      |

## 📊 Results Summary
<img src="https://github.com/sajan-sarker/web-attack-detection/blob/main/figures/avgcmp.png?raw=true" alt="Model Performances" width="800"/>
<img src="https://github.com/sajan-sarker/web-attack-detection/blob/main/figures/avgensm.png?raw=true" alt="Model Performances" width="800"/>
<img src="https://github.com/sajan-sarker/web-attack-detection/blob/main/figures/rmulti.png?raw=true" alt="Model Performances" width="800"/>

| Dataset Size | Accuracy | F1-Score | FPR    |
|--------------|----------|----------|--------|
| Small        | 97.29%   | 97.29%   | 0.0068 |
| Medium       | **97.30%**   | **97.26%**   | **0.0048** |
| Large        | 96.91%   | 96.88%   | 0.0088 |

## 🔍 Explainable AI Integration

- **SHAP**: Provides global + local interpretation of feature contributions.
- **LIME**: Breaks down individual predictions for transparency.
- Examples and visualizations are available in the `/explainability/` folder.

## 🧪 Real-World Testing

- Network traffic captured using **Wireshark**.
- Converted to flows using **CICFlowMeter**.

## 📈 Future Improvements

- Real-time intrusion detection system (live traffic monitoring).
- Inclusion of more diverse and recent attack types.
- Shift from synthetic to real-world minority data for better learning.
- Integration with active threat response systems.

## 🤝 Contributors

- **Sajan Kumer Sarker – 2111131642**
- **Mahzabeen Rahman Meem – 2021300642**
- **Arifa Akbar Srity – 2021149642**

Faculty Supervisor: **Dr. Mohammad Ashrafuzzaman Khan**  
Department of ECE, North South University
