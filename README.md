
#  Popups as HCI Novelties: A Machine Learning Approach to Classifying Popup Designs

This project uses machine learning to classify **popup designs** based on various attributes such as type, size, trigger mechanism, position, design style, and content type. It is built around a custom dataset of 3000+ popup samples and provides an interactive UI using Streamlit to test the model predictions.

## 📊 Project Objective

To analyze and classify different types of popups as part of Human-Computer Interaction (HCI) studies, using:
- Supervised ML models
- Label encoding of categorical features
- Deployment-ready user interaction through a web interface

---

## 🚀 Features

- **Multi-model comparison**: Logistic Regression, KNN, SVM, Decision Trees, Random Forest, MLP
- **Confusion matrix and classification report outputs**
- **Interactive Streamlit interface** for real-time popup category prediction
- **Sidebar help** to guide users on input values
- **Model saved using `joblib`** for reproducibility

---

## 🛠️ Tech Stack

| Layer            | Tech/Tool                         |
|------------------|-----------------------------------|
| Data Processing  | `pandas`, `numpy`, `LabelEncoder` |
| ML Models        | `scikit-learn`                    |
| Visualization    | `matplotlib`, `seaborn`           |
| UI/Deployment    | `Streamlit`                       |
| Model Persistence| `joblib`                          |

---

## 📁 Folder Structure

```
.
├── app.py                          # Streamlit App
├── popup.py                     # Model training & evaluation
├── popup_dataset_3000.csv          # Main dataset
├── README.md                       # Project documentation
```

---

## 📌 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sahajbeersingh/Popups-as-HCI-Novelities.git
cd Popups-as-HCI-Novelities
```

### 2. Install Requirements


```bash
pip install pandas scikit-learn seaborn matplotlib streamlit joblib
```

### 3. Train Model

```bash
python popup.py
```
this will create
```bash
target_encoder.joblib
popup_category_rf_model.joblib
features_encoders.joblib
```

### 4. Run the App

```bash
streamlit run app.py
```

Then open your browser at the address shown in the terminal (usually http://localhost:8501).

---

## 🧪 Models Compared

| Model              | Accuracy Achieved |
|--------------------|------------------|
| Logistic Regression| ~74.8888%             |
| K-Nearest Neighbors| ~93.1111%             |
| SVM (RBF Kernel)   | ~90.5555%             |
| Decision Tree      | ~87.2222%             |
| Random Forest      | **~93.3333%**         |
| MLP Classifier     | ~91.7778%             |

*Random Forest was chosen for final deployment.*

---


## 🙋‍♂️ Feedback

If you have any suggestions, feel free to open an [Issue](https://github.com/sahajbeersingh/Popups-as-HCI-Novelities/issues) or a Pull Request.
