# GlucoSense – Diabetes Prediction System

**A Machine Learning-Based Diabetes Risk Prediction Application**

GlucoSense is a Python-based machine learning project designed to predict the likelihood of diabetes using medical and health-related parameters. It applies data preprocessing, classification algorithms, and model evaluation techniques to analyze patient data and generate predictions.

The project demonstrates how machine learning can be applied to healthcare data analysis through a structured and modular implementation.

## 👨‍💻 Author

**Harsh Saini**  
B.Tech – Computer Science and Engineering (AI & ML)  
Manav Rachna International Institute of Research and Studies

## 🚀 Key Features

- Diabetes risk prediction using machine learning.
- Data preprocessing and missing-value handling.
- Comparison of multiple classification algorithms.
- Model training and performance evaluation.
- Data visualization and analysis.
- User-friendly Streamlit interface.
- Modular and maintainable Python code structure.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Web Interface:** Streamlit
- **Model Persistence:** Joblib

## 📊 Dataset

The project uses the Pima Indians Diabetes Dataset, containing medical attributes such as:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The target variable, `Outcome`, indicates whether diabetes was recorded in the dataset.

## 📁 Project Structure

```text
GlucoSense/
│
├── data/
│
├── models/
│
├── outputs/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   ├── prediction.py
│   └── visualization.py
│
├── main.py
├── app.py
├── config.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd GlucoSense
```

**2. Create a virtual environment**

```bash
py -3 -m venv .venv
```

**3. Install dependencies**

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## ▶️ Usage

**Train the machine learning models**

```bash
.venv\Scripts\python.exe main.py train
```

**Launch the Streamlit application**

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

**Generate a prediction using the command line**

```bash
.venv\Scripts\python.exe main.py predict --pregnancies 2 --glucose 130 --blood-pressure 70 --skin-thickness 25 --insulin 90 --bmi 28.5 --pedigree 0.45 --age 35
```

## 📈 Model Evaluation

The project evaluates classification models using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

Multiple machine learning models are compared using cross-validation to support model selection.

## 🔮 Future Improvements

- Integration with larger and more diverse datasets.
- Advanced machine learning and ensemble techniques.
- Improved prediction explainability.
- Enhanced dashboard and visualization features.
- Deployment as a cloud-based application.

## ⚠️ Disclaimer

GlucoSense is an educational machine learning project. Its predictions are based on the dataset used for training and are not clinically validated. It must not be used as a substitute for professional medical advice, diagnosis, or treatment.

## 📄 License

This project is intended for educational and academic purposes. A formal open-source license can be added if the project is distributed under specific licensing terms.

---

**Developed by Harsh Saini**  
*Machine Learning | Python | Healthcare Analytics*
