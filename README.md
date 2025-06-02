# White Wine Quality Prediction using Classification

This project aims to predict white wine quality using several machine learning classification algorithms. It is part of the 4th Semester Final Exam for the *Machine Learning for Intelligent System* course. The main models explored are Random Forest and Logistic Regression.

## Developers
- Lidya Imelda
- Elsen Wuiri Chuanda
- Luke Maximus Kawilarang
- Ignatius Valen Arwalembun

## Supervising Lecturers
- Eko Wahyu Prasetyo, S.T., M.Eng
- Team Tiket.com

## Dataset
The dataset used is the white wine quality dataset, which consists of 11 physicochemical features (such as *fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol*) and a target variable representing wine quality (scale 3-9). For classification purposes, this target variable was then binarized into low quality (value <= 5) and high quality (value >= 6).

The data is accessed via Google Drive and has been cleaned of null and duplicate values.

## Project Structure
- `UAS_MLIS_Kelompok_4_Wine_Quality.ipynb`: The main Jupyter Notebook containing the entire process of data analysis, preprocessing, modeling, tuning, and evaluation.

## Methodology
The process undertaken in this notebook includes:
1.  **Data Loading**: Reading the CSV dataset from Google Drive.
2.  **Exploratory Data Analysis (EDA)**:
    * Understanding data characteristics (info, statistical description).
    * Converting data types for numerical columns.
    * Examining the target class distribution.
    * Checking for null values and duplicates.
    * Outlier detection using boxplots.
    * Visualizing feature correlations using a heatmap.
    * Visualizing the distribution of each numerical feature using histograms.
3.  **Data Preprocessing**:
    * **Data Scaling**: Numerical features were scaled using `StandardScaler`.
    * **Data Splitting**: Data was split into training (80%) and testing (20%) sets with stratification on the target variable.
    * **Handling Class Imbalance**: Oversampling of the minority class using SMOTE (Synthetic Minority Over-sampling Technique) on the training data.
4.  **Baseline Modeling**:
    * Training a Random Forest Classifier model with `GridSearchCV` for initial parameter search.
    * Training a Logistic Regression model with `GridSearchCV` for initial parameter search.
5.  **Hyperparameter Tuning**:
    * Optimizing Random Forest model parameters using `RandomizedSearchCV` with a broader parameter space.
6.  **Model Evaluation**:
    * Comparing model performance based on accuracy and classification report (precision, recall, F1-score).
    * Visualizing Confusion Matrices for both models (Random Forest and Logistic Regression).
    * Analyzing Feature Importance from the best Random Forest model.
7.  **Conclusion**: Summarizing model performance results, optimization techniques, feature insights, and recommendations for the best model.

## Models Used
- Random Forest Classifier
- Logistic Regression

## Results
- The **Random Forest model (after tuning)** showed the best performance with an accuracy of approximately **84.29%** and a macro F1-score of **0.8187** on the test data.
- The use of SMOTE helped in handling class imbalance in the training data, improving the model's ability to recognize minority classes.
- Features such as *alcohol, volatile acidity, sulphates,* and *citric acid* were identified as some of the important features influencing wine quality prediction.

## Dependencies
This project uses the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`
- `seaborn`

To run the notebook, ensure the `imbalanced-learn` library is installed (`pip install imbalanced-learn`).

## How to Run
1.  Ensure all the above dependencies are installed in your Python environment.
2.  Open and run the `UAS_MLIS_Kelompok_4_Wine_Quality.ipynb` notebook using Jupyter Notebook, JupyterLab, or Google Colab.
3.  The dataset will be automatically downloaded from the Google Drive link provided in the notebook.

## Project Conclusion
Based on the analysis conducted, the optimized Random Forest model is the most recommended for predicting white wine quality. Preprocessing techniques such as feature scaling and handling imbalanced data with SMOTE proved crucial in enhancing model performance.
