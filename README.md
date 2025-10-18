# Diabetes Prediction using Support Vector Machine (SVM)

This project aims to predict whether a patient has diabetes based on several medical diagnostic measurements. A Support Vector Machine (SVM) classifier is trained using the "Pima Indians Diabetes Database".

## Dataset

The project utilizes the `diabetes.csv` dataset. This dataset contains information on 768 female patients of Pima Indian heritage.

* **Dataset Shape**: The dataset consists of 768 rows and 9 columns.
* **Features**: The 8 independent variables (features) used for prediction are:
    * `Pregnancies`
    * `Glucose`
    * `BloodPressure`
    * `SkinThickness`
    * `Insulin`
    * `BMI`
    * `DiabetesPedigreeFunction`
    * `Age`
* **Target Variable**: The dependent variable is `Outcome`, which is binary:
    * `0`: Represents a non-diabetic person.
    * `1`: Represents a diabetic person.

In this dataset, there are 500 instances of non-diabetic cases and 268 instances of diabetic cases.

## Project Workflow

The machine learning pipeline for this project follows these steps:

1.  **Data Collection and Analysis**: The `diabetes.csv` dataset is loaded using the Pandas library. Initial exploratory data analysis is performed to understand the data's structure, statistical properties, and distribution.

2.  **Data Preprocessing**:
    * The features (X) and the target variable (Y) are separated from the main dataset.
    * **Data Standardization**: The feature data is standardized using Scikit-learn's `StandardScaler`. This step is essential because the features are on different scales, and standardization brings them to a common scale, which helps the SVM model perform better.

3.  **Train-Test Split**:
    * The dataset is divided into a training set and a testing set using an 80-20 split (80% for training, 20% for testing).
    * The `stratify=Y` parameter is used to ensure that the proportion of diabetic and non-diabetic patients is consistent in both the training and test sets. A `random_state` is set for reproducibility.

4.  **Model Training**:
    * A Support Vector Machine (SVM) classifier with a linear kernel (`svm.SVC(kernel='linear')`) is instantiated.
    * The classifier is trained on the standardized training data (`X_train`, `Y_train`).

5.  **Model Evaluation**:
    * The model's performance is evaluated using the accuracy score metric.
    * **Training Data Accuracy**: $78.66\%$
    * **Test Data Accuracy**: $77.27\%$
    The close accuracy scores between the training and test data indicate that the model generalizes well and is not significantly overfitted.

6.  **Predictive System**:
    * A system is implemented to make predictions on new, unseen data. For a given set of input features, the system standardizes the data using the previously fitted scaler and then uses the trained SVM model to predict the outcome.

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Colab Notebook

## How to Run the Project

1.  Ensure you have Python and Jupyter Notebook installed on your system.
2.  Install the required Python libraries:
    ```bash
    pip install numpy pandas scikit-learn
    ```
3.  Download the `Diabetes_Prediction.ipynb` notebook file and the `diabetes.csv` dataset.
4.  Place both files in the same directory.
5.  Launch Jupyter Notebook and open the `Diabetes_Prediction.ipynb` file.
6.  Execute the cells in the notebook sequentially to see the workflow from data loading to prediction.
