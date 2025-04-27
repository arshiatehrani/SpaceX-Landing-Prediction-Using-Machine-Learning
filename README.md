# SpaceX Landing Prediction Using Machine Learning

## Project Title: SpaceX Landing Prediction Using Machine Learning

### Project Description

This repository contains a series of Jupyter notebooks developed for analyzing and predicting SpaceX rocket landings using machine learning models. The project explores various datasets and implements multiple machine learning algorithms to predict whether a rocket landing is successful or not. It involves data preprocessing, exploratory data analysis (EDA), and model development using several classification algorithms, including Logistic Regression, Support Vector Machines (SVM), Decision Trees, and K-Nearest Neighbors (KNN).

The project also explores the use of various techniques like GridSearchCV for hyperparameter tuning to improve model performance. The goal is to assess the prediction accuracy of different models and determine the best-performing model for the given task.

### Project Structure

The repository is structured as follows:

```
SpaceX-Landing-Prediction
│
├── datasets/
│   ├── dataset_part_2.csv
│   └── dataset_part_3.csv
│
├── notebooks/
│   ├── SpaceX_Landing_Prediction.ipynb
│   ├── Data_Preprocessing.ipynb
│   ├── Model_Training_and_Evaluation.ipynb
│   └── Hyperparameter_Tuning.ipynb
│
├── README.md
└── requirements.txt
```

### Files Description

- **datasets/**: This folder contains the raw datasets used for the analysis. These datasets contain information regarding the SpaceX rocket launches, including variables like Class, Launch Date, Rocket Type, and whether the landing was successful.
  
- **notebooks/**: This folder contains the Jupyter notebooks for each stage of the project:
  - `SpaceX_Landing_Prediction.ipynb`: The main notebook, where the project begins with data loading, preprocessing, and model evaluation.
  - `Data_Preprocessing.ipynb`: Contains the steps for cleaning and preparing the data for machine learning.
  - `Model_Training_and_Evaluation.ipynb`: Contains the implementation of various classification algorithms and evaluates their performance.
  - `Hyperparameter_Tuning.ipynb`: Focuses on using GridSearchCV to tune hyperparameters for different machine learning models.
  
- **requirements.txt**: Contains a list of all the Python libraries required to run the project (e.g., Pandas, NumPy, scikit-learn, etc.).

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SpaceX-Landing-Prediction.git
   cd SpaceX-Landing-Prediction
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Data Description

The project uses the following two datasets:
1. **dataset_part_2.csv**: Contains features like rocket type, landing success (Class), launch date, etc.
2. **dataset_part_3.csv**: Contains additional features relevant for training the models.

### Methodology

1. **Data Preprocessing**:
   - Cleaned and transformed the dataset by handling missing values, encoding categorical variables, and scaling numerical features.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized the distribution of various features.
   - Investigated relationships between features using plots like histograms, pair plots, and correlation matrices.

3. **Model Training**:
   - Implemented multiple classification algorithms, including Logistic Regression, SVM, Decision Trees, and KNN, to predict the success of a rocket landing.
   - Split the data into training and testing datasets using `train_test_split`.

4. **Hyperparameter Tuning**:
   - Used `GridSearchCV` to fine-tune the hyperparameters of the models and find the optimal settings for improved performance.

5. **Model Evaluation**:
   - Evaluated model performance using metrics like accuracy, precision, recall, and the confusion matrix.

### Model Performance

After training the models and tuning hyperparameters, the best-performing model based on accuracy and other metrics was selected. You can find the details of the evaluation results in the notebooks.

### How to Use

1. Open the notebook `SpaceX_Landing_Prediction.ipynb` in Jupyter Notebook or JupyterLab to run the code.
2. Ensure you have installed the required libraries mentioned in `requirements.txt`.
3. Follow the instructions in the notebooks to understand the steps of the project, from data preprocessing to model training and evaluation.

### Technologies Used

- Python 3.11
- Jupyter Notebook
- Pandas
- NumPy
- scikit-learn (for machine learning)
- Matplotlib
- Seaborn
- GridSearchCV for hyperparameter tuning

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgements

- The SpaceX datasets were used from publicly available sources on the internet.
- This project was developed as part of my personal learning journey in predictive analysis and machine learning.
