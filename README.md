# End-to-End Insurance Risk Analytics & Predictive Modeling

### Dive into real insurance data to uncover low-risk segments and build smart models that optimize premiums.

---
## Project Structure

<pre>
|-----.dvc/
|     |--- config
|     |--- .gitignore
|-----.env/
|---- .github/
|     |--- workflows
|     |    |--- unittests.yml
|---- data/
|     |---- cleaned_insurance_data.csv.dvc (Cleaned)
|---- notebooks/
|     |--- README.md
|     |--- cleaning_insurance_risk_analytics.ipynb
|     |--- eda_insurance_risk_analytics.ipynb
|     |--- statistical_hypotesis_Testing.ipynb
|     |--- Build_model_pipeline.ipynb (Model building pipeline)
|---- scripts/
|     |--- __init__.py
|     |--- load_data.py
|     |--- monthly_trend.py
|     |--- statistical_hyphothesis.py (function for hyphotessis testing)
|---- tests/
|     |--- __init__.py
|     |--- test_1.py
|-----.dvcignore
|---- .gitignore
|---- requirements.txt (Dependencies)
|---- LICENSE
|____ README.md
</pre>
## Task One
   1. Data summerization
      Cleaned the data with text and saved to cleaned csv
      loaded cleaned csv and done descriptive analysis
   2. Data Quality Assesment
      Checked for the missing values and found some significant amount of missing values fixed
   3. Made Visualization
## Task 2
   1. After checking out to the task-2 branch Installed and Intialized dvc
      ```
      pip install dvc
      ```
      ```
      dvc init
      ```
      ```
      git add .dvc .dvcignore
      ```
   2. Configured local and remote using dvc and tracked the cleaned data
       N.B. Make the path your own (/path/to/local/dvc-storage other than .dvc folder)
       I created folder for the local storage

       ```
       mkdir /path/to/local/dvc-storage
       ```
       Then I created the storage as a DVC Remote
       ```
       dvc remote add -d localstorage /path/to/local/dvc-storage
       ```
       Just add cleaned_data to the dvc track

       ```
       dvc add data/cleaned_insurance_data.csv
       ```   
       ```
       Commited different version of data
       ```
       dvc add data/cleaned_insurance_data_sample.csv
       ```
       ```
       git add data/cleaned_insurance_data_sample.csv.dvc
       git commit -m "Added another version data"
       ```
       Push Data to Remote
       ```
       dvc push
       ```
## Task 3
   1. Made A/B-Hypothesis Testing using chi-square, ANOVa where threshold p = 0.05
      Based on the p-value calculated the hypothesis is rejected or failed

## Task 4: Building and Evaluating Predictive model for Dynamic, Risk-Based Pricing System
1. Data Preparation (Conceptual & Code Structure)
   Handling Missing Data
   Feature Engineering
   Encoding Categorical Data
   Train-Test Split

2. Modeling Goals & Evaluation Metrics
   Claim Severity Prediction (Regression)
   Probability of Claim Occurrence (Classification)
   Premium Optimization (Conceptual - combining the above)

3. Statistical Modeling Techniques
   Linear Regression
   Decision Trees (Implicitly covered by Random Forests/XGBoost, but can be a baseline)
   Random Forests
   Gradient Boosting Machines (XGBoost)

4. Model Building 

5. Model Evaluation (Code Structure for Each Model and Goal)
   RMSE & R-squared for Regression
   Accuracy, Precision, Recall, F1-score for Classification

6. Feature Importance Analysis (Code Structure)
   Built-in feature importance
   SHAP

## Getting Started
1. Clone the Repository
   ``` 
   git clone https://github.com/tegbiye/End-to-End-Insurance-Risk-Analytics-Predictive-Modeling.git
   
   ```
   ```
    cd End-to-End-Insurance-Risk-Analytics-Predictive-Modeling
   ```
2. Create environment
   ```
   python -m venv .venv
   
   ```
   Windows
   ```
   .venv\Scripts\activate
   ```
   Linux/Mac
   ```
   source .venv\bin\activate
   ```
3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

📜 License
This project is licensed under the MIT License.
Feel free to use, modify, and distribute with proper attribution.