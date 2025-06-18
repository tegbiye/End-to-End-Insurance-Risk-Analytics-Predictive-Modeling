# End-to-End Insurance Risk Analytics & Predictive Modeling

### Dive into real insurance data to uncover low-risk segments and build smart models that optimize premiums.

---
## Project Structure

<pre>
|-----.env/
|---- .github/
|     |--- workflows
|     |    |--- unittests.yml
|---- data/
|     |---- cleaned_insurance_data.csv (Cleaned)
|     |---- MachineLearningRating_v3.txt (Row)
|---- notebooks/
|     |--- README.md
|     |--- cleaning_insurance_risk_analytics.ipynb
|     |--- eda_insurance_risk_analytics.ipynb
|---- scripts/
|     |--- __init__.py
|     |--- load_data.py
|     |--- monthly_trend.py
|---- tests/
|     |--- __init__.py
|     |--- test_1.py
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

## Getting Started
1. Clone the Repository
   ``` git clone https://github.com/tegbiye/End-to-End-Insurance-Risk-Analytics-Predictive-Modeling.git
   
   ```
   ```
    cd End-to-End-Insurance-Risk-Analytics-Predictive-Modeling
   ```
2. Create environment
   ```
   python -m venv .venv

   ```Windows
   .venv\Scripts\activate
   ```
   ```Linux/Mac
   source .venv\bin\activate
   ```
3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

📜 License
This project is licensed under the MIT License.
Feel free to use, modify, and distribute with proper attribution.