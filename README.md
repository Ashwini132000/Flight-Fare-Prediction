# 🛫 Flight Fare Prediction  
## 🧾 Problem Statement  
Airfare pricing is highly dynamic and influenced by various factors such as travel routes, airline carriers, number of stops, and added services like meals. Accurately predicting flight fares is a major challenge in the travel industry due to these complex dependencies. This project focuses on analyzing a real-world flight booking dataset sourced from the EaseMyTrip platform. By exploring key features such as **source**, **destination**, **route**, **airline**, and **meal inclusion**, we aim to gain meaningful insights and develop predictive models that estimate flight prices more effectively.  

![41822](https://github.com/user-attachments/assets/2b19d559-e5d6-42ca-8cba-c0b29f16e094)


---  

## 🎯 Objective  
To develop a **Machine Learning model** that:  
- 🔍 **Predicts the flight fare** based on customer booking information  
- ✈️ Uses key travel features including:  
  - 🛫 Source  
  - 🛬 Destination  
  - 🛣️ Route  
  - ⛔ Total Stops  
  - 🛩️ Airline  
  - 🍽️ Meal Inclusion  
- 📈 Applies regression algorithms such as:
  - Linear Regression  
  - Random Forest Regressor  
  - Other ML regression models  
- 📊 Performs Exploratory Data Analysis and Hypothesis Testing to understand pricing patterns  
- 💡 Helps users and businesses make informed decisions by understanding fare drivers and optimizing travel planning

---  

## 🗂️ Dataset Description  

| Column Name           | Description                                                                     |
| --------------------- | ------------------------------------------------------------------------------- |
| **Airline**           | Name of the airline (e.g., IndiGo, Air India, Jet Airways)                      |
| **Date\_of\_Journey** | Journey date in `dd/mm/yyyy` format                                             |
| **Source**            | Departure city                                                                  |
| **Destination**       | Arrival city                                                                    |
| **Route**             | Sequence of flight routes taken                                                 |
| **Dep\_Time**         | Departure time of the flight                                                    |
| **Arrival\_Time**     | Arrival time of the flight                                                      |
| **Duration**          | Total time taken by the flight                                                  |
| **Total\_Stops**      | Number of stops between source and destination (e.g., non-stop, 1 stop)         |
| **Additional\_Info**  | Extra information about the flight (e.g., No info, In-flight meal not included) |
| **Price**             | Target variable — Final ticket price (in INR)                                   |

---  

## 🔧 Tools and Technologies  

| Category             | Tools/Technologies Used                                            |
| -------------------- | ------------------------------------------------------------------ |
| **Programming**      | Python                                                             |
| **Libraries**        | pandas, numpy, matplotlib, seaborn, scikit-learn, catboost, joblib |
| **Machine Learning** | CatBoost Regressor, XGBoost, Random Forest, Linear Regression      |
| **Model Evaluation** | R² Score, MAE, MSE, RMSE                                           |
| **Data Handling**    | Excel (`.xlsx`) via `pandas.read_excel()`                          |
| **Web Framework**    | Streamlit (for interactive web app deployment)                     |
| **Model Saving**     | joblib (for saving and loading models)                             |
| **IDE/Editor**       | Jupyter Notebook, Visual Studio Code                               |
| **Version Control**  | Git & GitHub                                                       |

---  

## 📚 Project Workflow  
### 1️⃣ Import Necessary libraries  
```
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
```

---  

### 2️⃣ Data Loading and Preview  

<img width="1391" height="639" alt="Screenshot 2025-07-12 170003" src="https://github.com/user-attachments/assets/e50d2a93-f918-4caf-a9b6-a2665befec2d" />  

---  

### 3️⃣ Data Overview & Cleaning  
- **Dataset Shape**:  
  The dataset contains **10,683 rows** and **11 columns**.

- **Data Types Summary**:
  - 10 columns are of type `object` (categorical/text).
  - 1 column (`Price`) is of type `int64`.

- **Initial Null Value Check**:
  - `Route`: 0.009% missing values
  - `Total_Stops`: 0.009% missing values
  - All other columns have 0% missing values.

- **Handling Missing Values**:
  - Dropped the rows with missing values in `Route` and `Total_Stops` as the missing percentage was negligible (<0.01%).
  - Resulting dataset has **no null values**.
 
---  

### 4️⃣ Feature Engineering  
To enhance the dataset and make it suitable for machine learning models, several new features were extracted and irrelevant columns were dropped.  

**📅 Date of Journey Features**  
- Extracted the day, month, and year from the `Date_of_Journey` column:
  - `journey_day`
  - `journey_month`
  - `journey_year`
- The entire dataset was from the year 2019, so `journey_year` was dropped due to no variance.  

**⏰ Departure Time Features**  
- Extracted the hour and minute from the `Dep_Time` column:
  - `dep_hour`
  - `dep_min`
- Dropped the original `Dep_Time` column after transformation.  

**🛬 Arrival Time Features**  
- Extracted the hour and minute from the `Arrival_Time` column:
  - `arr_hour`
  - `arr_min`
- Dropped the original `Arrival_Time` column after transformation.  

**⏳ Flight Duration Features**  
- The `Duration` column contained flight durations in a mix of formats (`'2h 50m'`, `'5h'`, `'25m'`, etc.)
- Cleaned and standardized the duration format to ensure both hours and minutes were present.
- Extracted:  
  - `duration_hour`
  - `duration_mins`
- Dropped the original `Duration` column after feature extraction.  

**Final Feature List (After Engineering)**  
After transformation, the dataset contains **15 features** including both original and engineered ones:
- `Airline`, `Source`, `Destination`, `Route`, `Total_Stops`, `Additional_Info`, `Price`
- `journey_day`, `journey_month`
- `dep_hour`, `dep_min`
- `arr_hour`, `arr_min`
- `duration_hour`, `duration_mins`  
These features were later encoded and used for training various machine learning models.

---  

### 5️⃣ Exploratory Data Analysis (EDA)  

#### ✈️ Airline vs Price  

<img width="3212" height="800" alt="Airline" src="https://github.com/user-attachments/assets/973b03e9-b6fe-45b0-999a-77e85799fa22" />  

- **Jet Airways Business** has the highest ticket prices with a median around ₹55,000–₹60,000.
- **Jet Airways**, **Multiple carriers**, and **Air India** offer mid-range fares between ₹8,000–₹15,000.
- **Low-cost carriers** like **SpiceJet**, **GoAir**, **IndiGo**, and **Air Asia** have relatively lower ticket prices.
- **Vistara Premium Economy**, **Multiple carriers Premium Economy**, and **Trujet** offer niche services with smaller sample sizes but noticeable consistency in pricing.

#### 🛫 Source vs Price  

<img width="1812" height="600" alt="Source" src="https://github.com/user-attachments/assets/ebffd7e2-d0e5-472a-a68d-8554ca404725" />

- **Banglore** and **Delhi** sources show high variation and some extreme outliers, likely due to business class or long-haul combinations.
- **Kolkata** and **Mumbai** sources have relatively moderate prices.
- **Chennai** emerges as the most budget-friendly departure city with the narrowest fare range.

#### 🧭 Destination vs Price  

<img width="1812" height="600" alt="Destination" src="https://github.com/user-attachments/assets/934ec4cf-a593-4008-8506-aaa225d50f7b" />

- **New Delhi** and **Cochin** destinations have the widest price range, with several high-value outliers above ₹40,000.
- **Banglore** and **Hyderabad** follow a similar mid-range pricing pattern.
- **Kolkata** and **Delhi** are more consistent with lower median fares and fewer outliers, indicating stable pricing.  

Flight prices are significantly influenced by airline type (luxury vs budget), source city, and destination. Outliers in each category indicate the presence of premium services (e.g., business class) or longer durations.  

---  

### 6️⃣ Feature Scaling  

- Applied **one-hot encoding** to convert categorical features into numerical format:
  - `Airline` → Converted into dummy variables like `Airline_IndiGo`, `Airline_SpiceJet`, etc.
  - `Source` → Converted into `Source_Delhi`, `Source_Kolkata`, `Source_Chennai`, etc.
  - `Destination` → Converted into `Destination_Delhi`, `Destination_Cochin`, `Destination_Hyderabad`, etc.
  
- Replaced `"New Delhi"` with `"Delhi"` in the `Destination` column to avoid duplication.

- Extracted new temporal features:
  - `journey_day` and `journey_month` from the journey date.
  - `dep_hour`, `dep_min` from departure time.
  - `arr_hour`, `arr_min` from arrival time.

- Processed flight duration:
  - Created `duration_hour` and `duration_mins` as separate features to represent travel time more accurately.

- Dropped original columns (`Airline`, `Source`, `Destination`, `Additional_Info`, and `Route`) after feature extraction and transformation.

---  

### 7️⃣ Train-Test Split  

- The dataset was split into **independent features (X)** and the **target variable (y)**:
  - **X**: All columns except `Price`
  - **y**: The `Price` column, which is the target for prediction

- Applied **train-test split** to evaluate model performance on unseen data:
  - **Training Set**: 80% of the data (used for training the model)
  - **Test Set**: 20% of the data (used for evaluating the model)
  - Used `random_state=42` to ensure reproducibility of results

---  

### 8️⃣ Model Building  
📊 Model Performance Comparison After Hyperparameter Tuning  

| Model                     | Training_Accuracy (R²) | Testing_Accuracy (R²) | MAE      | MSE         | RMSE    |
|--------------------------|------------------------|------------------------|----------|-------------|---------|
| Linear Regression        | 0.6137                 | 0.5999                 | 1995.94  | 8625736.14  | 2936.96 |
| Decision Tree Regressor | 0.9692                 | 0.7292                 | 1331.49  | 5839293.81  | 2416.46 |
| Random Forest Regressor | 0.8822                 | 0.8264                 | 1258.07  | 3743398.52  | 1934.79 |
| GradientBoosting Regressor | 0.7857              | 0.7872                 | 1521.16  | 4589000.16  | 2142.20 |
| XGBoost Regressor        | 0.9234                 | 0.8595                 | 1141.03  | 3029237.10  | 1740.47 |
| CatBoost Regressor       | 0.9241                 | 0.8651                 | 1139.90  | 2908683.63  | 1705.49 |
| LightGBM Regressor       | 0.8730                 | 0.8260                 | 1250.42  | 3751264.01  | 1936.82 |  

🏆 **Best Model: CatBoost Regressor** (better performance on all evaluation metrics)  
**Reason for Selection:**  

**Highest Testing Accuracy (R²):** `0.8651`  
**Lowest MAE:** `1139.90`  
**Lowest MSE:** `2908683.63`  
**Lowest RMSE:** `1705.49`  

---  

## Feature Importances: CatBoost Regressor 

<img width="1404" height="737" alt="Screenshot 2025-07-12 165608" src="https://github.com/user-attachments/assets/0314206f-4fc1-499e-a01b-05933653d8a7" />


## 📊 Streamlit App Deployment  

<img width="1581" height="1002" alt="Travel_app_pred" src="https://github.com/user-attachments/assets/80fd3a53-60ca-48fd-bcbb-ab58c1e34648" />


## 🧑‍💻 Author

**Ashwini Bawankar**  
*Data Science Intern | Passionate about Machine Learning*

---

## 📬 Contact

📧 Email: [abawankar13@gmail.com]  
🔗 LinkedIn: [https://www.linkedin.com/in/ashwini-bawankar/]  

