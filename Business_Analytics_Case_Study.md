# Business Analytics Case Study

Before starting with the notebook, please complete the mandatory Pre-Test. This is critical for quantifying the learning progress.

You can find the Pre-Test here: https://forms.cloud.microsoft/e/wpFZwLNJyZ

## 1. Business Understanding

In this section, you will familiarize yourself with the Global Bike Inc. (GBI) model company and the two core use cases: **Customer Lifetime Value (CLV) Prediction** and **Customer Segmentation**.
### CLV-Prediction:
A Method with which Business Analysts can make data-driven decisions to predict how much money a customer will spend with the company. 

### Customer Segmentation:
Oftentimes, customers share certain characteristics with each other. Customer Segmentation analyzes these patterns to build clusters in which customers are expected to behave similarly. 

#### Exercise 1:
**Exercise 1: Identify which scenarios can be solved by applying CLV-Prediction and Customer Segmentation to improve business decision making** To add your answer double click the cell, select your box with "X" and then press Shift + Enter

**Note:** Please check exactly 2 boxes in total – one for CLV and one for Customer Segmentation

- [ ] Management wants to evaluate the impact of personalized marketing campaigns
- [ ] To minimize returns GBI is thinking about leveraging Machine Learning to predict production errors
- [ ] GBI is looking to optimize the sales process. By collecting transactional data they want to build a model to look for inefficiencies
- [ ] Customer Support is complaining about missing a good strategy to prioritize customer check ups and are asking for a data driven approach to choose which customers to call first

#### Exercise 2: Business Impact Estimation
GBI currently relies on a random calling strategy to prevent customers from leaving (churning). The customer support team has the capacity to make 100 calls per month. 
To estimate the business value of these calls, we have the following baseline metrics from past observations:
- **Average Churn Rate:** On average, about **3%** of customers are at risk of churning in any given month.
- **Average Customer Value:** The average lifetime value of a B2B customer at GBI is **$60,000**
- **Success Rate:** If a customer who is actually at risk of churning is called, there is a **20% chance** to convince them to stay. (Calling non-churning customers has no additional effect).

**Your Task:** Do a quick approximation on the expected impact of the support calls. If the support team randomly selects 100 customers to call this month, how much saved revenue can GBI expect from this random strategy?
Double-click this cell and write your estimation below:

**My Estimation:**
Saved Revenue (Random Strategy): $ ______




## 2. Data Understanding

The dataset that is available for our Business Analytics use case consists of four SAP ERP tables: `KNA1`, `KNVV`, `VBAK_messy`, and `VBAP`.

Before diving into the data, you can check out the **Companion PDF** to see the Entity-Relationship (ER) diagram, which gives you a theoretical overview of how these tables are connected.


### Exploratory Data Analysis (EDA)
Before we can start building models, we need to understand what our data actually looks like.
EDA is a method that provides a structured analysis of a new dataset, to visualize general patterns and trends in the data to get a quick first impression. 
Since EDA is a routine task of Data Analytics, we can leverage existing libraries that help us automate a lot of the process. 

**Pandas** is a versatile python library for data analysis and the industry standard for handling tabular data. 

#### Exercise 3: Code completion
Your colleague started writing the code to inspect the `VBAK_messy` dataset, but left two important functions blank. 

**Your Task:** We need three specific functions to get an overview of the data. 
Have a look at the **Data Analyst Cheat Sheet in the Companion PDF** to learn about the most important pandas functions. 


1. Find the method that **returns the first n rows** of a DataFrame.
2. Find the method that prints a **concise summary** of a DataFrame (including data types and non-null counts)
3. Find the method that provides a numerical overview of **patterns and distributions** of a DataFrame

Replace the `___()` placeholders in the code cell below with the functions you found!

*For more detailed information, you can visit the official pandas documentation: [pandas.pydata.org](https://pandas.pydata.org/docs/reference/frame.html)*



```python
import pandas as pd

# Load tables
df_kna1 = pd.read_csv('KNA1.csv')
df_knvv = pd.read_csv('KNVV.csv')
df_vbak = pd.read_csv('VBAK_messy.csv')
df_vbap = pd.read_csv('VBAP.csv')

# Show the first few rows of VBAK
# Exercise 3.1: Check the first row's to see the structure of the Data Frame
df_vbak.___()



```


```python
# Overview of data structure and missing values
# Exercise 3.2: Check the general information about the data frame. what variables can you identify
df_vbak.___()
```


```python
# Statistical summary of numerical columns
# Exercise 3.3: What method is best suited to learn about patterns and distributions in the dataset?
df_vbak.___()

```

#### Exercise 4: Data Visualization
Now that we have a broad understanding of the dataframe, we can investigate more closely what patterns and distributions are hidden within. 
Writing code for visualization can be complex, so your colleague already prepared the plotting scripts for you. 
Visualizing the dataset is the most effective way to get an intuitive understanding of the data quality without getting lost in the details. Often, this helps us spot anomalies or "dirty data" that would break our Machine Learning model later on. 
**Run the cells below to generate the graphs, then answer the questions!**




```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 4))

# We plot each transaction (index) against its order value (NETWR)
def scatterplot():
    plt.scatter(df_vbak.index, df_vbak['NETWR'], alpha=0.5, color="royalblue")
    plt.title("Scatterplot of all transactions (Outlier detection)")
    plt.xlabel("Transaction Index")
    plt.ylabel("Order Value (USD)")
    plt.show()
scatterplot()

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 13
         11     plt.ylabel("Order Value (USD)")
         12     plt.show()
    ---> 13 scatterplot()


    Cell In[1], line 8, in scatterplot()
          7 def scatterplot():
    ----> 8     plt.scatter(df_vbak.index, df_vbak['NETWR'], alpha=0.5, color="royalblue")
          9     plt.title("Scatterplot of all transactions (Outlier detection)")
         10     plt.xlabel("Transaction Index")


    NameError: name 'df_vbak' is not defined



    <Figure size 1200x400 with 0 Axes>


**Question 4.1: Analyzing the Order Values (Double-click to edit) Look at the scatterplot of the NETWR (Net Value) column. Which statement correctly describes the anomaly we see in the data?**
- [ ] Most of our orders have exactly the same value, creating a flat line at the bottom. 
- [ ] There is one extreme outlier which distorts the rest of the data 
- [ ]  Most of the orders have exactly a value of $0
- [ ] There are no negative values in this dataset. 


```python
# ==========================================
# Visually detect missing values (NaN)
# ==========================================
def plot_missing_values():
    # Count missing values per column
    missing_counts = df_vbak.isnull().sum()
    
    plt.figure(figsize=(8, 4))
    missing_counts.plot(kind='bar', color='tomato')
    plt.title("Number of missing values per column")
    plt.ylabel("Count of NaN (Not a Number)")
    plt.xticks(rotation=0)
    plt.show()

plot_missing_values()
```

**Question 4.2: Handling Missing Data (Double-click to edit) The bar chart shows that we have missing values (NaN) in the `KUNNR` (Customer ID) column. Why is this a critical problem for our specific CLV Use Case?**
- [ ] Missing values in KUNNR should simply be replaced by the average Customer ID to keep the dataset large. 
- [ ] Machine Learning models can automatically handle missing Customer IDs, so we can ignore this. 
- [ ] CLV is calculated per customer. If a transaction has no Customer ID, we cannot assign the revenue to anyone, making the data useless for our model. 

## 3. Data Preparation
This crucial phase prepares the dataset for the model training and consists of two main stages:
1. **Data Cleaning:** Since the ML model will learn patterns within the dataset, low-quality training data would result in a low-quality prediction model. This principle is known as **Garbage in, Garbage out**. To prevent dirty data from distorting our model, we have to apply data cleaning techniques and filter out problematic datapoints. 

2. **Feature Engineering:**
Once the data is cleaned, we have to make it interpretable for the ML algorithm. Raw, transactional data in a vacuum is hard for a model to handle. We have to add context by aggregating the raw variables into meaningful predictors, so called **Features**. This step is called **Feature Engineering** and we will have a closer look how it works right after the data cleaning step. 


### Data Cleaning
Pandas is a python library specialized in data manipulation and analysis. It provides powerful tools to clean and filter datasets. This is done by applying specific methods (like dropping empty rows) or by using **boolean indexing** (filtering rows based on a true/false condition). You can find the most important functions in the **Data Analyst Cheat Sheet in the Companion PDF** or check the **[official Pandas User Guide on Missing Data](https://pandas.pydata.org/docs/user_guide/missing_data.html)**.


During **EDA** we already identified some distinct data quality issues:
1. **Outlier:** Sometimes a "fat" finger during manual data entry can mess up a datapoint. This not only makes that datapoint useless, but even worse, an extreme outlier distorts the entire dataset and ruins statistical averages.
2. **Incomplete datapoints: (NaN)** When a new transaction gets added to the ERP System (e.g., SAP S/4HANA), human error can lead to empty values. Oftentimes, modern ERP systems have strict internal data quality rules to avoid incomplete entries. Nonetheless, this issue should always be checked and resolved. If an order has no Customer ID, we can't assign the revenue to anyone, making it useless for our CLV model.
3. **Negative Values:** Identifying this issue takes domain knowledge and experience. Luckily, your colleague is a seasoned veteran and already suspected that this dataset might contain returns. Have a look at the next cell, where he expanded the EDA to check for negative order values. Filtering these out is crucial: The machine learning algorithms we use for CLV prediction are mathematically designed to only work with positive numbers. Neglecting to clean this would result in the model crashing during the training phase!





```python
def plot_negative_values():
    # Count how many transactions are >= 0 and < 0
    neg_count = (df_vbak['NETWR'] < 0).sum()
    pos_count = (df_vbak['NETWR'] >= 0).sum()
    
    plt.figure(figsize=(6, 4))
    plt.bar(['Normal (>= 0)', 'Returns (< 0)'], [pos_count, neg_count], color=['mediumseagreen', 'crimson'])
    plt.title("Normal Orders vs. Returns")
    plt.ylabel("Number of transactions")
    
    # Write values as text above the bars
    plt.text(0, pos_count + 500, str(pos_count), ha='center')
    plt.text(1, neg_count + 500, str(neg_count), ha='center')
    plt.show()
plot_negative_values()
```


#### Exercise 5: Clean the Dataset
Your colleague was in a rush and left out some important parts of the code. Replace the `___`placeholders to finish the data cleaning step!

*Hint: At least he left some helpful comments! If you are stuck, read the green comments in the code cell below or ask your AI Assistant for help*


```python
# ==========================================
# Data Cleaning: Clean VBAK
# ==========================================

# Step 1: Remove missing Customer IDs (NaN in KUNNR)
# Remove all rows where a value is missing in the specified column.
df_vbak_clean = df_vbak.___(subset=['KUNNR'])


# Step 2: Remove negative values (returns)
# We overwrite the DataFrame with a filter that only keeps values >= 0.
df_vbak_clean = df_vbak_clean[df_vbak_clean['___'] >= 0]

# Step 3: Remove the extreme outlier
# We filter all realistic orders (e.g. everything under 10 million USD).
df_vbak_clean = ________

# For verification: How many rows did we filter?
print(f"Rows before cleaning: {len(df_vbak)}")
print(f"Rows after cleaning: {len(df_vbak_clean)}")

```

### Feature Engineering:
Feature Engineering is the process of taking raw transactional data and applying business context to improve the predictive capabilities of the model. In general, **feature engineering is our way to integrate human business understanding into a language that is understandable for the ML model.**

For our specific use case, ML models work best with so-called **RFM-Features**. This means that *Recency*, *Frequency*, and *Monetary* values are calculated for each customer and used as predictors to estimate their Customer Lifetime Value.

How do we extract these features from our raw data? 
Looking back at the Data Understanding phase, we can see that the `VBAK` table has the following structure. Each row in this table represents a single order: 

| Variable | Description |
|---|---|
| **VBELN** | Order ID (Primary Key) |
| **KUNNR** | Customer ID |
| **ERDAT** | Date of the order |
| **NETWR** | Net value of the order |
| **WAERS** | Currency |


To train our model, we must aggregate these individual orders up to the **customer level**. We can calculate our RFM features as follows:

**Recency:** When did the customer buy from us for the last time? This is the time interval between their *last* order (`ERDAT`) and today.

**Frequency:** How often does the customer buy? This is the total amount of orders per customer (Count of `VBELN`).

**Monetary Value:** How much money does the customer spend? This is the total value of all orders per customer (Sum of `NETWR`).


**The Code Implementation:**
To achieve this aggregation in Python, we can leverage Pandas. Since dataframes follow the same logic as tables in relational databases, Pandas provides almost the exact same operators as SQL. To calculate the RFM features, we need to group our orders by the Customer ID (`KUNNR`). If you are familiar with SQL, you already know the `GROUP BY` operator! You can check the **Data Analyst Cheat Sheet in the Companion PDF** to see how this translates into Pandas syntax.


#### Exercise 6: Feature Engineering
Use the pandas library to aggregate the columns of the dataset into RFM features that can be used by the ML model. 

*Hint: If you don't know the exact names of the pandas functions to count or sum up values, check the Data Analyst Cheat Sheet or ask your AI Assistant!*



```python
%%time 
import datetime as dt

# 1. Ensure the date is a true datetime object
df_vbak_clean['___'] = pd.to_datetime(df_vbak_clean['___'])



# 2. Set snapshot date (one day after the most recent purchase in the dataset)
snapshot_date = df_vbak_clean['___'].max() + dt.timedelta(days=1)


# 3. Aggregation (RFM)
# features are created for each customer
rfm = df_vbak_clean.___('___').___({
    'ERDAT': lambda x: (snapshot_date - x.max()).days,  # Recency
    'VBELN': 'nunique',                                 # Frequency
    'NETWR': 'sum'                                      # Monetary
}).reset_index()


# 4. Rename columns cleanly
rfm.rename(columns={
    'ERDAT': '___',
    'VBELN': '___',
    'NETWR': '___'
}, inplace=True)

# Check result
print("Number of customers after aggregation:", len(rfm))
rfm.head()

```

### Creating the Target Variable (Temporal Split)
To predict how much a customer will spend in the *future*, our model needs to learn from the *past*. If we just calculate the RFM values across the entire dataset and use that to predict their total spending, the model will just memorize the data instead of learning a real pattern (this is called Data Leakage). 

To prevent this, we split our data temporally:
1. **Features (The Past):** We calculate the RFM values (Recency, Frequency, Monetary) using only the data from **2022**. This represents the customer's historical behavior.
2. **Target (The Future):** We calculate the total revenue generated by each customer in **2023**. This is the `CLV` our model will try to predict.

By doing this, we teach the model: *"Given how a customer behaved in Year 1, here is how much they actually spent in Year 2."*




```python
# ==========================================
# Temporal Split for Model Training
# ==========================================
# To train a model that predicts FUTURE customer value, 
# we split the data by time:
# - Year 1 (2022): Features (how did the customer behave?)
# - Year 2 (2023): Target (how much did they actually spend?)

# Note: Just run this cell to prepare the training data.

cutoff_date = '2023-01-01'

# Features: RFM from Year 1 only
df_year1 = df_vbak_clean[df_vbak_clean['ERDAT'] < cutoff_date]
df_year2 = df_vbak_clean[df_vbak_clean['ERDAT'] >= cutoff_date]

snapshot = pd.to_datetime(cutoff_date)
rfm_training = df_year1.groupby('KUNNR').agg({
    'ERDAT': lambda x: (snapshot - pd.to_datetime(x).max()).days,
    'VBELN': 'nunique',
    'NETWR': 'sum'
}).reset_index()
rfm_training.columns = ['KUNNR', 'Recency', 'Frequency', 'Monetary_Year1']

# Target: Total spending in Year 2
target = df_year2.groupby('KUNNR')['NETWR'].sum().reset_index()
target.columns = ['KUNNR', 'CLV_Year2']

# Merge features + target
training_data = rfm_training.merge(target, on='KUNNR', how='left')
training_data['CLV_Year2'] = training_data['CLV_Year2'].fillna(0)

print(f"Training dataset ready: {len(training_data)} customers")
print(f"Features: Recency, Frequency, Monetary_Year1")
print(f"Target: CLV_Year2 (actual spending in 2023)")
training_data.head()

```

#### Exercise 7: Why use Pandas? (Performance Check)
In Exercise 6, you successfully used Pandas to aggregate the data. But what if you didn't know Pandas and tried to implement the exact same logic using standard Python `for`-loops?

**Your Task:**
Have a look at the runtime of the Pandas code cell you just executed in Exercise 6. The execution time is displayed as *Wall time* at the bottom of the cell. 

Now run the code cell below, which executes the exact same task using a manual Python for-loop. Note the execution time. 

How does the runtime compare? Can you imagine why there is such a massive difference?
*Hint: If you are curious, ask your AI Tutor about the concept of **Vectorization!***





```python
%%time
# ==========================================
# Manual Approach: Standard Python (For-Loop)
# ==========================================
rfm_manual = {}

# We iterate through every single row of the dataset
for index, row in df_vbak_clean.iterrows():
    customer = row['KUNNR']
    netwr = row['NETWR']
    
    # If the customer is new, create an empty profile
    if customer not in rfm_manual:
        rfm_manual[customer] = {'Frequency': 0, 'Monetary': 0}
    
    # Add the current order to the customer's profile
    rfm_manual[customer]['Frequency'] += 1
    rfm_manual[customer]['Monetary'] += netwr
    
print(f"Aggregated {len(rfm_manual)} customers manually.")

```

## 4. Modeling

Once we have the data fully prepared and our features engineered, we can move on to the Modeling phase. Here, we run and compare different **Machine Learning algorithms** to find the best fit for our specific RFM dataset.

Finding the perfect model architecture and tuning its parameters is usually a complex and time-consuming step. Luckily for us, we can leverage modern tools that automate a lot of this repetitive work. **Watsonx.ai's AutoAI Engine** is an automated pipeline that takes our dataset, tests a huge variety of different algorithms on it, and optimizes them. At the end, it evaluates every model using specific performance metrics and returns a leaderboard of the best models for us to choose from.

To learn more about the theory behind AutoAI and how to interpret the performance metrics (like R² or RMSE), have a look at the deep-dive in the PDF: **Data Analyst Cheat Sheet in the Companion PDF**

#### Exercise 8: Choosing the Model
First, we have to do a bit of setup. Retrieve your API key and Project ID as explained in the Setup Guide and add them to the authentication code below. 
Once the connection is established, you can run the AutoAI pipeline. This will train multiple models simultaneously. 

Afterwards, **evaluate** the output leaderboard. By analyzing the different performance metrics (e.g. RMSE, R²) choose the most suitable model for our CLV use case. 

**Note:** If there are any issues with the Watsonx.ai Authentication, do not despair. Take a minute to ask your AI Tutor if it knows a quick fix. If you cannot resolve the issue quickly, don't get hung up on it! Just scroll all the way down to the **Troubleshooting** section of this notebook. There, you will find a fallback method to train a model locally without having to run the AutoAI engine.



```python
# Run this cell to install the necessary dependencies
!pip install ibm-watsonx-ai
```


```python
# ==========================================
# Watsonx.ai Authentication
# Insert your authentication data and then run this cell
# ==========================================
from ibm_watsonx_ai import APIClient

credentials = {
    "url": "https://eu-de.ml.cloud.ibm.com", 
    # Insert your API key
    "apikey": "<YOUR_API_KEY_HERE>"
}
# Insert your project ID
project_id = "<YOUR_PROJECT_ID>"

# Initialize client and set project
client = APIClient(credentials)
client.set.default_project(project_id)

print("Setup successful. Your authentication with watsonx.ai worked!")
print("You can continue by executing the next cell")

```


```python
from ibm_watsonx_ai.experiment import AutoAI
from ibm_watsonx_ai.helpers import DataConnection, ContainerLocation

# 1. We tell Watsonx we want to start an AutoAI experiment
experiment = AutoAI(credentials, project_id=project_id)

# 2. We configure the optimizer
pipeline_optimizer = experiment.optimizer(
    name="CLV Regression (Monetary)",
    prediction_type=AutoAI.PredictionType.REGRESSION,
    prediction_column="CLV_Year2",
    scoring=AutoAI.Metrics.ROOT_MEAN_SQUARED_ERROR
)

# 3. We remove the ID since it is not a real feature
training_df = training_data.drop(columns=['KUNNR'])



# --- NEW: UPLOAD DATA ---
print("Uploading data to Watsonx Cloud Storage...")
data_connection = DataConnection(
    location=ContainerLocation(path="rfm_training_data.csv")
)
data_connection.set_client(client)
# 1. We save it 100% cleanly LOCALLY
training_df.to_csv("rfm_training_data_local.csv", index=False, header=True)
# 2. We upload the physical file (not the DataFrame)
data_connection.write(data="rfm_training_data_local.csv", remote_name="rfm_training_data.csv")
print("Upload completed!")
# ----------------------------

# 4. START: We pass the data connection as a list (as required by IBM)
print("Starting Watsonx AutoAI... (This will take a few minutes!)")
run_details = pipeline_optimizer.fit(
    training_data_reference=[data_connection],  
    background_mode=False 
)

training_df.rename(columns={'Monetary_Year1': 'Monetary', 'CLV_Year2': 'Monetary_Target'}, inplace=True)



```


```python
# ==========================================
# Evaluate AutoAI results
# Execute this cell to see the results
# ==========================================

# 1. Retrieve leaderboard: Which algorithms were tested?
summary = pipeline_optimizer.summary()
print("Top Models:")
display(summary)


```

### Running the Model
When working with Machine Learning, the deployment workflow consists of two distinct stages:

1. **Model Training (Fitting):** During this phase, the chosen algorithm analyzes the historical training data and finds statistical relations between the input features (RFM) and the target variable (CLV). The ouput of this phase is a fitted mathematical model. 
*We completed this stage using AutoAI, which trained and evaluated multiple model architectures.*

2. **Inference (Prediction):** In this stage, the fitted model is applied to new, unseen data. It utilizes the relationship that was learned in the fitting phase to predict the target variable based on the newly provided input features.  
*This is the step we are executing now. We will use the model to predict the CLV for our customers and translate these numerical predictions into actionable business insights.*


#### Exercise 9: Run the Model
Run the code cell below to apply the trained model to our customer dataset. The output will add a new column `Predicted_CLV` to each customer.
Once the predictions are generated, answer the following questions (*Double-click to edit*):

1. Which customer has the **highest** predicted CLV? What does this mean for the call center team?
2. Which customer has the **lowest** predicted CLV? Should we still invest resources into calling them?
3. Look at the `Absolute_Error`. At first glance, an error of e.g. $10,000 seems massive. But compare it to the `Actual_CLV`of that customer. Is this absolute error actually a problem for our model? Explain your reasoning.


```python
# ==========================================
# INFERENCE (Prediction using Watsonx Model)
# ==========================================

import numpy as np

# 1. Prepare data (Features & Target)
X = training_data[['Recency', 'Frequency', 'Monetary_Year1']]
y = training_data['CLV_Year2']

# 2. Get the best pipeline from AutoAI
best_pipeline = pipeline_optimizer.get_pipeline()

# 3. INFERENCE: Generate prediction (We predict the CLV)
predictions = best_pipeline.predict(X)

# The prediction is initially just a raw Numpy array of numbers:
print(predictions[:5])


# ==========================================
# Interpretation of predictions
# ==========================================
# We build a clear DataFrame to analyze the predictions
results = X.copy()
results['KUNNR'] = training_data['KUNNR']
results['Actual_CLV ($)'] = np.round(y, 2)
results['Predicted_CLV ($)'] = np.round(predictions, 2)
results['Absolute_Error ($)'] = np.round(abs(results['Actual_CLV ($)'] - results['Predicted_CLV ($)']), 2)

# Set KUNNR as index for a nicer representation
results.set_index('KUNNR', inplace=True)

print("Inference completed. Results DataFrame created.")
# Show top 5 customers with HIGHEST predicted CLV
print("Top 5 Customers (Highest CLV):")
display(results.sort_values(by='Predicted_CLV ($)', ascending=False).head(5))

# Show bottom 5 customers with LOWEST predicted CLV
print("\nBottom 5 Customers (Lowest CLV):")
display(results.sort_values(by='Predicted_CLV ($)', ascending=True).head(5))

```

#### Beyond Traditional ML: Tabular Foundation Models
During the modeling phase, we compared and evaluated different Machine Learning algorithms. While tools like AutoAI are extremely powerful, they are restricted to a predefined selection of "traditional" algorithms, heavily relying on so-called *Ensemble Methods* (like Gradient Boosting or Random Forests).

However, recent AI research has introduced a completely new paradigm for handling tabular data: **Tabular Foundation Models (TFMs)**, such as SAP's RPT-1. To understand the underlying architecture and how TFMs differ from traditional ML, please refer to the deep dive in the accompanying document: **Companion PDF: Chapter 2.4 The Future of AI (Taxonomy & TFMs)**.

To summarize the most critical difference: TFMs are **pre-trained** on massive amounts of data. This means they skip the resource-intensive model training stage entirely and can work *out of the box* on new datasets (Zero-Shot Learning).

Since this emerging technology is showing promising results and research suggests that TFMs might supersede traditional ML models for certain business applications in the future, it is our due diligence to evaluate this cutting-edge approach for Global Bike Inc.




```python
# ==========================================
# Phase 4: Data export for SAP RPT-1 Playground
# ==========================================

# 1. We draw 1,000 random customers as "memory" for the model (In-Context Learning)
# This is well below the limit of 2,048 rows.
rpt_context = training_df.sample(n=1000, random_state=42).copy()

# 2. We select 10 additional random customers who were NOT in the 1,000 context customers
# These will be our test customers (Prediction Targets)
test_kunden = training_df.drop(rpt_context.index).sample(n=10, random_state=42).copy()

# 3. The SAP RPT-1 model needs a placeholder to know what to predict.
# According to the documentation, we simply replace the real revenue with the string '[PREDICT]'
test_kunden['Monetary'] = '[PREDICT]' 

# 4. Append the test customers to the end of the dataset
rpt_final_df = pd.concat([rpt_context, test_kunden])

# 5. Save the file, ready for web upload
rpt_final_df.to_csv('rpt_1_playground_data.csv', index=False)
print("File 'rpt_1_playground_data.csv' successfully created! (1010 rows)")

```


#### Exercise 10: RPT-1 Hands on
Since TFMs do not require local training, we can test SAP's RPT-1 directly through their web interface.

1. Follow the **[RPT-1 Setup Guide](./Setup.pdf)** in your PDF to upload our dataset to the RPT-1 playground.
2. Run the code cell below to display the True Revenue and our Watsonx Predictions for the 10 test customers.
3. Put your browser and Jupyter Notebook side-by-side. Compare the predictions generated by SAP RPT-1 in your browser with our table. Just by eyeballing it, which model seems to be closer to the True Revenue on average?


```python
# Create a visual comparison table for the 10 test customers
comparison_df = test_kunden[['Recency', 'Frequency']].copy()
comparison_df.index.name = 'Customer_ID'

# 1. Get the TRUE Revenue for these 10 customers
comparison_df['True_Revenue'] = training_data.loc[test_kunden.index, 'CLV_Year2'].round(2)

# 2. Get the Watsonx Prediction for these 10 customers
X_test = training_data.loc[test_kunden.index, ['Recency', 'Frequency', 'Monetary_Year1']]
comparison_df['Watsonx_Prediction'] = np.round(best_pipeline.predict(X_test), 2)

display(comparison_df)

```


**Quiz:**
TFMs and traditional ML are two fundamentally distinct approaches to handling data. Let's see if you can identify the advantages and disadvantages of each. (Mark the correct statements with an x)

**Question 10.1: Time-to-Value & Training**
Which of the following statements correctly describes the biggest advantage of Tabular Foundation Models (TFMs) compared to traditional Machine Learning?
- [ ] TFMs are much easier to program in Python than traditional ML algorithms.
- [ ] TFMs train much faster on your local machine because they use less computing power.
- [ ] TFMs are pre-trained on massive datasets. They skip the model training stage entirely and can generate predictions on new datasets out-of-the-box (Zero-Shot)

**Question 10.2: Technical Limitations:** 
Despite their impressive capabilities, TFMs have significant technical limitations. Which of the following is the biggest bottleneck when applying TFMs like RPT-1 to large enterprise datasets?
- [ ] They only work on datasets that have exactly the same columns as their training data.
- [ ] The underlying Transformer architecture scales quadratically. This severely limits the maximum dataset size (Context Window) that the model can process at once.
- [ ] They cannot process text-based categories, only numerical data

**Question 10.3 Business Trade-offs** 
If TFMs perform so well, why might Global Bike Inc. still prefer a traditional AutoAI model for its Customer Lifetime Value prediction?
- [ ] Traditional models are often easier to explain to stakeholders (Explainability) and can be run completely offline, ensuring sensitive customer data doesn't leave the company (Data Privacy)
- [ ] Traditional models are usually much more expensive to run than Foundation Models.
- [ ] Traditional models can handle missing values automatically, whereas TFMs require perfectly clean data. 


## 5. Evaluation

The data mining process is nearing its completion. While we already briefly looked at the mathematical performance during the modeling phase, rigorous Data Science requires a formal Evaluation phase.

According to the CRISP-DM framework, the goal of this phase is not just to look at abstract mathematical metrics, but to evaluate if the model actually achieves our initial business objectives. We must translate the raw statistical output into actionable, valuable business insights.

#### Exercise 11: Translating Metrics to Business Value
Take a look at the two main performance metrics of our final model: $R^2$ (R-Squared) and **RMSE** (Root Mean Squared Error)

1. **Understanding $R^2$:** What does the $R^2$ score tell us about the general quality of our predictions? Is the model reliable enough to base business decisions on it?
2. **Understanding RMSE:** The RMSE is measured in the exact same currency as our target variable (e.g., USD). If our model has an RMSE of e.g. $1,200, what does this mean for our Call Center Agents when they look at a predicted CLV of a customer?


## 6. Deployment

Once the data mining is complete, the gained knowledge has to be translated into practical, actionable business decisions. In a real-world scenario, this would involve deploying the model into a production system and integrating it into the company's existing workflows.

For our case study, we will focus on the business side of the deployment: Presenting the value of our work to Global Bike Inc.'s management.



#### Exercise 12: Calculate the Business Impact of CLV Prediction
Remember your rough estimation from Exercise 2? Back then, we calculated the expected saved revenue from a random calling strategy:
100 calls × 3\% churn rate × 20\% success rate × $60,000 avg. customer value = $36,000

Now that we have a trained ML model, the support team no longer has to call random customers. Instead, they can prioritize the **Top 100 most valuable customers** based on our CLV predictions.

**Your Task:** Fill in the code gaps to calculate the improved business impact. How does the ML-informed strategy compare to the random baseline from Exercise 2?





```python
# ==========================================
# Deployment: Business Impact Calculation
# ==========================================

# 1. Sort customers by predicted CLV and select the Top 100
top_100 = results.sort_values(by='___', ascending=False).head(___)

# 2. Calculate the average CLV of our prioritized customers
avg_clv_top100 = top_100['___'].mean()

# 3. Apply the same formula as Exercise 2, but with the improved avg CLV
churn_rate = 0.03  
success_rate = 0.20
saved_revenue_ml = 100 * churn_rate * success_rate * avg_clv_top100

# 4. Compare with the random baseline
saved_revenue_random = 36000

print(f"Random Strategy:      ${saved_revenue_random:,.0f}")
print(f"ML-Informed Strategy: ${saved_revenue_ml:,.0f}")
print(f"Improvement:          ${saved_revenue_ml - saved_revenue_random:,.0f} (+{((saved_revenue_ml/saved_revenue_random)-1)*100:.0f}%)")

```


#### Bonus Question: B2B vs B2C
GBI operates in a Business-to-Business (B2B) environment. Why is retaining existing customers particularly critical for B2B companies compared to B2C (Business-to-Consumer)?
- [ ] B2B customers always buy more products than B2C customers.
- [ ] Acquiring new B2B customers is significantly more expensive and time-consuming (longer sales cycles, complex negotiations). At the same time, B2B relationships tend to generate higher revenue over longer contract durations, making each individual customer far more valuable.
- [ ] B2B companies have fewer competitors, so customer retention is less important.
- [ ] B2C customers are generally more loyal than B2B customers, so B2B companies need to invest more in retention.



# Part II: Customer Segmentation
CLV Prediction is just one of many business analytics use cases where data-driven decision making can greatly improve business profitability. In this second part of the case study, we will explore **Customer Segmentation**.

Unlike CLV Prediction, which belongs to the **Supervised Learning** class, Customer Segmentation is an **Unsupervised Learning** technique. In unsupervised learning, the algorithm looks for hidden patterns in the data without having a predefined target variable to predict. Instead of predicting a specific outcome, the algorithm groups similar data points together based on their inherent characteristics (Clustering). 

*(Note: For a full taxonomy of AI and ML approaches, check the **Companion PDF**!)*

## Business Understanding, Data Understanding and Data Preparation
Fortunately for us, Customer Segmentation works on the exact same RFM data as our CLV prediction! This means we can leverage our previous work and skip the CRISP-DM phases for Business Understanding, Data Understanding, and Data Preparation, heading straight into the Modeling phase.

## Modeling
For clustering tabular data, **K-Means** is the most established and widely used algorithm. This makes the modeling phase much easier for us, as we don't have to test and compare a huge variety of algorithms like we did with AutoAI.



#### Exercise 13: Understand your Customers
Execute the code cell below to run the K-Means clustering algorithm on our prepared dataset. The code will generate a table summarizing the average RFM values for each cluster, as well as an interactive 3D plot.

**Your Task:** Look at the summary table and analyze the characteristics of Cluster 0, 1, 2, and 3. Then, complete the Python dictionary in the code by mapping each Cluster ID to one of the following meaningful business labels:

- `Lost Customers`
- `New Customers`
- `Potential Customers`
- `Core Customers`




```python
# ==========================================
# Phase 5: Customer Segmentation (K-Means & 3D Plot)
# ==========================================
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px

# 1. Select features
features = ['Recency', 'Frequency', 'Monetary']
X_seg = training_df[features].copy()

# 2. Scaling: We use the log transformation!
# This compresses the massive B2B "whales" so they don't dominate K-Means.
for col in X_seg.columns:
    X_seg[col] = np.log1p(X_seg[col])

# 3. Train K-Means model
print("Executing K-Means Clustering (K=4)...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
# We assign the cluster directly to the original DataFrame!
training_df['Cluster_ID'] = kmeans.fit_predict(X_seg).astype(str)

# 4. DIDACTICS: Translate math into business value!
# The mapping depends on the averages. With random_state=42 it remains constant.
# TASK: Map the clusters to the following categories: Core Customers, New Customers, Potential Customers, Lost Customers
label_mapping = {
    '0': '___',
    '1': '___',
    '2': '___',
    '3': '___'
}
training_df['Customer_Segment'] = training_df['Cluster_ID'].map(label_mapping)

# 5. Show Business Insights
segment_summary = training_df.groupby('Customer_Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
segment_summary['Num_Customers'] = training_df.groupby('Customer_Segment').size()
print("\nAverage values of the customer segments (The basis for labeling!):")
display(segment_summary)

# 6. Interactive 3D Visualization
# We filter out the top 1% outliers just for the plot so the clouds are more visible
plot_df = training_df[training_df['Monetary'] < training_df['Monetary'].quantile(0.99)]

fig = px.scatter_3d(
    plot_df, 
    x='Recency', 
    y='Frequency', 
    z='Monetary',
    color='Customer_Segment',
    opacity=0.6,
    title='3D Customer Segments (You can rotate the plot!)',
    color_discrete_sequence=px.colors.qualitative.Set1
)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
fig.show()

```

## Evaluation:

Unsupervised Learning algorithms require different performance metrics than supervised algorithms. Since we do not have a "correct" target variable to compare our predictions against, we must evaluate the structure of the clusters themselves.

Conceptually, a good clustering model optimizes two attributes:

1. **Cohesion:** Data points *within* the same cluster should be as similar to each other as possible. This is also called *intra-cluster distance*
2. **Separation:** The distinct clusters should be as different from each other as possible and measured through the *inter-cluster distance*

Since our customers are effectively represented as vectors in a 3-dimensional **feature space** (Recency, Frequency, Monetary), we can use mathematical metrics like the *Euclidean distance* to measure these relationships.

The **Silhouette Score** is an established metric that calculates both cohesion and separation and boils them down into a single numerical value ranging from **-1 to +1**:

- **+1** indicates perfect clusters that are tight and far apart from each other
- **0** indicates overlapping clusters where the boundaries are blurry
- **-1** indicates that the data points have likely been assigned to the wrong clusters

![Silhuette Score Examples](silhouette_examples.png)

#### Exercise 14: Evaluate the Silhouette Score
Execute the code cell below to calculate the Silhouette Score of our K-Means model.

**Your Task:** (Double-click to edit): Look at the calculated score. In real-world business datasets, a perfect score of +1 is basically impossible because customer behavior is messy and naturally overlaps. Keeping this in mind, how would you evaluate our model's performance? Does the score suggest that our segments (Core, New, Potential, Lost) are distinct enough to base marketing strategies on them?


```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X_seg, kmeans.labels_)
print(f"Silhouette Score: {score:.3f}")

```

# Deployment:
In the final step, the analytical results are integrated into the business processes. For Customer Segmentation, this usually means handing the segmented lists over to the Marketing and Sales departments so they can adjust their strategies.

#### Exercise 15: Strategic Business Decisions
Based on the clusters we identified, evaluate the following statements

**Question 15.1: The Efficiency Trade-off**
- [ ] While hyper-personalization is highly effective, it is often too expensive and inefficient to scale. Clustering provides the perfect middle ground between high relevance and cost efficiency.
- [ ] Customers do not like personalized campaigns due to privacy concerns.
- [ ] It is mathematically impossible to create personalized campaigns for individual customers.

**Question 15.2 Resource Allocation**
If Global Bike Inc has a limited marketing budget of $50,000 for a new loyalty program, which segment should they primarily target?
- [ ] The *Lost Customers*, to try and win them all back.
- [ ] The *Core Customers*, as they already generate the most revenue and are the most likely to respond positively to loyalty programs, ensuring a high ROI.
- [ ] The New Customers, because they haven't bought much yet.

**Question 15.3: Actionable Insights**
How should the Sales team approach the *Potential Customers* segment?
- [ ] They should ignore them, as they don't generate enough revenue yet.
- [ ] They should target them with up-selling campaigns (e.g., offering discounts on premium products), since these customers buy frequently but currently only have low Monetary values.
- [ ] They should call them immediately and ask why they haven't bought anything recently. 



# Congratulations!
You have successfully completed the Business Analytics Case Study!

Over the course of this notebook, you have navigated the entire CRISP-DM lifecycle. You started with messy raw data, explored it, engineered valuable RFM features, and trained both supervised (AutoAI) and unsupervised (K-Means) machine learning models. Most importantly, you learned how to translate abstract mathematical metrics into tangible business value.

We hope this case study gave you a practical, hands-on perspective on how modern companies leverage Data Science to drive decision-making. We wish you all the best for your future path as Business Analysts!

**One last thing:** Please do not forget to fill out the final evaluation survey! Your feedback is incredibly valuable for our research and helps us to improve this course for future students. 
You can find the evaluation form here: https://forms.cloud.microsoft/e/mZGaDaxnHy

## Troubleshooting: Local Model Fallback
If you encounter critical issues with the Watsonx.ai cloud environment (e.g. service outages, API key issues) and cannot execute the AutoAI pipeline, you can use the following local fallback to continue the case study.

This fallback uses a standard **Ridge Regression** algorithm from the `scikit-learn` library. It will train directly on your local machine instead of the cloud.

**Instructions:**
1. Skip the following cells: `!pip install ibm-watsonx-ai`, `Watsonx.ai Authentication`, `AutoAI Training`, and `Evaluate AutoAI results`.
2. Run the code cell below instead. It will create all the variables needed for the rest of the notebook.
3. After running this cell, continue the notebook normally from **Exercise 9** onwards.

**Note:** Since this fallback skips the RPT-1 comparison, you can also skip the RPT-1 code cell and Exercise 10.



```python
import numpy as np
from sklearn.linear_model import Ridge

# 1. Prepare data from the temporal split (Features & Target)
X = training_data[['Recency', 'Frequency', 'Monetary_Year1']]
y = training_data['CLV_Year2']

# 2. Train model (Local fallback instead of Watsonx)
ridge_model = Ridge(random_state=42)
ridge_model.fit(X, y)

# 3. INFERENCE: Generate prediction
predictions = ridge_model.predict(X)

# 4. Create the training_df (needed for Customer Segmentation)
training_df = training_data.drop(columns=['KUNNR'])
training_df.rename(columns={'Monetary_Year1': 'Monetary', 'CLV_Year2': 'Monetary_Target'}, inplace=True)

# 5. Build the results DataFrame (same as Exercise 9)
results = X.copy()
results['KUNNR'] = training_data['KUNNR']
results['Actual_CLV ($)'] = np.round(y, 2)
results['Predicted_CLV ($)'] = np.round(predictions, 2)
results['Absolute_Error ($)'] = np.round(abs(results['Actual_CLV ($)'] - results['Predicted_CLV ($)']), 2)
results.set_index('KUNNR', inplace=True)

print("Local Ridge Model trained successfully.")
print("\nTop 5 Customers (Highest CLV):")
display(results.sort_values(by='Predicted_CLV ($)', ascending=False).head(5))

print("\nBottom 5 Customers (Lowest CLV):")
display(results.sort_values(by='Predicted_CLV ($)', ascending=True).head(5))

```
