import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, precision_score,
    recall_score, f1_score, accuracy_score
)
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------------
# Section 1: Load & Clean Data
# -------------------------
st.title("Superstore Analysis & Churn Prediction")
st.header("1. Load & Clean Data")

@st.cache_data

def load_data():
    df = pd.read_csv("data/Superstore.csv", encoding='latin1')
    df.drop(columns='Row ID', inplace=True)
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.to_period("M").astype(str)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df['Shipping Duration'] = df['Ship Date'] - df['Order Date']
    df['Shipping Duration (Days)'] = df['Shipping Duration'].dt.days
    return df

df = load_data()
st.write(df.head())

# -------------------------
# Section 2: EDA & Visualizations
# -------------------------
st.header("2. Exploratory Data Analysis")

st.subheader("Sales vs Profit Over Time")
monthly_sales = df.groupby('Order Month')[['Sales','Profit']].sum().reset_index()
monthly_sales['Order Month'] = pd.to_datetime(monthly_sales['Order Month'])
fig1, ax1 = plt.subplots()
sns.lineplot(data=monthly_sales, x='Order Month', y='Sales', label='Sales', color='red', ax=ax1)
sns.lineplot(data=monthly_sales, x='Order Month', y='Profit', label='Profit', color='green', ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

st.subheader("Sub-Category-wise Sales & Profit")
subcat_group = df.groupby('Sub-Category')[['Sales', 'Profit']].sum().reset_index()
fig2 = px.bar(subcat_group, x='Sub-Category', y=['Sales', 'Profit'], barmode='group')
st.plotly_chart(fig2)

# -------------------------
# Section 3: Customer Segmentation
# -------------------------
st.header("3. Customer Segmentation")

customer_seg = df.groupby('Customer ID').agg(
    total_sales=('Sales', 'sum'),
    profit=('Profit', 'sum'),
    order_count=('Order ID', 'count')
).reset_index()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_seg[['total_sales', 'profit', 'order_count']])
kmeans = KMeans(n_clusters=6, random_state=42)
customer_seg['cluster'] = kmeans.fit_predict(customer_seg[['total_sales', 'profit', 'order_count']])

fig3, ax3 = plt.subplots()
sns.scatterplot(data=customer_seg, x='total_sales', y='profit', hue='cluster', palette='tab10', ax=ax3)
plt.title("Customer Clusters")
st.pyplot(fig3)

# -------------------------
# Section 4: Profit Prediction Model
# -------------------------
st.header("4. Profit Prediction Model")

features = ['Category', 'Sub-Category', 'Segment', 'Region', 'Quantity', 'Discount', 'Sales', 'Shipping Duration (Days)']
target = 'Profit'
model_df = df[features + [target]].copy()
model_df = pd.get_dummies(model_df, columns=['Category', 'Sub-Category', 'Segment', 'Region'], drop_first=True)

X = model_df.drop(columns=[target])
y = model_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

st.write("### XGBoost Regressor Performance")
st.write({
    "MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "R2 Score": r2
})

# -------------------------
# Section 5: Churn Prediction
# -------------------------
st.header("5. Churn Prediction")

X_months = 6
last_purchase = df.groupby("Customer ID")['Order Date'].max().reset_index()
max_date = df['Order Date'].max()
last_purchase['Months Since Last Purchase'] = ((max_date - last_purchase['Order Date'])/pd.Timedelta(days=30)).round()
last_purchase['Churned'] = last_purchase['Months Since Last Purchase'] > X_months
le = LabelEncoder()
last_purchase['Churned'] = le.fit_transform(last_purchase['Churned'])

X_churn = last_purchase[['Months Since Last Purchase']]
y_churn = last_purchase['Churned']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)

clf = RandomForestClassifier(max_depth=5, random_state=42)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

st.write("### Random Forest Classifier (Churn Prediction)")
st.write({
    "Accuracy": accuracy_score(y_test_c, y_pred_c),
    "Precision": precision_score(y_test_c, y_pred_c),
    "Recall": recall_score(y_test_c, y_pred_c),
    "F1 Score": f1_score(y_test_c, y_pred_c)
})

# -------------------------
# Section 6: View Source Code
# -------------------------
st.header("6. Source Code")
st.markdown("[View Full Project on GitHub](https://github.com/yourusername/superstore-churn-prediction)")
