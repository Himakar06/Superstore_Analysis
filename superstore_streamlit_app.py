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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Superstore Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('Superstore.csv', encoding='latin1')
    # Data Cleaning
    df.drop(columns='Row ID', inplace=True)
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.to_period("M").astype('str')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df['Ship Year'] = df['Ship Date'].dt.year
    df['Ship Month'] = df['Ship Date'].dt.to_period("M")
    df['Shipping Duration'] = df['Ship Date'] - df['Order Date']
    return df

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Home", 
    "Exploratory Analysis", 
    "Customer Segmentation", 
    "Profit Prediction", 
    "Churn Analysis",
    "Interactive Visualizations"
])

# Home Page
if section == "Home":
    st.title("Superstore Analytics Dashboard")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71", width=800)
    st.markdown("""
    ### Comprehensive analysis of Superstore sales data
    This dashboard provides insights into:
    - Sales and profit trends
    - Product performance
    - Customer segmentation
    - Predictive modeling
    - Churn analysis
    """)
    
    st.dataframe(df.head())

# Exploratory Analysis
elif section == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Sales Trends", 
        "Category Analysis", 
        "Shipping Analysis", 
        "Customer Analysis"
    ])
    
    with tab1:
        st.header("Sales vs Profit Over Time")
        monthly_sales = df.groupby('Order Month')[['Sales','Profit']].sum().reset_index()
        monthly_sales['Order Month'] = pd.to_datetime(monthly_sales['Order Month'])
        
        fig, ax = plt.subplots(figsize=(10,6))
        sns.lineplot(x='Order Month', y='Sales', data=monthly_sales, label='Sales', color='red', ax=ax)
        sns.lineplot(x='Order Month', y='Profit', data=monthly_sales, label='Profit', color='green', ax=ax)
        ax.set_title("Sales vs Profit over time")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    with tab2:
        st.header("Category-wise Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(x='Category', y='Sales', data=df, ax=ax)
            ax.set_title('Category-wise Sales')
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(x='Category', y='Profit', data=df, ax=ax)
            ax.set_title('Category-wise Profit')
            st.pyplot(fig)
            
        st.header("Sub-Category Performance")
        fig = px.bar(
            df.groupby('Sub-Category')[['Sales', 'Profit']].sum().reset_index(),
            x='Sub-Category',
            y=['Sales', 'Profit'],
            barmode='group',
            title='Sub-Category-wise Sales & Profit'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.header("Shipping Mode Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            ship_counts = df['Ship Mode'].value_counts().reset_index()
            ship_counts.columns = ['Ship Mode', 'Count']
            fig = px.bar(ship_counts, x='Ship Mode', y='Count', title='Most Used Shipping Modes')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            df['Shipping Duration (Days)'] = df['Shipping Duration'].dt.days
            ship_duration = df.groupby('Ship Mode')['Shipping Duration (Days)'].mean().reset_index()
            fig = px.bar(ship_duration, x='Ship Mode', y='Shipping Duration (Days)', title='Average Shipping Duration')
            st.plotly_chart(fig, use_container_width=True)
            
    with tab4:
        st.header("Customer Analysis")
        
        customer_order = df['Customer ID'].value_counts()
        one_time = (customer_order == 1).sum()
        repeat = (customer_order > 1).sum()
        
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(
            [repeat, one_time],
            labels=['Repeat', 'One-time'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['green', 'red']
        )
        ax.set_title("Repeat vs One-time Customers")
        st.pyplot(fig)
        
        st.header("Top 10 Customers by Sales")
        clv = df.groupby("Customer ID").agg(
            total_sales=("Sales", 'sum'),
            total_orders=('Order ID', 'nunique'),
            total_Profit=('Profit', 'sum'),
            avg_sale=("Sales", 'mean')
        ).reset_index().sort_values(by='total_sales', ascending=False).head(10)
        
        st.dataframe(clv.style.background_gradient(cmap='Blues'))

# Customer Segmentation
elif section == "Customer Segmentation":
    st.title("Customer Segmentation Analysis")
    
    # Prepare data
    customer_seg = df.groupby("Customer ID").agg(
        total_sales=('Sales', 'sum'),
        profit=('Profit', 'sum'),
        order_count=('Order ID', 'count')
    ).reset_index()
    
    # Normalize features
    cols = ['total_sales', 'profit', 'order_count']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(customer_seg[cols])
    
    # KMeans Clustering
    kmeans = KMeans(n_clusters=6, random_state=42)
    customer_seg['cluster'] = kmeans.fit_predict(customer_seg[cols])
    
    # Label segments
    cluster_label = {
        0: 'Potential-Value',
        1: 'High-Value',
        2: 'Mid-Value',
        3: 'Mid-Value',
        4: 'Low-Value',
        5: 'Low-Value'
    }
    customer_seg['segment_label'] = customer_seg['cluster'].map(cluster_label)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Customer Segments Distribution")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.countplot(
            data=customer_seg,
            x='segment_label',
            order=['Potential-Value', 'High-Value', 'Mid-Value', 'Low-Value'],
            ax=ax
        )
        ax.set_title("Customer Segmentation Distribution")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    with col2:
        st.header("Segmentation Visualization")
        pca = PCA(n_components=2)
        X = customer_seg[['total_sales', 'order_count']]
        X_pca = pca.fit_transform(X)
        
        fig, ax = plt.subplots(figsize=(8,6))
        colors = {
            'Potential-Value': 'red',
            'High-Value': 'blue',
            'Mid-Value': 'orange',
            'Low-Value': 'green'
        }
        
        for label in customer_seg['segment_label'].unique():
            indices = customer_seg['segment_label'] == label
            ax.scatter(
                X_pca[indices, 0],
                X_pca[indices, 1],
                c=colors[label],
                label=label,
                s=50
            )
            
        ax.set_title("Customer Segmentation by PCA")
        ax.legend()
        st.pyplot(fig)
    
    st.header("Segment Characteristics")
    st.dataframe(
        customer_seg.groupby('segment_label')[['total_sales', 'profit', 'order_count']]
        .mean()
        .style.background_gradient(cmap='YlOrBr')
    )

# Profit Prediction
elif section == "Profit Prediction":
    st.title("Profit Prediction Model")
    
    # Prepare data
    features = ['Category', 'Sub-Category', 'Segment', 'Region', 'Quantity', 'Discount', 'Sales', 'Shipping Duration']
    target = 'Profit'
    model_df = df[features + [target]].copy()
    model_df['Shipping Duration'] = model_df['Shipping Duration'].dt.days
    model_df = pd.get_dummies(model_df, columns=['Category', 'Sub-Category', 'Segment', 'Region'], drop_first=True)
    
    X = model_df.drop(columns=[target])
    y = model_df[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"${mae:,.2f}")
    col2.metric("MSE", f"${mse:,.2f}")
    col3.metric("RMSE", f"${rmse:,.2f}")
    col4.metric("R2 Score", f"{r2:.2%}")
    
    # Feature Importance
    st.header("Feature Importance")
    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis', ax=ax)
    ax.set_title("Top 10 Important Features for Profit Prediction")
    st.pyplot(fig)

# Churn Analysis
elif section == "Churn Analysis":
    st.title("Customer Churn Analysis")
    
    # Define churned customers (no purchase in last X months)
    x_months = st.slider("Define churn threshold (months)", 1, 12, 6)
    
    last_purchase = df.groupby("Customer ID")['Order Date'].max().reset_index()
    last_purchase.columns = ['Customer ID', 'Last Purchase Date']
    max_date = df['Order Date'].max()
    
    last_purchase['Months Since Last Purchase'] = ((max_date - last_purchase['Last Purchase Date'])/pd.Timedelta(days=30)).round()
    last_purchase['Churned'] = (last_purchase['Months Since Last Purchase'] > x_months).astype(int)
    
    st.header(f"Churn Statistics (>{x_months} months inactivity)")
    churn_rate = last_purchase['Churned'].mean()
    st.metric("Churn Rate", f"{churn_rate:.1%}")
    
    # Model training
    X = last_purchase[['Months Since Last Purchase']]
    y = last_purchase['Churned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.1%}")
    col2.metric("Precision", f"{precision:.1%}")
    col3.metric("Recall", f"{recall:.1%}")
    col4.metric("F1 Score", f"{f1:.1%}")
    
    # Confusion Matrix
    st.header("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Churn Prediction Confusion Matrix")
    st.pyplot(fig)

# Interactive Visualizations
elif section == "Interactive Visualizations":
    st.title("Interactive Visualizations")
    
    tab1, tab2, tab3 = st.tabs([
        "Geographical Analysis", 
        "Product Bundling", 
        "Loss Analysis"
    ])
    
    with tab1:
        st.header("Region-wise Performance")
        reg_performance = df.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
        fig = px.bar(
            reg_performance,
            x='Region',
            y='Sales',
            color='Category',
            barmode='group',
            title='Region-wise Sales by Category'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.header("Product Bundling Analysis")
        try:
            basket = df.groupby(['Order ID', 'Product Name'])['Quantity'].sum().unstack().fillna(0)
            basket = basket.applymap(lambda x: 1 if x > 0 else 0)
            
            freq_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
            if not freq_itemsets.empty:
                rules = association_rules(freq_itemsets, metric='lift', min_threshold=1)
                st.dataframe(
                    rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                    .sort_values('confidence', ascending=False)
                    .head(10)
                )
            else:
                st.warning("No association rules found with current parameters")
        except:
            st.warning("Could not perform market basket analysis - try adjusting parameters")
            
    with tab3:
        st.header("Loss-Making Products Analysis")
        loss_products = df[df['Profit'] < 0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Most Frequent Loss-Makers")
            fig = px.bar(
                loss_products['Product Name'].value_counts().head(10).reset_index(),
                x='count',
                y='Product Name',
                orientation='h',
                title='Top 10 Loss-Making Products (Frequency)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Highest Total Loss")
            fig = px.bar(
                loss_products.groupby('Product Name')['Profit'].sum().sort_values().head(10).reset_index(),
                x='Profit',
                y='Product Name',
                orientation='h',
                title='Top 10 Loss-Making Products (Total Loss)'
            )
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Superstore Analytics Dashboard**  
Developed with Streamlit  
Data Source: Superstore Sales Dataset
""")
