import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.cluster import KMeans
import pygwalker as pyg
import streamlit.components.v1 as components

# ==========================================
# 1. PAGE CONFIG & ENTERPRISE CSS
# ==========================================
st.set_page_config(page_title="E-Commerce Intelligence", page_icon="🛍️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        /* Compact UI: Reduce top padding */
        .block-container { padding-top: 2rem; padding-bottom: 0rem; }
        /* Shrink global font for a sleeker dashboard look */
        html, body, [class*="css"] { font-size: 14px; }
        /* Style the KPI metrics */
        [data-testid="stMetricValue"] { font-size: 1.8rem; color: #1A365D; }
        /* Hide default Streamlit menu and footer for production */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING (With Mock Data Backup)
# ==========================================
@st.cache_data
def load_data():
    # Buddy, replace the next line with your actual dataset!
    # df = pd.read_csv("your_ecommerce_data.csv")
    
    # Generating a realistic E-commerce dataset for presentation
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=100)
    df = pd.DataFrame({
        "Date": np.random.choice(dates, 500),
        "CustomerID": np.random.randint(1000, 1050, 500),
        "Category": np.random.choice(["Electronics", "Apparel", "Home", "Sports"], 500),
        "Sales": np.random.uniform(50, 500, 500),
        "Profit": np.random.uniform(5, 100, 500),
        "Recency": np.random.randint(1, 60, 500),
        "Frequency": np.random.randint(1, 10, 500),
        "Monetary": np.random.uniform(100, 2000, 500)
    })
    df = df.sort_values("Date")
    return df

df = load_data()

# ==========================================
# 3. RESPONSIVE MOBILE-FRIENDLY NAVIGATION
# ==========================================
selected = option_menu(
    menu_title=None, 
    options=["Executive Dashboard", "Customer Intelligence", "Advanced Data Mining"], 
    icons=["bar-chart-fill", "people-fill", "cpu"], 
    menu_icon="cast", 
    default_index=0, 
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f8f9fa", "margin-bottom": "20px"},
        "icon": {"color": "#00E5FF", "font-size": "16px"}, 
        "nav-link": {"font-size": "14px", "text-align": "center", "margin":"0px", "--hover-color": "#e2e8f0"},
        "nav-link-selected": {"background-color": "#1A365D", "color": "white"},
    }
)

# ==========================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==========================================
if selected == "Executive Dashboard":
    st.subheader("📊 Executive Sales Performance")
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Revenue", f"${df['Sales'].sum():,.0f}")
    kpi2.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
    kpi3.metric("Total Orders", f"{len(df)}")
    kpi4.metric("Avg Order Value", f"${df['Sales'].mean():,.2f}")
    
    st.markdown("---")
    
    # Perfectly Aligned Tight Columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Revenue Trend Over Time**")
        trend_df = df.groupby("Date")["Sales"].sum().reset_index()
        fig1 = px.area(trend_df, x="Date", y="Sales", color_discrete_sequence=["#00E5FF"])
        # Margin hack for compact UI
        fig1.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.markdown("**Sales by Category**")
        cat_df = df.groupby("Category")["Sales"].sum().reset_index()
        fig2 = px.bar(cat_df, x="Category", y="Sales", color="Category", template="plotly_white")
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# PAGE 2: CUSTOMER INTELLIGENCE (RFM & ML)
# ==========================================
elif selected == "Customer Intelligence":
    st.subheader("🧠 Customer Segmentation & RFM Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Machine Learning Engine**")
        st.info("Utilizing K-Means Clustering to identify customer purchasing patterns.")
        clusters = st.slider("Select number of customer segments:", min_value=2, max_value=6, value=3)
        
        if st.button("Run Clustering Model"):
            st.success(f"Algorithm applied! Customers grouped into {clusters} distinct segments.")
            
    with col2:
        # Applying K-Means to the mock RFM data
        rfm_data = df[['Recency', 'Frequency', 'Monetary']].copy()
        kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
        df['Segment'] = kmeans.fit_predict(rfm_data)
        
        fig3 = px.scatter_3d(df, x='Recency', y='Frequency', z='Monetary', color='Segment', 
                             title="3D Customer Segmentation Space",
                             color_continuous_scale=px.colors.sequential.Viridis)
        fig3.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# PAGE 3: ADVANCED DATA MINING (RapidMiner Style)
# ==========================================
elif selected == "Advanced Data Mining":
    st.subheader("⚙️ Exploratory Data Mining Module")
    st.markdown("Use the interface below to drag, drop, and build custom predictive visualizations on the fly, similar to RapidMiner and Tableau.")
    
    # PyGWalker integration for dynamic data manipulation
    pyg_html = pyg.to_html(df)
    components.html(pyg_html, height=800, scrolling=True)
