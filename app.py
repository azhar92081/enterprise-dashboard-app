import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.cluster import KMeans

# ==========================================
# 1. PAGE CONFIG & ENTERPRISE CSS
# ==========================================
st.set_page_config(page_title="E-Commerce BI Dashboard", page_icon="🛍️", layout="wide")

st.markdown("""
    <style>
        /* Compact UI: Reduce top padding and tighten gaps */
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
        html, body, [class*="css"] { font-size: 14px; }
        /* Style the KPI metrics to look premium */
        [data-testid="stMetricValue"] { font-size: 1.8rem; color: #00E5FF; }
        /* Hide default Streamlit clutter */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    # ⚠️ BUDDY: Replace the mock data below with your actual dataset!
    # return pd.read_csv("your_actual_data.csv")
    
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
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")
    return df

raw_df = load_data()

# ==========================================
# 3. PROFESSIONAL SIDEBAR (Nav & Filters)
# ==========================================
with st.sidebar:
    st.markdown("### 🛍️ E-Commerce BI")
    
    # 1. Navigation Menu in Sidebar
    selected = option_menu(
        menu_title=None, 
        options=["Executive Dashboard", "Customer Intelligence"], 
        icons=["bar-chart-fill", "people-fill"], 
        menu_icon="cast", 
        default_index=0,
        styles={
            "nav-link-selected": {"background-color": "#1A365D", "color": "white"},
        }
    )
    
    st.divider()
    
    # 2. Global Filters
    st.markdown("### 🔍 Global Filters")
    
    # Category Filter
    categories = raw_df['Category'].unique().tolist()
    selected_categories = st.multiselect("Select Categories", categories, default=categories)
    
    # Date Filter
    min_date = raw_df['Date'].min().date()
    max_date = raw_df['Date'].max().date()
    start_date, end_date = st.date_input("Select Date Range", [min_date, max_date])

# ==========================================
# 4. APPLY FILTERS TO DATA
# ==========================================
# This ensures the whole app updates when sidebar filters are changed
mask = (
    (raw_df['Category'].isin(selected_categories)) & 
    (raw_df['Date'].dt.date >= start_date) & 
    (raw_df['Date'].dt.date <= end_date)
)
df = raw_df[mask]

# ==========================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==========================================
if selected == "Executive Dashboard":
    st.title("📊 Executive Sales Performance")
    
    if df.empty:
        st.warning("⚠️ No data available for the selected filters. Please adjust the sidebar filters.")
    else:
        # KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Revenue", f"${df['Sales'].sum():,.0f}")
        kpi2.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
        kpi3.metric("Total Orders", f"{len(df)}")
        kpi4.metric("Avg Order Value", f"${df['Sales'].mean():,.2f}")
        
        st.markdown("---")
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Revenue Trend Over Time**")
            trend_df = df.groupby("Date")["Sales"].sum().reset_index()
            fig1 = px.line(trend_df, x="Date", y="Sales", color_discrete_sequence=["#00E5FF"])
            fig1.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.markdown("**Sales by Category**")
            cat_df = df.groupby("Category")["Sales"].sum().reset_index()
            fig2 = px.bar(cat_df, x="Category", y="Sales", color="Category")
            fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            
        # Data Table View
        st.markdown("---")
        st.markdown("**Recent Transactions**")
        st.dataframe(df[['Date', 'CustomerID', 'Category', 'Sales', 'Profit']].tail(10), use_container_width=True)

# ==========================================
# PAGE 2: CUSTOMER INTELLIGENCE (Lightweight ML)
# ==========================================
elif selected == "Customer Intelligence":
    st.title("🧠 Customer Segmentation")
    
    if df.empty:
        st.warning("⚠️ No data available for the selected filters.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### AI Segmentation Engine")
            st.write("Group customers based on their buying behavior (Recency, Frequency, Monetary value).")
            clusters = st.slider("Select number of customer segments:", min_value=2, max_value=5, value=3)
            
            # Simple summary of the data being used
            st.info(f"Analyzing {len(df['CustomerID'].unique())} unique customers across {clusters} segments.")
                
        with col2:
            # Lightweight K-Means Clustering
            rfm_data = df[['Recency', 'Frequency', 'Monetary']].dropna()
            if len(rfm_data) > clusters:
                kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
                df_clustered = df.copy()
                df_clustered['Segment'] = kmeans.fit_predict(rfm_data)
                df_clustered['Segment'] = df_clustered['Segment'].astype(str) # Convert to string for discrete colors
                
                fig3 = px.scatter_3d(
                    df_clustered, x='Recency', y='Frequency', z='Monetary', 
                    color='Segment',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=450)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.error("Not enough data to perform clustering. Adjust filters.")
