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
        /* AI Chat box styling */
        .ai-box { background-color: #1e293b; padding: 20px; border-radius: 10px; border-left: 5px solid #00E5FF; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING (Sales & Traffic)
# ==========================================
@st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=100)
    
    # 1. Mock Sales Data
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
    
    # 2. Mock Website Traffic Data
    traffic_data = []
    for d in dates:
        for source in ["Organic Search", "Direct", "Social Media", "Paid Ads"]:
            visitors = int(np.random.normal(800, 200))
            if visitors < 50: visitors = 50
            traffic_data.append({
                "Date": d,
                "Source": source,
                "Visitors": visitors,
                "Bounce_Rate": np.random.uniform(30, 75),
                "Conversions": int(visitors * np.random.uniform(0.01, 0.08)) # 1% to 8% conversion
            })
    traffic_df = pd.DataFrame(traffic_data)
    
    return df, traffic_df

raw_df, raw_traffic_df = load_data()

# ==========================================
# 3. PROFESSIONAL SIDEBAR (Nav & Filters)
# ==========================================
with st.sidebar:
    st.markdown("### 🛍️ E-Commerce BI")
    
    # 1. Navigation Menu in Sidebar
    selected = option_menu(
        menu_title=None, 
        options=["Executive Dashboard", "Web Traffic & SEO", "Customer Intelligence", "🤖 AI Data Analyst"], 
        icons=["bar-chart-fill", "globe", "people-fill", "robot"], 
        menu_icon="cast", 
        default_index=0,
        styles={
            "nav-link-selected": {"background-color": "#1A365D", "color": "white"},
            "nav-link": {"font-size": "14px", "margin":"5px 0px"}
        }
    )
    
    st.divider()
    
    # 2. Global Filters
    st.markdown("### 🔍 Global Filters")
    
    categories = raw_df['Category'].unique().tolist()
    selected_categories = st.multiselect("Select Categories", categories, default=categories)
    
    min_date = raw_df['Date'].min().date()
    max_date = raw_df['Date'].max().date()
    start_date, end_date = st.date_input("Select Date Range", [min_date, max_date])

# ==========================================
# 4. APPLY FILTERS TO DATA
# ==========================================
# Filter Sales Data
mask = (
    (raw_df['Category'].isin(selected_categories)) & 
    (raw_df['Date'].dt.date >= start_date) & 
    (raw_df['Date'].dt.date <= end_date)
)
df = raw_df[mask]

# Filter Traffic Data
traffic_mask = (
    (raw_traffic_df['Date'].dt.date >= start_date) & 
    (raw_traffic_df['Date'].dt.date <= end_date)
)
traffic_df = raw_traffic_df[traffic_mask]

# ==========================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==========================================
if selected == "Executive Dashboard":
    st.title("📊 Executive Sales Performance")
    
    if df.empty:
        st.warning("⚠️ No data available for the selected filters.")
    else:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Revenue", f"${df['Sales'].sum():,.0f}")
        kpi2.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
        kpi3.metric("Total Orders", f"{len(df)}")
        kpi4.metric("Avg Order Value", f"${df['Sales'].mean():,.2f}")
        
        st.markdown("---")
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

# ==========================================
# PAGE 2: WEB TRAFFIC & SEO
# ==========================================
elif selected == "Web Traffic & SEO":
    st.title("🌐 Website Traffic & Conversion Analytics")
    
    if traffic_df.empty:
        st.warning("⚠️ No traffic data available for the selected date range.")
    else:
        kpi1, kpi2, kpi3 = st.columns(3)
        total_visitors = traffic_df['Visitors'].sum()
        total_conversions = traffic_df['Conversions'].sum()
        avg_bounce = traffic_df['Bounce_Rate'].mean()
        
        kpi1.metric("Total Website Visitors", f"{total_visitors:,.0f}")
        kpi2.metric("Avg Bounce Rate", f"{avg_bounce:.1f}%")
        kpi3.metric("Global Conversion Rate", f"{(total_conversions/total_visitors)*100:.2f}%")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Traffic Sources Breakdown**")
            source_df = traffic_df.groupby("Source")["Visitors"].sum().reset_index()
            fig_pie = px.pie(source_df, names="Source", values="Visitors", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            st.markdown("**Daily Visitors by Source**")
            daily_traffic = traffic_df.groupby(["Date", "Source"])["Visitors"].sum().reset_index()
            fig_area = px.area(daily_traffic, x="Date", y="Visitors", color="Source")
            fig_area.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350)
            st.plotly_chart(fig_area, use_container_width=True)

# ==========================================
# PAGE 3: CUSTOMER INTELLIGENCE
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
            st.info(f"Analyzing {len(df['CustomerID'].unique())} unique customers.")
                
        with col2:
            rfm_data = df[['Recency', 'Frequency', 'Monetary']].dropna()
            if len(rfm_data) > clusters:
                kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
                df_clustered = df.copy()
                df_clustered['Segment'] = kmeans.fit_predict(rfm_data).astype(str)
                
                fig3 = px.scatter_3d(
                    df_clustered, x='Recency', y='Frequency', z='Monetary', 
                    color='Segment', color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=450)
                st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# PAGE 4: AI DATA ANALYST
# ==========================================
elif selected == "🤖 AI Data Analyst":
    st.title("🤖 Automated AI Business Insights")
    st.markdown("Your virtual analyst has reviewed the current filtered dataset and generated the following insights:")
    
    if df.empty or traffic_df.empty:
        st.warning("⚠️ Please adjust your filters. Not enough data to generate insights.")
    else:
        # Calculate Insights Dynamically
        top_category = df.groupby("Category")["Sales"].sum().idxmax()
        top_cat_sales = df.groupby("Category")["Sales"].sum().max()
        
        best_traffic_source = traffic_df.groupby("Source")["Conversions"].sum().idxmax()
        best_traffic_conv = traffic_df.groupby("Source")["Conversions"].sum().max()
        
        avg_order = df["Sales"].mean()
        
        best_day = df.groupby(df['Date'].dt.date)["Sales"].sum().idxmax()
        best_day_sales = df.groupby(df['Date'].dt.date)["Sales"].sum().max()

        # Display Insights like an AI Chat
        st.markdown(f"""
        <div class="ai-box">
            <h4 style="margin-top:0px; color: #f8fafc;">📦 Product Performance</h4>
            <p style="color: #cbd5e1; margin-bottom:0px;">Your highest performing category in this time period is <strong>{top_category}</strong>, bringing in a total of <strong>${top_cat_sales:,.2f}</strong> in revenue. Consider increasing marketing spend on this category to maximize returns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="ai-box">
            <h4 style="margin-top:0px; color: #f8fafc;">🎯 Traffic & Acquisition</h4>
            <p style="color: #cbd5e1; margin-bottom:0px;">The traffic source generating the most actual purchases (conversions) is <strong>{best_traffic_source}</strong> with <strong>{best_traffic_conv:,}</strong> successful conversions. Your average order value across all channels remains healthy at <strong>${avg_order:.2f}</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="ai-box">
            <h4 style="margin-top:0px; color: #f8fafc;">📈 Peak Sales Alert</h4>
            <p style="color: #cbd5e1; margin-bottom:0px;">The highest single day for revenue was <strong>{best_day}</strong>, generating <strong>${best_day_sales:,.2f}</strong>. You should cross-reference this date with any marketing campaigns or holidays to replicate this success.</p>
        </div>
        """, unsafe_allow_html=True)
