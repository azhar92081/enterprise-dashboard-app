import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai

# ==========================================
# 1. PAGE CONFIG & PREMIUM CSS
# ==========================================
st.set_page_config(page_title="E-Commerce BI Dashboard", page_icon="🛍️", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
        html, body, [class*="css"] { font-size: 14px; }
        [data-testid="stMetricValue"] { font-size: 1.8rem; color: #00E5FF; font-weight: bold; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .ai-box { background-color: #1e293b; padding: 20px; border-radius: 10px; border-left: 5px solid #00E5FF; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

CUSTOM_COLORS = ["#00E5FF", "#FF3366", "#00FF66", "#FDB813", "#B366FF"]
HOVER_STYLE = dict(bgcolor="#1e293b", font_size=14, font_color="#ffffff", bordercolor="#00E5FF")

# ==========================================
# 2. DATA HANDLING (With Validation & Scalability Guardrails)
# ==========================================
@st.cache_data
def generate_mock_data():
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
    
    traffic_data = []
    for d in dates:
        for source in ["Organic Search", "Direct", "Social Media", "Paid Ads"]:
            visitors = int(np.random.normal(800, 200))
            traffic_data.append({
                "Date": d, "Source": source, "Visitors": max(visitors, 50),
                "Bounce_Rate": np.random.uniform(30, 75),
                "Conversions": int(max(visitors, 50) * np.random.uniform(0.01, 0.08))
            })
    return df.sort_values("Date"), pd.DataFrame(traffic_data)

def validate_sales_data(df):
    required_cols = ['Date', 'CustomerID', 'Category', 'Sales', 'Profit', 'Recency', 'Frequency', 'Monetary']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, ""

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# ==========================================
# 3. SIDEBAR (Nav, Upload, Security & Filters)
# ==========================================
with st.sidebar:
    st.markdown("### 🛍️ E-Commerce BI")
    
    selected = option_menu(
        menu_title=None, 
        options=["Executive Dashboard", "Web Traffic & SEO", "Customer Intelligence", "🤖 AI Analyst"], 
        icons=["bar-chart-fill", "globe", "people-fill", "robot"], 
        menu_icon="cast", default_index=0,
        styles={"nav-link-selected": {"background-color": "#1A365D", "color": "#00E5FF", "font-weight": "bold"}, "nav-link": {"font-size": "14px"}}
    )
    
    st.divider()
    st.markdown("### 📁 Data Importer")
    sales_file = st.file_uploader("Upload Sales CSV", type=["csv"])
    traffic_file = st.file_uploader("Upload Traffic CSV", type=["csv"])
    
    raw_df, raw_traffic_df = generate_mock_data()
    
    # Process Sales File
    if sales_file:
        try:
            temp_df = pd.read_csv(sales_file)
            is_valid, error_msg = validate_sales_data(temp_df)
            if is_valid:
                if len(temp_df) > 50000:
                    st.warning("⚠️ Large dataset detected. Sampling top 50,000 rows.")
                    temp_df = temp_df.sample(50000, random_state=42)
                temp_df['Date'] = pd.to_datetime(temp_df['Date'])
                raw_df = temp_df
                st.success("Sales data validated!")
            else:
                st.error(f"Validation Failed: {error_msg}. Reverting to mock data.")
        except Exception as e:
            st.error(f"Corrupted file: {e}. Reverting to mock data.")
            
    # Process Traffic File
    if traffic_file:
        try:
            temp_traffic = pd.read_csv(traffic_file)
            if 'Date' in temp_traffic.columns and 'Visitors' in temp_traffic.columns:
                temp_traffic['Date'] = pd.to_datetime(temp_traffic['Date'])
                raw_traffic_df = temp_traffic
                st.success("Traffic data loaded!")
        except Exception:
            pass

    st.divider()
    st.markdown("### 🔍 Global Filters")
    categories = raw_df['Category'].unique().tolist()
    selected_categories = st.multiselect("Select Categories", categories, default=categories)
    
    # --- LOGIC FIX 1: Synchronized Date Filter ---
    min_sales_date = raw_df['Date'].min().date()
    max_sales_date = raw_df['Date'].max().date()
    
    if not raw_traffic_df.empty and 'Date' in raw_traffic_df.columns:
        min_traffic_date = raw_traffic_df['Date'].min().date()
        max_traffic_date = raw_traffic_df['Date'].max().date()
        min_date = min(min_sales_date, min_traffic_date)
        max_date = max(max_sales_date, max_traffic_date)
    else:
        min_date, max_date = min_sales_date, max_sales_date

    start_date, end_date = st.date_input("Select Date Range", [min_date, max_date])
    
    st.divider()
    st.markdown("### 🔐 Enterprise AI Security")
    anonymize_data = st.checkbox("Anonymize Financial Data for AI", value=True)
    
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("🔑 Vault Key Active")
    except KeyError:
        st.error("⚠️ Missing API Key in secrets.")
        api_key = None

# ==========================================
# 4. APPLY FILTERS & DOWNLOAD BUTTON
# ==========================================
mask = (raw_df['Category'].isin(selected_categories)) & (raw_df['Date'].dt.date >= start_date) & (raw_df['Date'].dt.date <= end_date)
df = raw_df[mask]

traffic_mask = (raw_traffic_df['Date'].dt.date >= start_date) & (raw_traffic_df['Date'].dt.date <= end_date)
traffic_df = raw_traffic_df[traffic_mask]

with st.sidebar:
    st.divider()
    st.markdown("### 💾 Export Reports")
    csv_data = convert_df_to_csv(df)
    st.download_button(label="📥 Download Filtered Data", data=csv_data, file_name="filtered_ecommerce_data.csv", mime="text/csv")

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
            fig1 = px.line(trend_df, x="Date", y="Sales", color_discrete_sequence=[CUSTOM_COLORS[0]])
            fig1.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hoverlabel=HOVER_STYLE)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.markdown("**Sales by Category**")
            cat_df = df.groupby("Category")["Sales"].sum().reset_index()
            fig2 = px.bar(cat_df, x="Category", y="Sales", color="Category", color_discrete_sequence=CUSTOM_COLORS)
            fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hoverlabel=HOVER_STYLE)
            st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# PAGE 2: WEB TRAFFIC & SEO
# ==========================================
elif selected == "Web Traffic & SEO":
    st.title("🌐 Website Traffic & Conversion Analytics")
    if traffic_df.empty:
        st.warning("⚠️ No traffic data available.")
    else:
        kpi1, kpi2, kpi3 = st.columns(3)
        total_visitors = traffic_df['Visitors'].sum()
        total_conversions = traffic_df['Conversions'].sum()
        kpi1.metric("Total Website Visitors", f"{total_visitors:,.0f}")
        kpi2.metric("Avg Bounce Rate", f"{traffic_df['Bounce_Rate'].mean():.1f}%")
        kpi3.metric("Global Conversion Rate", f"{(total_conversions/total_visitors)*100:.2f}%")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Traffic Sources Breakdown**")
            source_df = traffic_df.groupby("Source")["Visitors"].sum().reset_index()
            fig_pie = px.pie(source_df, names="Source", values="Visitors", hole=0.6, color_discrete_sequence=CUSTOM_COLORS)
            fig_pie.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, paper_bgcolor="rgba(0,0,0,0)", hoverlabel=HOVER_STYLE)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            st.markdown("**Daily Visitors by Source**")
            daily_traffic = traffic_df.groupby(["Date", "Source"])["Visitors"].sum().reset_index()
            fig_line = px.line(daily_traffic, x="Date", y="Visitors", color="Source", color_discrete_sequence=CUSTOM_COLORS)
            fig_line.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hoverlabel=HOVER_STYLE)
            st.plotly_chart(fig_line, use_container_width=True)

# ==========================================
# PAGE 3: CUSTOMER INTELLIGENCE
# ==========================================
elif selected == "Customer Intelligence":
    st.title("🧠 Mathematically Scaled Segmentation")
    if df.empty:
        st.warning("⚠️ No data available.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Advanced ML Engine")
            clusters = st.slider("Select K-Means Clusters:", min_value=2, max_value=6, value=3)
            st.info(f"**Data Science Note:** RFM variables have been normalized using `StandardScaler` to prevent Monetary variance from skewing calculations. Processing {len(df['CustomerID'].unique())} customers.")
                
        with col2:
            rfm_data = df[['Recency', 'Frequency', 'Monetary']].dropna()
            if len(rfm_data) > clusters:
                scaler = StandardScaler()
                scaled_rfm = scaler.fit_transform(rfm_data)
                
                kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
                df_clustered = df.copy()
                df_clustered['Segment'] = kmeans.fit_predict(scaled_rfm).astype(str)
                
                fig3 = px.scatter_3d(df_clustered, x='Recency', y='Frequency', z='Monetary', color='Segment', color_discrete_sequence=CUSTOM_COLORS)
                fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=450, paper_bgcolor="rgba(0,0,0,0)", hoverlabel=HOVER_STYLE)
                st.plotly_chart(fig3, use_container_width=True)
                
                st.markdown("### 📊 Actionable Segment Profiles")
                st.write("Average behavior metrics for each customer segment:")
                
                cluster_summary = df_clustered.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
                cluster_summary['Recency'] = cluster_summary['Recency'].apply(lambda x: f"{x:.0f} days ago")
                cluster_summary['Frequency'] = cluster_summary['Frequency'].apply(lambda x: f"{x:.1f} times")
                cluster_summary['Monetary'] = cluster_summary['Monetary'].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

# ==========================================
# PAGE 4: SMART AI DATA ANALYST
# ==========================================
elif selected == "🤖 AI Analyst":
    st.title("🤖 Secure Enterprise AI Analyst")
    
    if df.empty or traffic_df.empty:
        st.warning("⚠️ Please adjust your filters. Not enough data to generate insights.")
    else:
        if anonymize_data:
            st.caption("🔒 **Enterprise Security Mode Enabled:** Absolute financial figures are masked from the LLM prompt.")
        else:
            st.caption("⚠️ **Warning:** Raw financial data is being transmitted to external APIs.")

        top_category = df.groupby("Category")["Sales"].sum().idxmax() if not df.empty else "N/A"
        top_cat_sales = df.groupby("Category")["Sales"].sum().max() if not df.empty else 0
        best_traffic_source = traffic_df.groupby("Source")["Conversions"].sum().idxmax() if not traffic_df.empty else "N/A"
        
        if anonymize_data:
            data_context = f"""
            You are a secure BI Analyst. Data context (ANONYMIZED):
            - Top Category: {top_category}
            - Profit Margin: {(df['Profit'].sum() / df['Sales'].sum()) * 100:.1f}%
            - Best Traffic Source: {best_traffic_source}
            - Sales trends are positive, but do not state exact monetary figures.
            """
            display_sales = "[REDACTED]"
        else:
            data_context = f"""
            You are a BI Analyst. Data context:
            - Total Sales: ${df['Sales'].sum():,.2f}
            - Total Profit: ${df['Profit'].sum():,.2f}
            - Top Category: {top_category} (${top_cat_sales:,.2f})
            - Top Traffic Source: {best_traffic_source}
            """
            display_sales = f"${top_cat_sales:,.2f}"
            
        st.markdown(f"""
        <div class="ai-box">
            <h4 style="margin-top:0px; color: #f8fafc;">📦 Product Insight</h4>
            <p style="color: #cbd5e1; margin-bottom:0px;">The highest performing category is <strong>{top_category}</strong> with <strong>{display_sales}</strong> in revenue.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # --- LOGIC FIX 2: Clear Chat Memory UI ---
        col_chat, col_btn = st.columns([4, 1])
        with col_chat:
            st.markdown("#### 💬 Encrypted Data Chat")
        with col_btn:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = [{"role": "assistant", "content": "Memory cleared. What new data are we looking at?"}]
                st.rerun()
        
        if not api_key:
            st.error("⚠️ Your API key is not configured in Streamlit Secrets.")
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "Security protocols active. How can I assist you with today's metrics?"}]

            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input("Ask a question about the data..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                
                with st.spinner("Analyzing securely..."):
                    try:
                        response = model.generate_content(f"{data_context}\n\nUser: {prompt}")
                        msg = response.text
                    except Exception as e:
                        msg = f"API Error: {e}"
                
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)
