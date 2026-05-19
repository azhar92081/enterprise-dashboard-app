import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import google.generativeai as genai

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="E-Commerce BI Dashboard", page_icon="📊", layout="wide")
st.title("🛍️ E-Commerce Intelligence & Data Mining Dashboard")

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. SIDEBAR & DATA INGESTION
# ==========================================
st.sidebar.header("⚙️ Command Center")

# API Key & Security
api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password")
privacy_mode = st.sidebar.toggle("🔒 Enterprise Privacy Mode", value=True, 
                                 help="Masks exact financial figures before sending to AI.")

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Upload Datasets")
sales_file = st.sidebar.file_uploader("1. Upload Sales Data (CSV)", type=["csv"])
traffic_file = st.sidebar.file_uploader("2. Upload Web Traffic (CSV)", type=["csv"])

# ==========================================
# 3. CORE FUNCTIONS (With Caching & Guardrails)
# ==========================================
@st.cache_data
def load_and_validate_sales(file):
    df = pd.read_csv(file)
    required_cols = ['Date', 'CustomerID', 'Category', 'Sales', 'Profit', 'Recency', 'Frequency', 'Monetary']
    
    # Validation
    if not all(col in df.columns for col in required_cols):
        st.sidebar.error("❌ Invalid Schema! Missing required columns.")
        return None
        
    # Scalability Guardrail (OOM Protection)
    if len(df) > 50000:
        st.sidebar.warning(f"⚠️ Dataset too large ({len(df)} rows). Downsampling to 50,000 to prevent memory crash.")
        df = df.sample(50000, random_state=42)
        
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def load_traffic(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def calculate_optimal_kmeans(df):
    features = ['Recency', 'Frequency', 'Monetary']
    X = df[features].copy()
    
    # Mathematical Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Silhouette Evaluation Loop
    best_k = 3
    best_score = -1
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
            
    # Apply Best K
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['Segment'] = final_model.fit_predict(X_scaled).astype(str)
    
    # Map Segment Names for better UI
    segment_map = {'0': 'High-Value Loyalists', '1': 'At-Risk Customers', '2': 'Recent/Low-Spend', '3': 'Mid-Tier', '4': 'Others'}
    df['Segment_Name'] = df['Segment'].map(segment_map).fillna(f"Cluster {df['Segment']}")
    
    return df, best_k, best_score

# ==========================================
# 4. DASHBOARD TABS
# ==========================================
if sales_file and traffic_file:
    df_sales = load_and_validate_sales(sales_file)
    df_traffic = load_traffic(traffic_file)
    
    if df_sales is not None:
        st.sidebar.success("✅ Data Ingested & Validated!")
        
        # Date Filter
        min_date, max_date = df_sales['Date'].min(), df_sales['Date'].max()
        selected_dates = st.sidebar.date_input("Filter by Date Range:", [min_date, max_date])
        
        if len(selected_dates) == 2:
            mask = (df_sales['Date'] >= pd.to_datetime(selected_dates[0])) & (df_sales['Date'] <= pd.to_datetime(selected_dates[1]))
            df_filtered = df_sales.loc[mask]
        else:
            df_filtered = df_sales

        # Run ML Clustering
        df_ml, optimal_k, sil_score = calculate_optimal_kmeans(df_filtered)

        # Build Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Executive KPIs", "🌐 Web Traffic", "🧠 ML Intelligence", "🤖 AI Analyst"])

        # --- TAB 1: EXECUTIVE KPIs ---
        with tab1:
            st.header("Executive Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Revenue", f"${df_filtered['Sales'].sum():,.2f}")
            col2.metric("Net Profit", f"${df_filtered['Profit'].sum():,.2f}")
            col3.metric("Total Orders", len(df_filtered))
            col4.metric("Avg Order Value", f"${df_filtered['Sales'].mean():,.2f}")
            
            fig_sales = px.bar(df_filtered.groupby('Category')['Sales'].sum().reset_index(), 
                               x='Category', y='Sales', title="Revenue by Category", color='Category')
            st.plotly_chart(fig_sales, use_container_width=True)

        # --- TAB 2: WEB TRAFFIC ---
        with tab2:
            st.header("SEO & Traffic Conversion")
            col1, col2 = st.columns(2)
            fig_traffic = px.pie(df_traffic, values='Visitors', names='Source', title="Traffic Sources", hole=0.4)
            col1.plotly_chart(fig_traffic, use_container_width=True)
            
            fig_conv = px.line(df_traffic, x='Date', y='Conversions', color='Source', title="Conversions Over Time")
            col2.plotly_chart(fig_conv, use_container_width=True)

        # --- TAB 3: ML INTELLIGENCE ---
        with tab3:
            st.header("Customer Segmentation (K-Means)")
            st.success(f"🤖 Auto-Scaled to **{optimal_k} Clusters** based on a Silhouette Score of **{sil_score:.2f}**.")
            
            fig_3d = px.scatter_3d(df_ml, x='Recency', y='Frequency', z='Monetary',
                                   color='Segment_Name', title="3D RFM Customer Clusters",
                                   opacity=0.7, size_max=10)
            st.plotly_chart(fig_3d, use_container_width=True)

        # --- TAB 4: AI ANALYST (THE UPGRADE) ---
        with tab4:
            st.header("💬 Prescriptive AI Analyst")
            st.markdown("Ask natural language questions about your business data. The AI will provide **structured, visual insights**.")
            
            if st.button("🗑️ Clear Chat Memory"):
                st.session_state.messages = []
                st.rerun()

            # Prepare Data Context for AI
            total_sales = df_filtered['Sales'].sum()
            top_category = df_filtered.groupby('Category')['Sales'].sum().idxmax()
            
            if privacy_mode:
                context_string = f"Data Summary: Total rows={len(df_filtered)}. Top Category is {top_category}. Exact financials are [REDACTED FOR ENTERPRISE PRIVACY]. Talk in percentages and trends."
            else:
                context_string = f"Data Summary: Total Sales=${total_sales:,.2f}. Total rows={len(df_filtered)}. Top Category is {top_category}."

            # Display Chat History
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Handle New Input
            if prompt := st.chat_input("E.g., 'What are my most profitable categories? Use a table.'"):
                if not api_key:
                    st.error("⚠️ Please enter your Gemini API Key in the sidebar first!")
                else:
                    # Append User Message
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Call Gemini
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing data..."):
                            try:
                                genai.configure(api_key=api_key)
                                model = genai.GenerativeModel('gemini-2.5-flash')
                                
                                # Strict Prompt Engineering for Visual Output
                                system_instruction = f"""
                                You are an expert E-Commerce Business Analyst. 
                                Context: {context_string}.
                                RULES FOR YOUR RESPONSE:
                                1. ALWAYS format your data comparisons using Markdown Tables.
                                2. Use bold text to highlight Key Performance Indicators (KPIs).
                                3. Use relevant emojis (📈, 💰, ⚠️, 🏆) to make the text visually appealing and easy to scan.
                                4. Keep insights sharp, prescriptive, and professional.
                                """
                                
                                response = model.generate_content(f"{system_instruction}\n\nUser Question: {prompt}")
                                st.markdown(response.text)
                                st.session_state.messages.append({"role": "assistant", "content": response.text})
                            except Exception as e:
                                st.error(f"API Error: {e}")
else:
    st.info("👋 Welcome! Please upload both the Sales and Traffic CSV files in the sidebar to generate the dashboard.")
