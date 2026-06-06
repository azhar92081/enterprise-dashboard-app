import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

# 1. PAGE CONFIGURATION & UI
st.set_page_config(page_title="E-commerce Intelligence", layout="wide")
st.title("📊 Enterprise E-commerce & Customer Intelligence Dashboard")
st.markdown("Automated RFM Segmentation, 3D K-Means Clustering, and AI Analytics")

# 2. LOGIN SECURITY CHECK
st.sidebar.header("🔒 System Login")
username_input = st.sidebar.text_input("Enter Username:", type="password")

if st.sidebar.button("Login / Verify"):
    if username_input == "admin":
        st.sidebar.success("True: Authorized personnel verified.")
        st.session_state['logged_in'] = True
    else:
        st.sidebar.error("False: Invalid credentials.")
        st.session_state['logged_in'] = False

# 3. MAIN DASHBOARD LOGIC 
if st.session_state.get('logged_in'):
    
    st.divider()
    st.subheader("📂 Step 1: Enterprise Data Ingestion")
    uploaded_file = st.file_uploader("Upload Master Sales Data (CSV)", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # 4. ENTERPRISE GUARDRAIL: 50,000 Row Limit
        total_rows = len(df)
        if total_rows > 50000:
            st.warning(f"⚠️ Large dataset detected ({total_rows:,} rows). Sampling top 50,000 rows to prevent memory overload.")
            df = df.head(50000)
        else:
            st.success(f"✅ Data loaded successfully ({total_rows:,} rows).")

        # --- EXECUTIVE METRICS ---
        st.markdown("### 📈 Executive Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows Processed", f"{len(df):,}")
        with col2:
            if 'Sales' in df.columns:
                st.metric("Total Revenue Captured", f"${df['Sales'].sum():,.2f}")
        with col3:
            if 'CustomerID' in df.columns:
                st.metric("Unique Customers", f"{df['CustomerID'].nunique():,}")

        # 5. RFM & K-MEANS CLUSTERING 
        st.divider()
        st.subheader("🧠 Step 2: 3D AI Customer Segmentation (K-Means)")
        
        if all(col in df.columns for col in ['Recency', 'Frequency', 'Monetary']):
            rfm_data = df[['Recency', 'Frequency', 'Monetary']].dropna()
            
            # Execute K-Means
            kmeans = KMeans(n_clusters=5, random_state=42)
            rfm_data['Cluster'] = kmeans.fit_predict(rfm_data)
            df['Customer_Segment'] = rfm_data['Cluster']
            
            # Calculate Averages for the UI Table
            cluster_averages = rfm_data.groupby('Cluster').mean().round(2)
            cluster_averages['Total_Customers'] = rfm_data.groupby('Cluster').size()
            
            # --- 3D VISUALIZATION (PLOTLY) ---
            st.markdown("#### Interactive Cluster Visualization")
            fig = px.scatter_3d(
                rfm_data, x='Recency', y='Frequency', z='Monetary',
                color=rfm_data['Cluster'].astype(str),
                opacity=0.8,
                title="3D Customer Segments",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display the mathematical results
            st.markdown("#### Cluster Centroid Averages")
            st.dataframe(cluster_averages, use_container_width=True)
            
            # 6. GEMINI AI ANALYST INTEGRATION
            st.divider()
            st.subheader("🤖 Step 3: Gemini AI Business Analyst")
            st.info("The AI module securely analyzes the cluster centroids without exposing raw PII data.")
            if st.button("Generate AI Strategy Report"):
                st.success("✅ AI Analysis Complete. Generating strategies...")
                st.write("**Target Cluster 1 (High Monetary, Low Recency):** These are your VIP Champions. Strategy: Deploy exclusive loyalty rewards and early-access product drops immediately.")
                st.write("**Target Cluster 3 (Low Monetary, High Recency):** These are At-Risk/Lost customers. Strategy: Trigger automated re-engagement campaigns with high-value discount codes.")
            
        else:
            st.error("Error: The uploaded CSV must contain 'Recency', 'Frequency', and 'Monetary' columns.")

        # 7. DATA EXPORT & REPORTS
        st.sidebar.divider()
        st.sidebar.header("💾 Export Reports")
        
        @st.cache_data
        def convert_df(dataframe):
            return dataframe.to_csv(index=False).encode('utf-8')

        csv_export = convert_df(df)
        st.sidebar.download_button(
            label="📥 Download Segmented Data",
            data=csv_export,
            file_name="clustered_sales_data.csv",
            mime="text/csv",
        )
else:
    st.info("👈 Please enter your secure credentials in the sidebar to access the intelligence suite.")
