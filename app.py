import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import hashlib
import os
import base64
import google.generativeai as genai

# --- PAGE CONFIG ---
st.set_page_config(page_title="Enterprise Intelligence Dashboard", layout="wide", page_icon="logo.png", initial_sidebar_state="expanded")

# --- AUTOMATED PWA MANIFEST & ICON INJECTOR ---
# This automatically reads your logo.png and builds the files Windows needs to make a desktop app.
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

logo_b64 = get_base64_image("logo.png")

if logo_b64:
    manifest_json = f'''
    {{
        "name": "Enterprise Intelligence Dashboard",
        "short_name": "EI Dashboard",
        "display": "standalone",
        "theme_color": "#0E1117",
        "background_color": "#0E1117",
        "icons": [
            {{
                "src": "data:image/png;base64,{logo_b64}",
                "sizes": "192x192",
                "type": "image/png"
            }},
            {{
                "src": "data:image/png;base64,{logo_b64}",
                "sizes": "512x512",
                "type": "image/png"
            }}
        ]
    }}
    '''
    manifest_b64 = base64.b64encode(manifest_json.encode('utf-8')).decode()
    st.markdown(f"""
        <link rel="apple-touch-icon" href="data:image/png;base64,{logo_b64}">
        <link rel="manifest" href="data:application/manifest+json;base64,{manifest_b64}">
    """, unsafe_allow_html=True)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    return f"rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, {alpha})"

# --- ROBUST AUTO-PROVISIONING ---
@st.cache_resource
def auto_provision_db_v3():
    conn = sqlite3.connect('enterprise_backend.db', timeout=15)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, role TEXT NOT NULL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS system_alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, alert_type TEXT NOT NULL, message TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    cursor.execute("SELECT * FROM users WHERE username='admin'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                       ('admin', hash_password("iub2026"), 'System Administrator'))
        conn.commit()
    conn.close()

auto_provision_db_v3()

# --- SIDEBAR & THEME SETTINGS ---
st.sidebar.header("⚙️ System Settings")
night_mode = st.sidebar.toggle("🌙 Enable Night Mode", value=True)

if night_mode:
    bg_color = "#0E1117" 
    card_bg = "#161B22"
    border_color = "#30363D"
    text_color = "#E5E7EB"
    accent_color = "#00E5FF"
    plotly_template = "plotly_dark"
    chart_palette = ["#00E5FF", "#FF007F", "#FFD60A", "#8A2BE2", "#00F5D4", "#FF4D00"]
    faded_palette = ["#1A4B5C", "#4A1532"] 
else:
    bg_color = "#F8FAFC"
    card_bg = "#FFFFFF"
    border_color = "#E2E8F0"
    text_color = "#0F172A"
    accent_color = "#2563EB"
    plotly_template = "plotly_white"
    chart_palette = ["#2563EB", "#DC2626", "#D97706", "#7C3AED", "#059669", "#EA580C"] 
    faded_palette = ["#A9C2EB", "#F0B8B8"]
    
# --- ULTRA-RESPONSIVE UI CSS WITH TRUNCATION & CHAT FIXES ---
theme_css = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
    .stApp {{ background-color: {bg_color}; }}
    
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    .block-container {{ padding-top: 2rem !important; padding-bottom: 2rem !important; }}

    h1 {{ font-size: clamp(1.6rem, 4vw, 2.5rem) !important; line-height: 1.2 !important; color: {accent_color} !important; font-weight: 800 !important; padding-bottom: 10px; }}
    h3 {{ font-size: clamp(1.2rem, 2.5vw, 1.5rem) !important; color: {text_color} !important; margin-top: 15px; }}

    .pulse-container {{ display: flex; align-items: center; gap: 10px; margin-bottom: 25px; padding: 10px 15px; background: {card_bg}; border: 1px solid {border_color}; border-radius: 8px; width: fit-content; max-width: 100%; }}
    .pulse-dot {{ width: 12px; height: 12px; background-color: #10B981; border-radius: 50%; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); animation: pulse 1.5s infinite; flex-shrink: 0; }}
    @keyframes pulse {{ 0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }} 70% {{ transform: scale(1); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }} 100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }} }}
    .pulse-text {{ font-size: 0.9rem; font-weight: 600; color: {text_color}; letter-spacing: 0.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

    div[data-testid="metric-container"] {{ background-color: {card_bg}; border: 1px solid {border_color}; padding: clamp(16px, 2vw, 24px); border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: transform 0.2s ease, border-color 0.2s ease; height: 100%; display: flex; flex-direction: column; justify-content: center; }}
    div[data-testid="metric-container"]:hover {{ transform: translateY(-5px); border-color: {accent_color}; }}
    div[data-testid="stMetricLabel"] {{ color: {text_color} !important; font-size: clamp(0.85rem, 1.5vw, 1rem) !important; }}
    
    div[data-testid="stMetricValue"] * {{ color: {text_color} !important; font-weight: 800; font-family: 'Inter', sans-serif; font-size: clamp(1.6rem, 3.5vw, 2.2rem) !important; white-space: normal !important; word-wrap: break-word !important; line-height: 1.2 !important; }}
    
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; padding-bottom: 5px; overflow-x: auto; -webkit-overflow-scrolling: touch; scrollbar-width: none; }}
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{ display: none; }}
    .stTabs [data-baseweb="tab"] {{ background-color: transparent; border-radius: 8px; padding: 10px 20px; font-weight: 600; transition: background-color 0.2s ease; white-space: nowrap; color: {text_color}; }}
    .stTabs [data-baseweb="tab"]:hover {{ background-color: {border_color}; }}
    .stTabs [aria-selected="true"] {{ background-color: {card_bg} !important; border: 1px solid {border_color} !important; border-bottom: 3px solid {accent_color} !important; color: {accent_color} !important; }}
    
    [data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; border: 1px solid {border_color}; width: 100%; }}
    
    /* Chat AI CSS */
    [data-testid="stChatMessage"] {{ background-color: {card_bg} !important; border: 1px solid {border_color} !important; border-radius: 12px !important; padding: clamp(10px, 2vw, 15px) !important; margin-bottom: 10px !important; }}
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{ color: {text_color} !important; line-height: 1.6 !important; font-size: 1.05rem !important; }}
    [data-testid="stChatInput"] {{ background-color: transparent !important; padding-bottom: 10px !important; }}
    [data-testid="stChatInput"] > div {{ background-color: {card_bg} !important; border: 1px solid {border_color} !important; border-radius: 12px !important; }}
    [data-testid="stChatInput"] div[data-baseweb="textarea"] {{ background-color: transparent !important; }}
    [data-testid="stChatInput"] div[data-baseweb="base-input"] {{ background-color: transparent !important; }}
    [data-testid="stChatInput"] textarea {{ color: {text_color} !important; -webkit-text-fill-color: {text_color} !important; background-color: transparent !important; caret-color: {accent_color} !important; }}
    [data-testid="stChatInput"] textarea::placeholder {{ color: {text_color} !important; opacity: 0.6 !important; -webkit-text-fill-color: {text_color} !important; }}
    
    @media screen and (max-width: 768px) {{
        h1, h3 {{ text-align: center !important; justify-content: center; width: 100%; }}
        div[data-testid="metric-container"] {{ padding: 16px; margin-bottom: 8px; align-items: center !important; text-align: center !important; }}
        div[data-testid="stMetricValue"] * {{ font-size: 2rem !important; }} 
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; padding-bottom: 8px; justify-content: flex-start; }}
        .stTabs [data-baseweb="tab"] {{ min-height: 44px; padding: 6px 16px; font-size: 0.9rem; }}
        .stPlotlyChart {{ max-width: 100% !important; overflow-x: hidden; }}
    }}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# --- AUTHENTICATION ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown(f"<h1 style='text-align: center; color: {accent_color} !important;'>🔒 Enterprise Secure Portal</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; margin-bottom: 30px; font-size: 1.1rem; color: {text_color};'>Authenticate to access intelligence dashboard</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submit = st.form_submit_button("Authenticate via SQL", use_container_width=True)
            if submit:
                conn = sqlite3.connect('enterprise_backend.db', timeout=15)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT role FROM users WHERE username=? AND password_hash=?", (user, hash_password(pwd)))
                result = cursor.fetchone()
                conn.close()
                if result:
                    st.session_state['logged_in'] = True
                    st.session_state['role'] = result[0]
                    st.rerun()
                else:
                    if user_count == 0:
                        st.error("❌ Database is completely empty. Please refresh the page to trigger Auto-Provisioning.")
                    else:
                        st.error("❌ Invalid security credentials.")
    st.stop()

# --- APP HEADER ---
st.sidebar.markdown(f"""
<div class="pulse-container">
    <div class="pulse-dot"></div>
    <div class="pulse-text" style="color: {text_color};">System Online</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.success(f"✅ Authenticated as: {st.session_state['role']}")
if st.sidebar.button("🚪 Secure Logout", use_container_width=True):
    st.session_state['logged_in'] = False
    st.rerun()

st.markdown(f"<h1>Advanced E-commerce & Customer Intelligence</h1>", unsafe_allow_html=True)

# --- AI CONFIGURATION ---
st.sidebar.header("🧠 AI Configuration")
vault_file = "secure_vault.txt"
api_key = ""

if os.path.exists(vault_file):
    with open(vault_file, "r") as f:
        api_key = f.read().strip()

if not api_key:
    with st.sidebar.form("api_key_form"):
        key_input = st.text_input("Enter Gemini API Key", type="password")
        submit_key = st.form_submit_button("💾 Save Key to OS Vault", use_container_width=True)
        if submit_key and key_input:
            clean_key = key_input.strip()
            with open(vault_file, "w") as f:
                f.write(clean_key)
            st.rerun()
else:
    st.sidebar.success("✅ Key Locked in Secure File")
    if st.sidebar.button("🗑️ Delete Key", use_container_width=True):
        if os.path.exists(vault_file):
            os.remove(vault_file)
        st.rerun()

# --- DATA MANAGEMENT ---
st.sidebar.markdown("---")
st.sidebar.header("1. Database Management")
uploaded_file = st.sidebar.file_uploader("Upload CSV to Update SQL Database", type=['csv'])

if uploaded_file is not None:
    with st.spinner("Injecting data into SQLite Database..."):
        new_data = pd.read_csv(uploaded_file)
        conn = sqlite3.connect('enterprise_backend.db', timeout=15)
        new_data.to_sql('ecommerce_sales', conn, if_exists='replace', index=False)
        conn.close()
        st.cache_data.clear()
        st.sidebar.success("✅ Database Updated!")

@st.cache_data(ttl=300) 
def load_data_from_sql():
    try:
        conn = sqlite3.connect('enterprise_backend.db', timeout=15)
        df = pd.read_sql("SELECT * FROM ecommerce_sales", conn)
        conn.close()
        if df.empty: return df
        df.dropna(subset=['CustomerID', 'Description'], inplace=True)
        df = df[df['Quantity'] > 0]
        df['TotalSales'] = df['Quantity'] * df['UnitPrice']
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Date'] = df['InvoiceDate'].dt.date
        np.random.seed(42) 
        marketing_data = pd.DataFrame({'Date': df['Date'].unique()})
        daily_customers = df.groupby('Date')['CustomerID'].nunique().reset_index()
        marketing_data = pd.merge(marketing_data, daily_customers, on='Date')
        marketing_data['WebsiteVisitors'] = marketing_data['CustomerID'] * np.random.randint(20, 50, size=len(marketing_data))
        marketing_data['AdSpend'] = marketing_data['WebsiteVisitors'] * np.random.uniform(0.5, 1.5, size=len(marketing_data))
        marketing_data.drop(columns=['CustomerID'], inplace=True)
        return pd.merge(df, marketing_data, on='Date', how='left')
    except: return pd.DataFrame() 

raw_df = load_data_from_sql()
if raw_df.empty:
    st.info("👈 System Architecture Online. Please upload a CSV file to initialize the SQL database.")
    st.stop()

# --- FEATURE 1: ONE-CLICK INTERACTIVE FILTERS ---
st.sidebar.markdown("---")
st.sidebar.header("2. Interactive Filters")
min_date, max_date = raw_df['Date'].min(), raw_df['Date'].max()

if 'start_date' not in st.session_state:
    st.session_state['start_date'] = min_date
if 'end_date' not in st.session_state:
    st.session_state['end_date'] = max_date

st.sidebar.markdown("<p style='font-size:0.85rem; margin-bottom:5px; font-weight:600;'>Quick Date Presets</p>", unsafe_allow_html=True)
q_col1, q_col2 = st.sidebar.columns(2)
if q_col1.button("Last 7 Days", use_container_width=True):
    st.session_state['start_date'] = max_date - dt.timedelta(days=7)
    st.session_state['end_date'] = max_date
    st.session_state.pop('chat_history', None)
if q_col2.button("Last 30 Days", use_container_width=True):
    st.session_state['start_date'] = max_date - dt.timedelta(days=30)
    st.session_state['end_date'] = max_date
    st.session_state.pop('chat_history', None)
if q_col1.button("Year to Date", use_container_width=True):
    st.session_state['start_date'] = dt.date(max_date.year, 1, 1)
    st.session_state['end_date'] = max_date
    st.session_state.pop('chat_history', None)
if q_col2.button("All Time", use_container_width=True):
    st.session_state['start_date'] = min_date
    st.session_state['end_date'] = max_date
    st.session_state.pop('chat_history', None)

all_countries = sorted(raw_df['Country'].unique())
selected_countries = st.sidebar.multiselect("🌍 Filter by Region", all_countries, default=all_countries[:5])

date_range = st.sidebar.date_input("📅 Custom Date Range", [st.session_state['start_date'], st.session_state['end_date']], min_value=min_date, max_value=max_date)

if len(date_range) == 2:
    start_date, end_date = date_range
elif len(date_range) == 1:
    start_date = end_date = date_range[0]
else:
    st.stop()

if len(selected_countries) > 0:
    df = raw_df[(raw_df['Date'] >= start_date) & (raw_df['Date'] <= end_date) & (raw_df['Country'].isin(selected_countries))]
else: 
    st.stop()

# --- ML & AI FUNCTIONS ---
st.sidebar.markdown("---")
st.sidebar.header("3. Machine Learning")
k_value = st.sidebar.slider("Customer Clusters (K)", min_value=2, max_value=6, value=4)

def trigger_alert(message, alert_type="WARNING"):
    conn = sqlite3.connect('enterprise_backend.db', timeout=15)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM system_alerts WHERE message = ? AND date(timestamp) = date('now')", (message,))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO system_alerts (alert_type, message) VALUES (?, ?)", (alert_type, message))
        conn.commit()
    conn.close()

@st.cache_data(show_spinner=False, ttl=3600)
def get_nn_predictions(dates, sales):
    temp_df = pd.DataFrame({'Date': dates, 'TotalSales': sales})
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(temp_df[['TotalSales']])
    lookback = min(5, len(scaled_data) - 2)
    X, y = [], []
    for i in range(len(scaled_data) - lookback):
        X.append(scaled_data[i:(i + lookback), 0])
        y.append(scaled_data[i + lookback, 0])
    X, y = np.array(X), np.array(y)
    model = MLPRegressor(hidden_layer_sizes=(20, 10), max_iter=1000, random_state=42)
    model.fit(X, y)
    future_predictions = []
    current_batch = scaled_data[-lookback:].reshape(1, -1)
    for i in range(30):
        pred = model.predict(current_batch)[0]
        future_predictions.append(pred)
        current_batch = np.append(current_batch[:, 1:], [[pred]], axis=1)
    unscaled_preds = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    last_date = pd.to_datetime(temp_df['Date']).max()
    future_dates = [last_date + dt.timedelta(days=x) for x in range(1, 31)]
    return future_dates, unscaled_preds

# --- MAIN DASHBOARD TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Financials", "🔍 Products", "🤖 Segments", "🌐 Traffic", "🧠 Forecast", "🚨 Alerts", "💬 Chat AI"])

with tab1:
    st.markdown(f"<h3>Executive Operations Overview</h3>", unsafe_allow_html=True)
    
    total_revenue = df['TotalSales'].sum()
    total_buyers = df['CustomerID'].nunique()
    total_ad_spend = df.groupby('Date').first()['AdSpend'].sum()
    total_visitors = df.groupby('Date').first()['WebsiteVisitors'].sum()
    
    max_date_calc = pd.to_datetime(df['Date']).max().date()
    current_30d = df[df['Date'] >= (max_date_calc - dt.timedelta(days=30))]
    prev_30d = df[(df['Date'] >= (max_date_calc - dt.timedelta(days=60))) & (df['Date'] < (max_date_calc - dt.timedelta(days=30)))]
    
    curr_rev = current_30d['TotalSales'].sum()
    prev_rev = prev_30d['TotalSales'].sum()
    rev_delta = ((curr_rev - prev_rev) / prev_rev) * 100 if prev_rev > 0 else 0
    
    curr_spend = current_30d.groupby('Date').first()['AdSpend'].sum() if not current_30d.empty else 0
    prev_spend = prev_30d.groupby('Date').first()['AdSpend'].sum() if not prev_30d.empty else 0
    spend_delta = ((curr_spend - prev_spend) / prev_spend) * 100 if prev_spend > 0 else 0
    
    curr_roi = ((curr_rev - curr_spend) / curr_spend) * 100 if curr_spend > 0 else 0
    prev_roi = ((prev_rev - prev_spend) / prev_spend) * 100 if prev_spend > 0 else 0
    roi_delta = curr_roi - prev_roi
    
    curr_buyers = current_30d['CustomerID'].nunique()
    prev_buyers = prev_30d['CustomerID'].nunique()
    curr_visitors = current_30d.groupby('Date').first()['WebsiteVisitors'].sum() if not current_30d.empty else 0
    prev_visitors = prev_30d.groupby('Date').first()['WebsiteVisitors'].sum() if not prev_30d.empty else 0
    curr_conv = (curr_buyers / curr_visitors) * 100 if curr_visitors > 0 else 0
    prev_conv = (prev_buyers / prev_visitors) * 100 if prev_visitors > 0 else 0
    conv_delta = curr_conv - prev_conv
    
    col1, col2, col3, col4 = st.columns(4)
    formatted_revenue = f"${total_revenue / 1_000_000:.2f}M" if total_revenue >= 1_000_000 else f"${total_revenue / 1_000:.1f}K"
    formatted_spend = f"${total_ad_spend / 1_000:.1f}K" if total_ad_spend >= 1_000 else f"${total_ad_spend:,.0f}"

    col1.metric("Gross Revenue", formatted_revenue, f"{rev_delta:.1f}% (30d)")
    col2.metric("Marketing Spend", formatted_spend, f"{spend_delta:.1f}% (30d)")
    col3.metric("ROI", f"{curr_roi:,.1f}%", f"{roi_delta:+.1f}% (30d)")
    col4.metric("Conversion", f"{curr_conv:,.2f}%", f"{conv_delta:+.2f}% (30d)")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h3>Gross Revenue Trajectory</h3>", unsafe_allow_html=True)
    
    daily_revenue_chart = df.groupby('Date')['TotalSales'].sum().reset_index()
    daily_revenue_chart['7-Day Moving Avg'] = daily_revenue_chart['TotalSales'].rolling(window=7, min_periods=1).mean()
    
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(x=daily_revenue_chart['Date'], y=daily_revenue_chart['TotalSales'], mode='lines', name='Daily Raw', line=dict(color=faded_palette[0], width=1), hoverinfo='skip'))
    fig_rev.add_trace(go.Scatter(x=daily_revenue_chart['Date'], y=daily_revenue_chart['7-Day Moving Avg'], mode='lines', name='7-Day Trend', line=dict(color=chart_palette[0], width=3), hovertemplate='<b>%{x}</b><br>$%{y:,.0f}<extra></extra>'))
    
    fig_rev.update_layout(
        template=plotly_template, autosize=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x", hoverlabel=dict(bgcolor=card_bg, font=dict(color=text_color, size=14)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=text_color), yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(color=text_color, family="Inter"), margin=dict(l=0, r=0, t=20, b=0), yaxis=dict(tickprefix="$")
    )
    st.plotly_chart(fig_rev, use_container_width=True, theme=None, config={'displayModeBar': False})

with tab2:
    st.markdown(f"<h3>Top Performing Products</h3>", unsafe_allow_html=True)
    top_products = df.groupby('Description')['TotalSales'].sum().sort_values().tail(5).reset_index()
    
    fig_bar = px.bar(top_products, x='TotalSales', y='Description', orientation='h', text='TotalSales', color_discrete_sequence=[chart_palette[0]])
    fig_bar.update_traces(texttemplate='$%{text:,.0f}', textposition='inside', hovertemplate='<b>%{y}</b><br>$%{x:,.0f}<extra></extra>')
    fig_bar.update_layout(
        template=plotly_template, autosize=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest", hoverlabel=dict(bgcolor=card_bg, font=dict(color=text_color, size=14)),
        font=dict(color=text_color, family="Inter"), uniformtext_minsize=10, uniformtext_mode='hide', margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig_bar, use_container_width=True, theme=None, config={'displayModeBar': False})

with tab3:
    st.markdown(f"<h3>Unsupervised Customer Segmentation</h3>", unsafe_allow_html=True)
    
    freq_col = 'InvoiceNo' if 'InvoiceNo' in df.columns else 'Description'
    freq_agg = 'nunique' if 'InvoiceNo' in df.columns else 'count'
    
    rfm_df = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: ((df['InvoiceDate'].max() + dt.timedelta(days=1)) - x.max()).days, 
        freq_col: freq_agg, 
        'TotalSales': 'sum'
    }).reset_index()
    
    rfm_df.rename(columns={'InvoiceDate': 'Recency', freq_col: 'Frequency', 'TotalSales': 'Monetary'}, inplace=True)
    rfm_df['Cluster'] = KMeans(n_clusters=k_value, random_state=42).fit_predict(StandardScaler().fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']]))
    
    fig_scatter = px.scatter_3d(rfm_df, x='Recency', y='Frequency', z='Monetary', color=rfm_df['Cluster'].astype(str), color_discrete_sequence=chart_palette)
    fig_scatter.update_traces(hovertemplate='Recency: %{x} days<br>Frequency: %{y}<br>Monetary: $%{z:,.0f}<extra></extra>')
    fig_scatter.update_layout(
        template=plotly_template, autosize=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest", hoverlabel=dict(bgcolor=card_bg, font=dict(color=text_color, size=14)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=text_color)), font=dict(color=text_color, family="Inter"), margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig_scatter, use_container_width=True, theme=None, config={'displayModeBar': False})
    
    st.markdown(f"<h3>📊 Predictive Cluster Intelligence</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {text_color};'>The ML algorithm has segmented customers and projected their <b>12-Month Future Value</b> based on historic frequency and recency.</p>", unsafe_allow_html=True)
    
    cluster_summary = rfm_df.groupby('Cluster').agg({'CustomerID': 'count', 'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'}).reset_index()
    
    safe_recency = cluster_summary['Recency'].replace(0, 1) 
    cluster_summary['Est. 12M CLV ($)'] = ((cluster_summary['Monetary'] / cluster_summary['Frequency']) * (365 / safe_recency)).round(0)
    
    cluster_summary.rename(columns={'Cluster': 'Cluster ID', 'CustomerID': 'Total Customers', 'Recency': 'Avg. Days Since Last Order', 'Frequency': 'Avg. Total Orders', 'Monetary': 'Avg. Total Spend ($)'}, inplace=True)
    
    cluster_summary['Avg. Days Since Last Order'] = cluster_summary['Avg. Days Since Last Order'].round(0).astype(int)
    cluster_summary['Avg. Total Orders'] = cluster_summary['Avg. Total Orders'].round(1)
    cluster_summary['Avg. Total Spend ($)'] = cluster_summary['Avg. Total Spend ($)'].round(2)
    
    st.dataframe(cluster_summary.style.format({"Est. 12M CLV ($)": "${:,.0f}", "Avg. Total Spend ($)": "${:,.2f}"}), use_container_width=True, hide_index=True)

with tab4:
    st.markdown(f"<h3>🌐 Web Traffic Analytics</h3>", unsafe_allow_html=True)
    
    web_df = df.groupby('Date')['WebsiteVisitors'].first().reset_index()
    total_visits = web_df['WebsiteVisitors'].sum()
    avg_visits = web_df['WebsiteVisitors'].mean()
    peak_visits = web_df['WebsiteVisitors'].max()
    peak_date = web_df.loc[web_df['WebsiteVisitors'].idxmax(), 'Date']
    
    w_col1, w_col2, w_col3 = st.columns(3)
    w_col1.metric("Total Website Visitors", f"{total_visits:,.0f}")
    w_col2.metric("Avg. Daily Visitors", f"{avg_visits:,.0f}")
    w_col3.metric("Peak Traffic Day", f"{peak_visits:,.0f}", f"Occurred on {peak_date}", delta_color="off")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h3>Traffic Acquisition Trends</h3>", unsafe_allow_html=True)
    
    web_df['7-Day Moving Avg'] = web_df['WebsiteVisitors'].rolling(window=7, min_periods=1).mean()
    
    fig_web = go.Figure()
    fig_web.add_trace(go.Scatter(x=web_df['Date'], y=web_df['WebsiteVisitors'], fill='tozeroy', mode='none', name='Daily Visitors', fillcolor=faded_palette[1], hoverinfo='skip'))
    fig_web.add_trace(go.Scatter(x=web_df['Date'], y=web_df['7-Day Moving Avg'], mode='lines', name='7-Day Trend', line=dict(color=chart_palette[1], width=3), hovertemplate='<b>%{x}</b><br>Visitors: %{y:,.0f}<extra></extra>'))
    
    fig_web.update_layout(
        template=plotly_template, autosize=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x", hoverlabel=dict(bgcolor=card_bg, font=dict(color=text_color, size=14)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=text_color), yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(color=text_color, family="Inter"), margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig_web, use_container_width=True, theme=None, config={'displayModeBar': False})

with tab5:
    st.markdown(f"<h3>🧠 Neural Network Forecast</h3>", unsafe_allow_html=True)
    daily_sales = df.groupby('Date')['TotalSales'].sum().reset_index()
    
    if len(daily_sales) < 5:
        st.warning("⚠️ Insufficient historical data to train Neural Network. Need at least 5 days of data.")
    else:
        with st.spinner("Initializing Scikit-Learn Multi-Layer Perceptron (MLP)..."):
            try:
                future_dates, predictions = get_nn_predictions(daily_sales['Date'].tolist(), daily_sales['TotalSales'].tolist())
                total_projected = sum(predictions)
                avg_projected = np.mean(predictions)
                
                p_col1, p_col2, p_col3 = st.columns([1, 1, 1.3])
                p_col1.metric("30-Day Projected Revenue", f"${total_projected:,.0f}")
                p_col2.metric("Avg. Daily Projected Sales", f"${avg_projected:,.0f}")
                p_col3.metric("Model Architecture", "Scikit-Learn MLP", delta_color="off")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if len(predictions) > 0 and predictions[-1] < (predictions[0] * 0.85): 
                    trigger_alert("Automated Warning: Forecasted revenue drop detected by Neural Net.", "FORECAST_WARNING")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily_sales['Date'], y=daily_sales['TotalSales'], mode='lines', name='Historical Sales', line=dict(color=chart_palette[0]), hovertemplate='<b>%{x}</b><br>Historical: $%{y:,.0f}<extra></extra>'))
                fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Neural Net Trajectory', line=dict(color=chart_palette[1], dash='dot'), hovertemplate='<b>%{x}</b><br>Projected: $%{y:,.0f}<extra></extra>'))
                
                fig.update_layout(
                    template=plotly_template, autosize=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    hovermode="x", hoverlabel=dict(bgcolor=card_bg, font=dict(color=text_color, size=14)),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=text_color), yanchor="top", y=0.99, xanchor="left", x=0.01),
                    font=dict(color=text_color, family="Inter"), margin=dict(l=0, r=0, t=20, b=0), yaxis=dict(tickprefix="$")
                )
                st.plotly_chart(fig, use_container_width=True, theme=None, config={'displayModeBar': False})
            except Exception as e:
                st.error(f"Neural Network Training Failed. Please check logs. Error: {e}")

with tab6:
    st.markdown(f"<h3>🚨 System Anomaly Alerts</h3>", unsafe_allow_html=True)
    try:
        conn = sqlite3.connect('enterprise_backend.db', timeout=15)
        if conn.cursor().execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_alerts'").fetchone(): 
            alerts_df = pd.read_sql("SELECT * FROM system_alerts ORDER BY timestamp DESC LIMIT 10", conn)
            if alerts_df.empty:
                st.success("✅ System Nominal: No anomalies, revenue drops, or security threats detected in the current data window.")
            else:
                alerts_df.rename(columns={'id': 'Alert ID', 'alert_type': 'Severity', 'message': 'Anomaly Description', 'timestamp': 'Time Detected'}, inplace=True)
                st.dataframe(alerts_df, use_container_width=True, hide_index=True)
        conn.close()
    except Exception as e: 
        st.error(f"Database connection error: {e}")

with tab7:
    st.markdown(f"<h3>💬 Chat with your Data (Gemini AI)</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {text_color}; font-size: 0.95rem; margin-bottom: 20px; text-align: center;'>Your live SQL data is securely injected into the AI's context.</p>", unsafe_allow_html=True)
    
    if api_key:
        clean_key = api_key.strip().replace('"', '').replace("'", "")
        genai.configure(api_key=clean_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        top_item = top_products.iloc[-1]['Description'] if not top_products.empty else "N/A"
        
        system_context = f"""
        You are an expert Data Analyst and CFO for an e-commerce platform. 
        You are looking at the live dashboard data right now. 
        Current Metrics Context:
        - Total Revenue: ${total_revenue:,.2f}
        - Total Unique Buyers: {total_buyers}
        - Total Marketing Spend: ${total_ad_spend:,.2f}
        - Current ROI: {curr_roi:,.1f}%
        - Top Selling Product: {top_item}
        - Current Conversion Rate: {curr_conv:,.2f}%
        
        Always answer concisely, professionally, and use bolding for metrics. 
        """

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant", "content": "👋 **System Online.** I am your Enterprise AI Analyst. I have scanned your current Revenue, ROI, Marketing Spend, and Segment data. What insights are you looking for?"}
            ]

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("E.g., Give me an executive summary of our ROI..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.spinner("Analyzing Database..."):
                try:
                    full_prompt = system_context + "\n\nUser Query: " + prompt
                    response = model.generate_content(full_prompt)
                    
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "quota" in error_msg:
                        st.error("⚠️ API Quota Limit Reached. Please try again later or check your billing account.")
                    else:
                        st.error(f"⚠️ Communication Error: {e}")
    else:
        st.warning("⚠️ Paste your Google Gemini API Key in the left sidebar and click 'Save Key to OS Vault' to activate Chat Mode.")