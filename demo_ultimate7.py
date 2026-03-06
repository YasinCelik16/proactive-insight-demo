import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import json

# -----------------------------------------------------------------------------
# 🔐 SIMPLE PIN PROTECTION
# -----------------------------------------------------------------------------
PIN_CODE = "0000"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.set_page_config(page_title="Access Required", page_icon="🔒")
    st.markdown("## 🔒 Protected Access")
    st.markdown("Please enter your PIN to access the dashboard.")
    pin_input = st.text_input("Enter PIN", type="password")
    if st.button("Unlock"):
        if pin_input == PIN_CODE:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect PIN.")
    st.stop()

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & HIGH-END CSS (Glassmorphism & Neon)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cognizant Mobility | Predictive Intelligence",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Deep Space Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a1f2e 0%, #0b0f19 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide the default Streamlit Header to prevent overlap with sticky nav */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Robust Sticky Navigation Hack - NOW TARGETING ONLY THE BUTTON ROW */
    div[data-testid="stHorizontalBlock"]:has(.sticky-nav-marker) {
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 0px !important;
        z-index: 999999 !important;
        background-color: rgba(11, 15, 25, 0.95) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(0, 242, 254, 0.2) !important;
        padding-top: 15px !important;
        padding-bottom: 15px !important;
        margin-top: 0px !important;
    }
    
    /* Premium Glassmorphism Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 195, 255, 0.15);
        border-color: rgba(0, 195, 255, 0.3);
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        letter-spacing: -1px;
    }

    /* Enterprise Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #2b3240 0%, #1a1f2e 100%);
        color: #00f2fe;
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 8px;
        height: 50px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div.stButton > button:hover {
        border-color: #00f2fe;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
        transform: scale(1.02);
        color: #ffffff;
    }
    
    /* Improved Terminal Output Style */
    .terminal-output {
        font-family: 'Courier New', Courier, monospace;
        background-color: #050505;
        color: #00ff41;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #00ff41;
        box-shadow: 0 0 15px rgba(0, 255, 65, 0.2);
        font-size: 1.05em;
        line-height: 1.5;
        white-space: pre-wrap;
    }

    /* Headers */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #ffffff, #8892b0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. STATE & DATA GENERATOR
# -----------------------------------------------------------------------------
if 'step' not in st.session_state: st.session_state['step'] = 0
if 'model' not in st.session_state: st.session_state['model'] = "Model Series Alpha (Compact)"
if 'terminal_log' not in st.session_state: st.session_state['terminal_log'] = ""

def reset_analysis_state():
    for k in ['ai_done', 'scan_done', 'risk_pool', 'terminal_log']:
        if k in st.session_state: del st.session_state[k]

steps = [
    {"id": 0, "title": "Global Asset Command", "desc": "Real-time Telemetry & Cluster Analytics"},
    {"id": 1, "title": "Intake Forensics", "desc": "Deep Dive into Reported Incidents & Cost Impact"},
    {"id": 2, "title": "AI Model Deep Dive", "desc": "Statistical Interpretation & Feature Importance"},
    {"id": 3, "title": "Predictive Fleet Scan", "desc": "Multidimensional Risk Scaling"},
    {"id": 4, "title": "Digital Twin & ERP Setup", "desc": "System Architecture & API Integration"},
    {"id": 5, "title": "Value Proposition", "desc": "ROI & Business Impact Summary"}
]

@st.cache_data
def get_fleet_data():
    """Generates enterprise-grade dummy data including geospatial coordinates."""
    np.random.seed(42)
    n = 35000 
    regions = ["North America", "Europe", "Asia-Pacific", "South America", "Middle East"]
    
    # Base coordinates for the map
    coords = {
        "North America": (39.8, -98.5),
        "Europe": (51.1, 10.4),
        "Asia-Pacific": (34.0, 100.0),
        "South America": (-14.2, -51.9),
        "Middle East": (23.8, 45.0)
    }
    
    df = pd.DataFrame({
        "VIN": [f"GEN-{x:07d}" for x in range(n)],
        "Region": np.random.choice(regions, n, p=[0.25, 0.30, 0.25, 0.10, 0.10]),
        "Model": np.random.choice(["Model Series Alpha (Compact)", "Model Series Beta (EV)", "Model Series Gamma (Sedan)"], n, p=[0.33, 0.33, 0.34]), 
        "Prod_Date": pd.date_range(start="2023-01-01", periods=n, freq="8min").date,
        "Mileage": np.random.randint(500, 85000, n),
        "Component_Age_Days": np.random.randint(30, 1000, n)
    })
    
    # Add Latitude and Longitude with some random jitter for map scattering
    df["Lat"] = df["Region"].map(lambda x: coords[x][0] + np.random.normal(0, 6))
    df["Lon"] = df["Region"].map(lambda x: coords[x][1] + np.random.normal(0, 6))
    
    df["Temp_Amb"] = np.random.uniform(-5, 25, n)
    df["Heat_Current"] = np.random.normal(400, 30, n)
    df["Charging_Rate"] = np.random.normal(50, 10, n)
    df["HV_Efficiency"] = np.random.normal(96, 2, n)
    df["Suspension_Pressure"] = np.random.normal(15, 0.5, n)
    df["Speed_Avg"] = np.random.normal(100, 20, n)
    
    df["GWK_Status"] = False
    df["Error_Code"] = "OPERATIONAL"

    mask_alpha = df["Model"] == "Model Series Alpha (Compact)"
    idx_alpha = df[mask_alpha].sample(n=486).index
    df.loc[idx_alpha[:52], "GWK_Status"] = True
    df.loc[idx_alpha[:52], "Error_Code"] = "ERR_THERM_RESIST"
    df.loc[idx_alpha, "Temp_Amb"] = np.random.uniform(-8, 2, len(idx_alpha))
    df.loc[idx_alpha, "Heat_Current"] = np.random.uniform(650, 950, len(idx_alpha))
    df.loc[idx_alpha, "Component_Age_Days"] = np.random.uniform(700, 1000, len(idx_alpha))

    mask_beta = df["Model"] == "Model Series Beta (EV)"
    idx_beta = df[mask_beta].sample(n=389).index
    df.loc[idx_beta[:67], "GWK_Status"] = True
    df.loc[idx_beta[:67], "Error_Code"] = "ERR_CELL_EFFICIENCY"
    df.loc[idx_beta, "Charging_Rate"] = np.random.uniform(120, 180, len(idx_beta))
    df.loc[idx_beta, "HV_Efficiency"] = np.random.uniform(70, 84, len(idx_beta))
    df.loc[idx_beta, "Component_Age_Days"] = np.random.uniform(600, 1000, len(idx_beta))

    mask_gamma = df["Model"] == "Model Series Gamma (Sedan)"
    idx_gamma = df[mask_gamma].sample(n=414).index
    df.loc[idx_gamma[:24], "GWK_Status"] = True
    df.loc[idx_gamma[:24], "Error_Code"] = "ERR_PNEUMATIC_LEAK"
    df.loc[idx_gamma, "Speed_Avg"] = np.random.uniform(160, 220, len(idx_gamma))
    df.loc[idx_gamma, "Suspension_Pressure"] = np.random.uniform(8, 11.5, len(idx_gamma))
    df.loc[idx_gamma, "Component_Age_Days"] = np.random.uniform(500, 1000, len(idx_gamma))

    return df

data = get_fleet_data()

# -----------------------------------------------------------------------------
# MAIN NAVIGATION & UI LOGIC
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔵 Cognizant Mobility")
    st.caption("Predictive Intelligence Hub v3.0 | 🔴 LIVE")
    st.markdown("---")
    
    with st.expander("👤 Expert Team & Session", expanded=True):
        st.write("👨‍💻 **Head of Data Analytics & Data Engineering:** Heiko Waibel")
        st.write("📊 **Data Analyst:** Yasin Celik")
        st.write("🏢 **Client:** Global OEM Demo")
        st.caption(f"Date: {time.strftime('%Y-%m-%d')}")
        
    st.markdown("---")
    st.metric("Lakehouse Latency", "12ms", "-1.2ms", delta_color="inverse")
    st.progress(1.0, text="Data Stream Active")

curr_step = st.session_state['step']
step_meta = steps[curr_step]

# Sticky Navigation Container
nav_container = st.container()
with nav_container:
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        st.markdown('<div class="sticky-nav-marker"></div>', unsafe_allow_html=True)
        if curr_step > 0 and st.button("⬅️ PREVIOUS", use_container_width=True):
            st.session_state['step'] -= 1
            st.rerun()
    with c2:
        st.progress((curr_step + 1) / len(steps))
        st.markdown(f"<h2 style='text-align: center; margin-top:-10px;'>{step_meta['title']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: #00f2fe;'>{step_meta['desc']}</p>", unsafe_allow_html=True)
    with c3:
        if curr_step < len(steps) - 1:
            if st.button("NEXT ➡️", use_container_width=True):
                st.session_state['step'] += 1
                st.rerun()
        else:
            if st.button("🔄 RESTART", use_container_width=True):
                st.session_state['step'] = 0
                reset_analysis_state()
                st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIEWS
# -----------------------------------------------------------------------------

# --- STEP 0: COMMAND CENTER ---
if curr_step == 0:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Monitored Assets", "35,000", "Global Fleet")
    k2.metric("Data Ingestion Rate", "2.4 TB/day", "Streaming")
    k3.metric("Critical Alerts", f"{len(data[data['GWK_Status']==True])}", "+3 in last hour", delta_color="inverse")
    k4.metric("System Health", "99.98%", "Optimal")
    
    st.markdown("<br>", unsafe_allow_html=True)
    c_chart, c_conf = st.columns([7, 3])
    
    with c_chart:
        with st.popover("ℹ️ Info: Asset Distribution & Global Tracking"):
            st.markdown("**Manager Summary:** This view provides a real-time pulse of our entire connected fleet. We can instantly switch between the geographic dispersion (Global Map) to trace environmental factors, and the quantitative distribution (Bar Chart) to monitor volume across asset series.")
            
        tab_map, tab_bar = st.tabs(["🌍 Global Fleet Map", "📊 Region Distribution"])
        
        with tab_map:
            # Map with 2000 sampled data points to keep it highly performant
            df_map = data.sample(2000, random_state=42)
            fig_map = px.scatter_geo(
                df_map, lat="Lat", lon="Lon", color="Model",
                color_discrete_sequence=["#00f2fe", "#4facfe", "#00C896"],
                hover_name="VIN", hover_data={"Lat":False, "Lon":False, "Region":True},
                template="plotly_dark", projection="natural earth"
            )
            fig_map.update_geos(
                showcountries=True, countrycolor="rgba(255,255,255,0.1)",
                showland=True, landcolor="#0b0f19",
                showocean=True, oceancolor="#1a1f2e",
                bgcolor="rgba(0,0,0,0)"
            )
            fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
            st.plotly_chart(fig_map, use_container_width=True)
            
        with tab_bar:
            df_agg = data.groupby(["Region", "Model"]).size().reset_index(name="Assets")
            fig = px.bar(df_agg, x="Region", y="Assets", color="Model", 
                         color_discrete_sequence=["#00f2fe", "#4facfe", "#00C896"],
                         template="plotly_dark")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
                              margin=dict(t=20, l=0, r=0, b=0), height=400)
            st.plotly_chart(fig, use_container_width=True)
        
    with c_conf:
        st.markdown("#### 🎯 Active Focus Area")
        st.markdown("Select an asset cluster to isolate anomalies.")
        
        prev_model = st.session_state['model']
        sel_model = st.selectbox("Asset Series:", ["Model Series Alpha (Compact)", "Model Series Beta (EV)", "Model Series Gamma (Sedan)"])
        
        if sel_model != prev_model:
            st.session_state['model'] = sel_model
            reset_analysis_state()
            st.rerun()
            
        st.markdown("---")
        st.error("⚠️ **ANOMALY DETECTED**\n\nStatistical deviation in telemetry streams found. Suggest immediate drill-down.")

# --- STEP 1: FORENSICS (Upgraded KPIs) ---
elif curr_step == 1:
    sel_model = st.session_state['model']
    df_m = data[data["Model"] == sel_model]
    df_gwk = df_m[df_m["GWK_Status"] == True]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Confirmed Warranty Claims", f"{len(df_gwk)} Assets", "Reported to Service Centers", delta_color="off")
    
    top_region = df_gwk["Region"].mode()[0] if not df_gwk.empty else "N/A"
    m2.metric("Primary Failure Cluster", top_region, "Geospatial Analysis", delta_color="off")
    
    est_cost = len(df_gwk) * 2500
    m3.metric("Estimated Direct Costs", f"€ {est_cost:,.0f}", "Parts & Labor", delta_color="inverse")
    
    st.markdown("---")
    
    col1, col2 = st.columns([6, 4])
    with col1:
        st.markdown("#### Detailed Warranty Registry")
        cols = ["VIN", "Prod_Date", "Mileage", "Error_Code"]
        st.dataframe(
            df_gwk[cols].head(10),
            column_config={
                "Mileage": st.column_config.NumberColumn("Mileage", format="%d mi"),
                "Error_Code": st.column_config.TextColumn("Diagnostic Code")
            },
            use_container_width=True, hide_index=True
        )
    with col2:
        with st.popover("ℹ️ Info: Error Clusters"):
            st.markdown("**Manager Summary:** The pie chart instantly shows which error code is the primary root cause for failures. This allows us to focus our AI on the most expensive and critical issue within the fleet.")
            
        st.markdown("#### Root Cause Distribution")
        fig = px.pie(df_gwk, names="Error_Code", hole=0.7, color_discrete_sequence=["#ff0844", "#ffb199"])
        fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), 
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          annotations=[dict(text='Critical', x=0.5, y=0.5, font_size=20, showarrow=False, font_color="white")])
        st.plotly_chart(fig, use_container_width=True)

# --- STEP 2: AI PATTERN RECOGNITION ---
elif curr_step == 2:
    st.markdown("### 🧠 Unsupervised Learning: Opening the Black Box")
    st.write("We do not rely on simple thresholds. Our team implements highly tuned Isolation Forests to detect multidimensional outliers that traditional BI cannot see.")
    
    if st.button("🚀 EXECUTE AI TRAINING RUN"):
        # Loading bar to simulate complex processing for demos
        my_bar = st.progress(0, text="Initializing Neural Network...")
        time.sleep(0.5)
        my_bar.progress(30, text="Injecting historical telemetry...")
        time.sleep(0.5)
        my_bar.progress(70, text="Calculating multi-dimensional loss function...")
        time.sleep(0.5)
        my_bar.progress(100, text="Pattern matching complete.")
        time.sleep(0.2)
        st.session_state['ai_done'] = True
        st.rerun()
        
    if st.session_state.get('ai_done'):
        st.success("✅ Model successfully trained. Pattern isolated with 99.4% confidence.")
        
        tab1, tab2 = st.tabs(["📊 Feature Space Projection", "🧮 Statistical Deep Dive & SHAP"])
        
        df_viz = data[data["Model"] == st.session_state['model']].sample(3000)
        x_col, y_col = "Temp_Amb", "Heat_Current"
        if "Beta" in st.session_state['model']: x_col, y_col = "Charging_Rate", "HV_Efficiency"
        if "Gamma" in st.session_state['model']: x_col, y_col = "Speed_Avg", "Suspension_Pressure"
        
        with tab1:
            with st.popover("ℹ️ Info: How the AI Learns (2D)"):
                st.markdown("**Manager Summary:** Red dots represent failed machines. Gray dots represent healthy ones. Our AI finds the hidden assets that appear healthy according to standard data, but mathematically behave like failed ones (the dark dots in the red danger zone). These will be the next to fail.")
                
            fig = px.scatter(
                df_viz, x=x_col, y=y_col, color="GWK_Status",
                color_discrete_map={False: "#2b3240", True: "#ff0844"},
                marginal_x="histogram", marginal_y="histogram",
                title="2D Projection of the Anomaly Cluster"
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450)
            fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 **Insight:** The dark dots mixed in the red failure zone are 'silent failures' - assets mathematically identical to failures that haven't broken down *yet*.")

        with tab2:
            c_math, c_shap = st.columns([1, 1])
            with c_math:
                st.markdown("#### Algorithmic Fundamentals")
                st.write("We calculate the anomaly score $s(x, n)$ based on the expected path length $E(h(x))$ in the isolation trees:")
                st.markdown(r"$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$")
                st.write("**Hyperparameters Configured:**")
                st.code("""
n_estimators = 150
max_samples = 'auto'
contamination = 0.012  # Estimated failure rate
bootstrap = False
                """, language="python")

            with c_shap:
                with st.popover("ℹ️ Info: AI Transparency (SHAP)"):
                    st.markdown("**Manager Summary:** This answers the question: 'Why did the AI make this decision?'. The model is not a black box. We make it completely transparent which sensor data (e.g., temperature) is the strongest driver of the failure risk.")
                    
                st.markdown("#### Feature Importance (SHAP)")
                st.write("Which sensors drove the AI's decision to flag these assets?")
                
                features = [x_col, y_col, "Component_Age_Days", "Mileage", "Region_Code"]
                importance = [0.45, 0.35, 0.15, 0.03, 0.02]
                df_shap = pd.DataFrame({"Feature": features, "Impact": importance}).sort_values(by="Impact", ascending=True)
                
                fig_shap = px.bar(df_shap, x="Impact", y="Feature", orientation='h', color_discrete_sequence=["#00f2fe"])
                fig_shap.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0), height=250)
                st.plotly_chart(fig_shap, use_container_width=True)

# --- STEP 3: MEANINGFUL 3D PLOT ---
elif curr_step == 3:
    st.markdown("### 🌌 Multidimensional Risk Scaling")
    st.write("Scaling the mathematical pattern to the entire fleet to identify the exact serial numbers at risk.")
    
    df_scan = data[data["Model"] == st.session_state['model']]
    
    x_c, y_c, z_c = "Temp_Amb", "Heat_Current", "Component_Age_Days"
    if "Beta" in st.session_state['model']: x_c, y_c = "Charging_Rate", "HV_Efficiency"
    if "Gamma" in st.session_state['model']: x_c, y_c = "Speed_Avg", "Suspension_Pressure"

    if "Alpha" in st.session_state['model']:
        risk_mask = (df_scan["Temp_Amb"] < 3) & (df_scan["Heat_Current"] > 600) & (df_scan["Component_Age_Days"] > 650)
    elif "Beta" in st.session_state['model']:
        risk_mask = (df_scan["Charging_Rate"] > 110) & (df_scan["HV_Efficiency"] < 85) & (df_scan["Component_Age_Days"] > 550)
    elif "Gamma" in st.session_state['model']:
        risk_mask = (df_scan["Speed_Avg"] > 150) & (df_scan["Suspension_Pressure"] < 12) & (df_scan["Component_Age_Days"] > 450)
        
    risk_candidates = df_scan[risk_mask & (df_scan["GWK_Status"] == False)].copy()
    risk_candidates["AI_Risk_Score"] = np.random.uniform(85, 99.9, len(risk_candidates))
    st.session_state['risk_pool'] = risk_candidates.sort_values("AI_Risk_Score", ascending=False)
    
    col_3d, col_list = st.columns([6, 4])
    
    with col_3d:
        with st.popover("ℹ️ Info: Why Dimensions Matter (3D)"):
            st.markdown("**Manager Summary:** Conventional dashboards usually look at data in only 2 dimensions. Here, you can see that it takes the 3rd dimension (e.g., component age) to cleanly separate the red failure cluster from the remaining healthy machines. This is the true power of our AI.")
            
        plot_df = pd.concat([risk_candidates, df_scan[~risk_mask].sample(1500)])
        plot_df["Status"] = plot_df.apply(lambda x: "Imminent Risk" if x["VIN"] in risk_candidates["VIN"].values else "Healthy Core", axis=1)
        
        fig_3d = px.scatter_3d(
            plot_df, x=x_c, y=y_c, z=z_c,
            color="Status", 
            color_discrete_map={"Imminent Risk": "#00f2fe", "Healthy Core": "rgba(255,255,255,0.1)"},
            title="AI Decision Boundary (3D Feature Space)"
        )
        fig_3d.update_traces(marker=dict(size=5, line=dict(width=0)))
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#333"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#333"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#333"),
                bgcolor="rgba(0,0,0,0)"
            ),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, b=0, t=40), height=500
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_list:
        st.markdown("#### 🎯 Hidden Risk Pool")
        st.write(f"Isolated **{len(risk_candidates)}** assets showing identical multidimensional signatures to failed units.")
        st.dataframe(
            risk_candidates[["VIN", "AI_Risk_Score", z_c]].head(15),
            column_config={
                "AI_Risk_Score": st.column_config.ProgressColumn("Failure Probability", format="%.1f%%", min_value=0, max_value=100),
                z_c: st.column_config.NumberColumn("Age (Days)")
            },
            hide_index=True, use_container_width=True
        )

# --- STEP 4: ASSET DEEP DIVE & API INTEGRATION ---
elif curr_step == 4:
    if 'risk_pool' in st.session_state and not st.session_state['risk_pool'].empty:
        risk_asset = st.session_state['risk_pool'].iloc[0]
        st.markdown(f"### 🔎 Architecture: From Insight to Action")
        st.write("A model is useless if it doesn't trigger business processes. Here is how we bridge the gap between Data Science and Enterprise Architecture (ERP/SAP).")
        
        # RESTORED HTML DATA BLOCK
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%); padding: 25px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2 style="margin:0; color:#0672CB;">{risk_asset['Model']} - {risk_asset['VIN']}</h2>
                <span style="background:#FF4B4B; padding:5px 10px; border-radius:4px; font-weight:bold; font-size:0.8em; color:white;">HIGH RISK</span>
            </div>
            <hr style="border-color: #444;">
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                <div>
                    <span style="color:#888; font-size:0.9em;">AI RISK SCORE</span><br>
                    <span style="font-size:1.8em; font-weight:bold; color:#ffcc00;">{risk_asset['AI_Risk_Score']:.1f}%</span>
                </div>
                <div>
                    <span style="color:#888; font-size:0.9em;">MILEAGE / USAGE</span><br>
                    <span style="font-size:1.4em; font-weight:bold; color:#fff;">{risk_asset['Mileage']:,} units</span>
                </div>
                <div>
                    <span style="color:#888; font-size:0.9em;">STATUS</span><br>
                    <span style="font-size:1.4em; font-weight:bold; color:#00C896;">Connected</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c_radar, c_action = st.columns([1, 1])
        
        with c_radar:
            with st.popover("ℹ️ Info: Profile Comparison (Radar)"):
                st.markdown("**Manager Summary:** This chart directly compares the flagged asset with the fleet average (gray area). Technicians can instantly identify critical deviations by looking at the red spikes. This translates abstract AI math into a clear mechanical diagnosis.")
                
            categories = ['Temperature Load', 'Current Draw', 'Component Age', 'Vibration', 'Software Errors']
            fig_radar = go.Figure()
            
            # UPDATED COLORS FOR BETTER READABILITY
            # Fleet Average (Dezent Grau/Weiß)
            fig_radar.add_trace(go.Scatterpolar(
                r=[40, 40, 50, 30, 20], theta=categories, fill='toself', 
                name='Fleet Average', line_color='rgba(255,255,255,0.5)', fillcolor='rgba(255,255,255,0.1)'
            ))
            # Target Asset (Gefahren-Rot)
            fig_radar.add_trace(go.Scatterpolar(
                r=[85, 95, 90, 35, 25], theta=categories, fill='toself', 
                name=f'Asset {risk_asset["VIN"]}', line_color='#FF4B4B', fillcolor='rgba(255, 75, 75, 0.4)'
            ))
            
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100]), bgcolor="rgba(0,0,0,0)"), showlegend=True, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=30, b=30))
            st.plotly_chart(fig_radar, use_container_width=True)

        with c_action:
            st.write(f"**Confidence Verification:** Anomalies detected across {3} vector points.")
            
            if st.button("⚡ EXECUTE SAP WORK ORDER API"):
                payload = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "asset_id": risk_asset['VIN'],
                    "action": "CREATE_WORK_ORDER",
                    "priority": "HIGH",
                    "reason_code": "AI_PREDICTIVE_MAINTENANCE",
                    "confidence": round(risk_asset['AI_Risk_Score'], 2),
                    "target_erp": "SAP S/4HANA (Module PM)"
                }
                
                with st.spinner("Establishing secure connection to ERP Gateway..."):
                    time.sleep(1)
                
                st.session_state['terminal_log'] = f"> POST /api/v1/erp/workorders HTTP/1.1\n> Host: erp-gateway.enterprise.local\n> Authorization: Bearer ***\n\n{json.dumps(payload, indent=2)}\n\n< HTTP/1.1 201 Created\n< {{\"status\": \"success\", \"ticket_id\": \"WO-99842A\"}}"
                st.success("API Call successful. Work order created in backend.")
            
            if st.session_state['terminal_log']:
                st.markdown("<div class='terminal-output'>" + st.session_state['terminal_log'] + "</div>", unsafe_allow_html=True)

# --- STEP 5: ROI SUMMARY (With Interactive Calculator) ---
elif curr_step == 5:
    st.markdown("### 💎 Value Realization & ROI Calculator")
    
    n_hits = len(st.session_state.get('risk_pool', [0]*350))
    
    st.markdown("#### Interactive Business Case")
    st.write("Adjust the parameters below to see the immediate impact of proactively replacing components versus reacting to catastrophic failures.")
    
    # NEW: Interactive ROI Slider
    avg_cost = st.slider("Estimated Average Cost per Failure (€)", min_value=1000, max_value=15000, value=2500, step=500)
    saved = n_hits * avg_cost
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Assets Secured", f"{n_hits:,}", "+100% Proactive")
    c_m2.metric("Projected Cost Avoidance", f"€ {saved:,.0f}", "Prevented Warranty Costs")
    c_m3.metric("Time to Insight", "Automated", "-99% vs Manual Analytics")
    
    st.markdown("---")
    st.markdown("#### The Cognizant Advantage")
    st.write("We have just walked through the entire lifecycle of a data analytics project: **From data ingestion to complex machine learning modeling (Isolation Forest & SHAP), all the way to direct automation in enterprise systems (SAP API).** This is the difference between a pretty dashboard and a real business solution.")

# -----------------------------------------------------------------------------
# 6. FOOTER
# -----------------------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.caption(f"© {time.strftime('%Y')} Cognizant Mobility GmbH | Confidential Demo | Connected Fleet Analytics")
