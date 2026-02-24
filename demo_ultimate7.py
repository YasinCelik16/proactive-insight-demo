import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & CSS (High-End Enterprise)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cognizant Mobility | Proactive Insight Hub",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    /* Global Dark Theme Tuning */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Metric Boxes */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Buttons: Enterprise Grade */
    div.stButton > button {
        background: linear-gradient(90deg, #1f2329 0%, #2d323b 100%);
        color: #e0e0e0;
        border: 1px solid #444;
        border-radius: 6px;
        height: 48px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        border-color: #0672CB;
        color: #0672CB;
        box-shadow: 0 0 10px rgba(6, 114, 203, 0.2);
    }
    
    /* Primary Action Button */
    div.stButton > button.primary-btn {
        background: #0672CB;
        border: none;
        color: white;
    }

    /* Table Header Styling */
    thead tr th {
        background-color: #1a1d24 !important;
        color: #0672CB !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #11141a;
        border-right: 1px solid #222;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. STATE & DATA GENERATOR
# -----------------------------------------------------------------------------
if 'step' not in st.session_state: st.session_state['step'] = 0
if 'model' not in st.session_state: st.session_state['model'] = "BMW X1 (U11)"

# Helper function to reset analysis data
def reset_analysis_state():
    keys = ['ai_done', 'scan_done', 'risk_pool']
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

steps = [
    {"id": 0, "title": "Global Fleet View", "desc": "Live Monitoring & Cluster Detection"},
    {"id": 1, "title": "Intake Analysis (Warranty)", "desc": "Analysis of Incoming Warranty Cases"},
    {"id": 2, "title": "AI Pattern Recognition", "desc": "Root Cause Finding via Deep Learning"},
    {"id": 3, "title": "Predictive Fleet Scan", "desc": "Proactive Identification of At-Risk Vehicles"},
    {"id": 4, "title": "Deep Dive & Action", "desc": "Vehicle Diagnosis and Actions"},
    {"id": 5, "title": "Management Summary", "desc": "Business Impact & ROI"}
]

@st.cache_data
def get_fleet_data():
    """Generates data with exact specs for Warranty and SILENT RISKS."""
    np.random.seed(42)
    n = 35000 
    
    regions_list = ["North (Hamburg)", "East (Berlin)", "West (Ruhr)", "South (Munich)", "Central (Frankfurt)"]
    
    # Base Data
    df = pd.DataFrame({
        "VIN": [f"WBA{x:07d}" for x in range(n)],
        "Region": np.random.choice(regions_list, n, p=[0.15, 0.2, 0.25, 0.25, 0.15]),
        "Model": np.random.choice(["BMW X1 (U11)", "BMW iX (I20)", "BMW 5er (G60)"], n, p=[0.33, 0.33, 0.34]), 
        "Prod_Date": pd.date_range(start="2023-01-01", periods=n, freq="8min").date,
        "Mileage": np.random.randint(500, 85000, n)
    })
    
    # --- BASE TELEMETRY ---
    df["Temp_Amb"] = np.random.uniform(-5, 25, n)
    df["Heat_Current"] = np.random.normal(400, 30, n)     # X1 Feature
    df["Charging_Rate"] = np.random.normal(50, 10, n)     # iX Feature
    df["HV_Efficiency"] = np.random.normal(96, 2, n)      # iX Feature
    df["Suspension_Pressure"] = np.random.normal(15, 0.5, n) # 5er Feature
    df["Speed_Avg"] = np.random.normal(100, 20, n)        # 5er Feature
    
    df["GWK_Status"] = False
    df["Error_Code"] = "OK"

    # 1. BMW X1 (SENS_HEAT_RESIST_HIGH)
    mask_x1 = df["Model"] == "BMW X1 (U11)"
    idx_x1_gwk = df[mask_x1].sample(n=52).index
    df.loc[idx_x1_gwk, "GWK_Status"] = True
    df.loc[idx_x1_gwk, "Error_Code"] = "SENS_HEAT_RESIST_HIGH"
    df.loc[idx_x1_gwk, "Temp_Amb"] = np.random.uniform(-6, 2, len(idx_x1_gwk))
    df.loc[idx_x1_gwk, "Heat_Current"] = np.random.uniform(650, 900, len(idx_x1_gwk))
    
    possible_x1_silent = df[mask_x1 & (df["GWK_Status"] == False)]
    idx_x1_silent = possible_x1_silent.sample(n=434).index
    df.loc[idx_x1_silent, "Temp_Amb"] = np.random.uniform(-6, 2, len(idx_x1_silent))
    df.loc[idx_x1_silent, "Heat_Current"] = np.random.uniform(620, 880, len(idx_x1_silent))

    # 2. BMW iX (HV_BATTERY_EFF_DROP)
    mask_ix = df["Model"] == "BMW iX (I20)"
    idx_ix_gwk = df[mask_ix].sample(n=67).index
    df.loc[idx_ix_gwk, "GWK_Status"] = True
    df.loc[idx_ix_gwk, "Error_Code"] = "HV_BATTERY_EFF_DROP"
    df.loc[idx_ix_gwk, "Charging_Rate"] = np.random.uniform(120, 180, len(idx_ix_gwk))
    df.loc[idx_ix_gwk, "HV_Efficiency"] = np.random.uniform(70, 82, len(idx_ix_gwk))
    
    possible_ix_silent = df[mask_ix & (df["GWK_Status"] == False)]
    idx_ix_silent = possible_ix_silent.sample(n=322).index
    df.loc[idx_ix_silent, "Charging_Rate"] = np.random.uniform(120, 180, len(idx_ix_silent))
    df.loc[idx_ix_silent, "HV_Efficiency"] = np.random.uniform(72, 84, len(idx_ix_silent))

    # 3. BMW 5er (AIR_SUSPENSION_LEAK)
    mask_5er = df["Model"] == "BMW 5er (G60)"
    idx_5er_gwk = df[mask_5er].sample(n=24).index
    df.loc[idx_5er_gwk, "GWK_Status"] = True
    df.loc[idx_5er_gwk, "Error_Code"] = "AIR_SUSPENSION_LEAK"
    df.loc[idx_5er_gwk, "Speed_Avg"] = np.random.uniform(160, 220, len(idx_5er_gwk))
    df.loc[idx_5er_gwk, "Suspension_Pressure"] = np.random.uniform(8, 11, len(idx_5er_gwk))
    
    possible_5er_silent = df[mask_5er & (df["GWK_Status"] == False)]
    idx_5er_silent = possible_5er_silent.sample(n=390).index
    df.loc[idx_5er_silent, "Speed_Avg"] = np.random.uniform(160, 220, len(idx_5er_silent))
    df.loc[idx_5er_silent, "Suspension_Pressure"] = np.random.uniform(8.5, 11.5, len(idx_5er_silent))

    return df

data = get_fleet_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔵 Cognizant Mobility")
    st.caption("Proactive Insight Hub v2.5.1")
    st.markdown("---")
    
    with st.expander("👤 Session Info", expanded=True):
        st.write("**Head of Analytics:** Heiko Waibel")
        st.write("**Data Analyst:** Yasin Celik")
        st.write("**Client:** BMW Group")
        st.caption("Date: 26.01.2026")

    st.markdown("---")
    st.success("🟢 MDR API: **Online**")
    st.metric("Latency", "24ms", "-2ms", delta_color="inverse")
    st.info("ℹ️ **Live Connection**\nConnection to Data Lake (Fasta) active. Data refresh every 30s.")

# -----------------------------------------------------------------------------
# 4. HEADER & NAVIGATION
# -----------------------------------------------------------------------------
curr_step = st.session_state['step']
step_meta = steps[curr_step]

col_nav1, col_nav2, col_nav3 = st.columns([1, 4, 1])

with col_nav1:
    if curr_step > 0:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state['step'] -= 1
            st.rerun()

with col_nav2:
    prog_val = (curr_step + 1) / len(steps)
    st.progress(prog_val)
    st.markdown(f"<h2 style='text-align: center; margin-top: -10px;'>{step_meta['title']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #888;'>{step_meta['desc']}</p>", unsafe_allow_html=True)

with col_nav3:
    if curr_step < len(steps) - 1:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state['step'] += 1
            st.rerun()
    else:
        # LOGIC FOR RESET
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state['step'] = 0
            reset_analysis_state() # Clear all data
            st.rerun()

st.divider()

# -----------------------------------------------------------------------------
# 5. MAIN CONTENT VIEWS
# -----------------------------------------------------------------------------

# === STEP 0: GLOBAL FLEET VIEW ===
if curr_step == 0:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Connected Vehicles", "35,000", "+124 (24h)")
    k2.metric("Data Ingest", "1.8 GB/s", "Stable")
    
    gwk_count = len(data[data["GWK_Status"]==True])
    k3.metric("Total Active Cases", f"{gwk_count}", "+5 Cases", delta_color="inverse")
    k4.metric("Fleet Health", "99.6%", "-0.2%")
    
    st.markdown("### 📊 Regional Asset Allocation")
    
    c_chart, c_conf = st.columns([3, 1])
    
    with c_chart:
        df_agg = data.groupby(["Region", "Model"]).size().reset_index(name="Count")
        fig_bar = px.bar(
            df_agg, x="Region", y="Count", color="Model", barmode="group",
            color_discrete_sequence=["#0672CB", "#00C896", "#FF4B4B"],
            title="Vehicle Density by Region & Model"
        )
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="", yaxis_title="Vehicles", font=dict(color="white"),
            margin=dict(t=40, l=0, r=0, b=0), height=450
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c_conf:
        st.info("💡 **Insight Detector**")
        st.markdown("The system detects anomalies in telemetry data.")
        st.markdown("**Please select focus group:**")
        
        # LOGIC FOR AUTO-RESET ON MODEL CHANGE
        previous_model = st.session_state.get('model', "BMW X1 (U11)")
        sel_model = st.radio("Model Series:", ["BMW X1 (U11)", "BMW iX (I20)", "BMW 5er (G60)"])
        
        # If model changes -> RESET ALL
        if sel_model != previous_model:
            st.session_state['model'] = sel_model
            reset_analysis_state()
            st.rerun()
        
        st.markdown("---")
        if "X1" in sel_model:
            st.warning(f"⚠️ **Status: Warning**\nSignificant accumulation of heating resistance signals.")
        elif "iX" in sel_model:
            st.warning(f"⚠️ **Status: Warning**\nCharging efficiency anomalies detected during HPC.")
        elif "5er" in sel_model:
            st.warning(f"⚠️ **Status: Warning**\nAir suspension pressure loss detected at Vmax.")
            
        st.markdown("Recommended Action:\n`-> Analyze Intake Data`")

# === STEP 1: INTAKE ANALYSIS ===
elif curr_step == 1:
    sel_model = st.session_state['model']
    st.markdown(f"### 🔍 Detailed Intake Analysis: **{sel_model}**")
    
    df_m = data[data["Model"] == sel_model]
    df_gwk = df_m[df_m["GWK_Status"] == True]
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown(f"#### Reported Cases ({len(df_gwk)} Total)")
        
        cols_to_show = ["VIN", "Prod_Date", "Mileage", "Error_Code"]
        if "X1" in sel_model:
            cols_to_show.append("Heat_Current")
        elif "iX" in sel_model:
            cols_to_show.append("HV_Efficiency")
        elif "5er" in sel_model:
            cols_to_show.append("Suspension_Pressure")
            
        st.dataframe(
            df_gwk[cols_to_show].head(15),
            use_container_width=True,
            hide_index=True
        )

    with col_right:
        st.markdown("#### Error Clusters")
        fig_pie = px.pie(df_gwk, names="Error_Code", hole=0.6, color_discrete_sequence=px.colors.sequential.RdBu)
        fig_pie.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        top_error = df_gwk["Error_Code"].mode()[0] if len(df_gwk) > 0 else "N/A"
        
        if "X1" in sel_model:
             st.error(f"**Dominant Error:**\n{top_error}\n\nCritical accumulation at low temperatures.")
        elif "iX" in sel_model:
             st.error(f"**Dominant Error:**\n{top_error}\n\nLoss of cell chemistry performance.")
        elif "5er" in sel_model:
             st.error(f"**Dominant Error:**\n{top_error}\n\nSafety-critical suspension issue.")

# === STEP 2: AI PATTERN RECOGNITION ===
elif curr_step == 2:
    st.markdown("### 🧠 Deep Learning Correlation Engine")
    
    c_ctrl, c_vis = st.columns([1, 3])
    
    with c_ctrl:
        st.markdown("#### AI Configuration")
        st.markdown("Model: **IsolationForest v2.4**")
        
        if "X1" in st.session_state['model']:
             st.markdown("Target: `Thermal Resistance`")
        elif "iX" in st.session_state['model']:
             st.markdown("Target: `Charging Efficiency`")
        elif "5er" in st.session_state['model']:
             st.markdown("Target: `Pneumatic Integrity`")
        
        st.markdown("---")
        
        if st.button("▶️ Start Pattern Matching", type="primary"):
            st.toast("Pattern successfully isolated!", icon="✅")
            st.session_state['ai_done'] = True
        
        if st.session_state.get('ai_done'):
             st.success("Pattern ID: #A-9942\nConfidence: 99.4%")

    with c_vis:
        if st.session_state.get('ai_done'):
            st.markdown("#### 🎯 Result: Multi-dimensional Correlation")
            
            df_viz = data[data["Model"] == st.session_state['model']].sample(5000)
            
            # --- SCENARIO 1: BMW X1 ---
            if "X1" in st.session_state['model']:
                fig = px.scatter(
                    df_viz, x="Temp_Amb", y="Heat_Current",
                    color="GWK_Status", 
                    color_discrete_map={False: "rgba(255,255,255,0.2)", True: "#FF4B4B"}, 
                    labels={"Temp_Amb": "Ambient Temp (°C)", "Heat_Current": "Sensor Current (mA)"},
                    title="Anomaly X1: Current Draw at Low Temp"
                )
                fig.add_shape(type="rect", x0=-8, y0=600, x1=3, y1=950, line=dict(color="#FFD700", width=3, dash="dot"))
                insight_text = "**AI Insight:** Control unit overload (>600mA) at temperatures below 3°C. **Vehicles without error messages (white dots) are also in the danger zone!**"
                insight_type = st.error

            # --- SCENARIO 2: BMW iX ---
            elif "iX" in st.session_state['model']:
                fig = px.scatter(
                    df_viz, x="Charging_Rate", y="HV_Efficiency",
                    color="GWK_Status", 
                    color_discrete_map={False: "rgba(255,255,255,0.2)", True: "#FF4B4B"},
                    labels={"Charging_Rate": "Charging Power (kW)", "HV_Efficiency": "Battery Efficiency (%)"},
                    title="Anomaly iX: Efficiency Drop at HPC"
                )
                fig.add_shape(type="rect", x0=110, y0=65, x1=190, y1=85, line=dict(color="#FFD700", width=3, dash="dot"))
                insight_text = "**AI Insight:** Cell efficiency drops critically (<85%) at charging power above 120kW. Many vehicles show this pattern without having failed yet."
                insight_type = st.error

            # --- SCENARIO 3: BMW 5er ---
            elif "5er" in st.session_state['model']:
                fig = px.scatter(
                    df_viz, x="Speed_Avg", y="Suspension_Pressure",
                    color="GWK_Status", 
                    color_discrete_map={False: "rgba(255,255,255,0.2)", True: "#FF4B4B"},
                    labels={"Speed_Avg": "Speed (km/h)", "Suspension_Pressure": "Air Pressure (Bar)"},
                    title="Anomaly 5 Series: Air Suspension Pressure Loss"
                )
                fig.add_shape(type="rect", x0=150, y0=7, x1=230, y1=12, line=dict(color="#FFD700", width=3, dash="dot"))
                insight_text = "**AI Insight:** Pressure loss in bellows (<12 Bar) at speeds over 160 km/h. High predictive potential."
                insight_type = st.error
            
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            insight_type(insight_text)
        else:
            st.info("Please start analysis...")
            st.markdown("<div style='height:400px; border:1px dashed #444; border-radius:10px;'></div>", unsafe_allow_html=True)

# === STEP 3: PREDICTIVE FLEET SCAN ===
elif curr_step == 3:
    st.markdown(f"### 🔮 Predictive Fleet Scan: {st.session_state['model']}")
    st.markdown("Applying the learned pattern to the **entire** fleet (35,000 units).")
    
    if not st.session_state.get('scan_done'):
        if st.button("🚀 START SCAN", type="primary", use_container_width=True):
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            st.session_state['scan_done'] = True
            st.rerun()
    
    else:
        df_scan = data[data["Model"] == st.session_state['model']]
        
        # Filter Logic for Risk Candidates
        if "X1" in st.session_state['model']:
            risk_candidates = df_scan[
                (df_scan["Temp_Amb"] < 3) & 
                (df_scan["Heat_Current"] > 600) & 
                (df_scan["GWK_Status"] == False)
            ].copy()
        elif "iX" in st.session_state['model']:
            risk_candidates = df_scan[
                (df_scan["Charging_Rate"] > 110) & 
                (df_scan["HV_Efficiency"] < 85) & 
                (df_scan["GWK_Status"] == False)
            ].copy()
        elif "5er" in st.session_state['model']:
            risk_candidates = df_scan[
                (df_scan["Speed_Avg"] > 150) & 
                (df_scan["Suspension_Pressure"] < 12) & 
                (df_scan["GWK_Status"] == False)
            ].copy()
            
        risk_candidates["Risk_Score"] = np.random.uniform(79.8, 99.9, len(risk_candidates))
        risk_candidates = risk_candidates.sort_values("Risk_Score", ascending=False)
        st.session_state['risk_pool'] = risk_candidates

        st.markdown("#### ⚠️ High-Risk Candidates Identified")
        
        cols = ["VIN", "Risk_Score", "Mileage"]
        if "X1" in st.session_state['model']: cols.append("Temp_Amb")
        elif "iX" in st.session_state['model']: cols.append("Charging_Rate")
        elif "5er" in st.session_state['model']: cols.append("Speed_Avg")
            
        st.dataframe(
            risk_candidates[cols].head(20),
            use_container_width=True,
            column_config={
                "Risk_Score": st.column_config.ProgressColumn("Probability", format="%.1f%%", min_value=0, max_value=100)
            },
            hide_index=True
        )
    
        c_down, c_3d = st.columns([1, 2])
        
        with c_down:
            st.markdown("### Export & Actions")
            st.write(f"Identified Vehicles: **{len(risk_candidates)}**")
            st.metric("Predicted Hit Rate", f"{(len(risk_candidates)/len(df_scan)*100):.1f}%")
            
            st.download_button(
                label="📥 Full Risk List (CSV)",
                data=risk_candidates.to_csv(index=False),
                file_name="predictive_maintenance_list.csv",
                mime="text/csv"
            )
            
        with c_3d:
            st.markdown("### 🧊 Multi-Dimensional Risk Cube")
            plot_df = pd.concat([risk_candidates.head(300), df_scan.sample(800)])
            plot_df["Status"] = plot_df.apply(lambda x: "High Risk" if x["VIN"] in risk_candidates["VIN"].values else "Normal", axis=1)
            
            x_ax, y_ax, z_ax = "Temp_Amb", "Heat_Current", "Mileage"
            if "iX" in st.session_state['model']: x_ax, y_ax = "Charging_Rate", "HV_Efficiency"
            if "5er" in st.session_state['model']: x_ax, y_ax = "Speed_Avg", "Suspension_Pressure"

            fig_3d = px.scatter_3d(
                plot_df, x=x_ax, y=y_ax, z=z_ax,
                color="Status", color_discrete_map={"High Risk": "#FF0000", "Normal": "#222222"},
                opacity=0.7, title=f"Risk Cluster ({st.session_state['model']})"
            )
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=500, scene=dict(bgcolor="rgba(0,0,0,0)"), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_3d, use_container_width=True)

# === STEP 4: DEEP DIVE & ACTION ===
elif curr_step == 4:
    if 'risk_pool' in st.session_state and not st.session_state['risk_pool'].empty:
        risk_car = st.session_state['risk_pool'].iloc[0]
        st.markdown(f"### 🛠️ Detail Diagnosis: {risk_car['VIN']}")
        
        col_car, col_action = st.columns([2, 1])
        with col_car:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%); padding: 25px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h2 style="margin:0; color:#0672CB;">{risk_car['Model']}</h2>
                    <span style="background:#FF4B4B; padding:5px 10px; border-radius:4px; font-weight:bold; font-size:0.8em;">HIGH RISK</span>
                </div>
                <hr style="border-color: #444;">
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                    <div>
                        <span style="color:#888; font-size:0.9em;">RISK SCORE</span><br>
                        <span style="font-size:1.8em; font-weight:bold; color:#ffcc00;">{risk_car['Risk_Score']:.1f}%</span>
                    </div>
                    <div>
                        <span style="color:#888; font-size:0.9em;">MILEAGE</span><br>
                        <span style="font-size:1.4em; font-weight:bold; color:#fff;">{risk_car['Mileage']:,} km</span>
                    </div>
                    <div>
                        <span style="color:#888; font-size:0.9em;">STATUS</span><br>
                        <span style="font-size:1.4em; font-weight:bold; color:#00C896;">Connected</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Live Telemetrie Dummy
            chart_data = pd.DataFrame({"Time": pd.date_range("now", periods=60, freq="1s"), "Value": np.random.normal(100, 10, 60)})
            fig_live = px.area(chart_data, x="Time", y="Value")
            fig_live.update_traces(line_color="#FF4B4B", fillcolor="rgba(255, 75, 75, 0.2)")
            fig_live.update_layout(yaxis_title="Sensor Signal", height=250, showlegend=False, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_live, use_container_width=True)

        with col_action:
            st.info("System Recommendation:")
            st.markdown("**Proactive Maintenance**")
            st.text_area("Technician Note:", "Automated Ticket generated due to predictive score > 80%.")
            if st.button("📨 Execute Work Order", type="primary"):
                st.success(f"Ticket #WO-{risk_car['VIN'][-4:]} sent to SAP.")
                st.balloons()

# === STEP 5: MANAGEMENT SUMMARY ===
elif curr_step == 5:
    st.markdown("### 📈 Executive Summary")
    
    n_hits = len(st.session_state.get('risk_pool', []))
    # Fallback for Demo if user jumps here directly
    if n_hits == 0:
        if "X1" in st.session_state['model']: n_hits = 434
        elif "iX" in st.session_state['model']: n_hits = 322
        elif "5er" in st.session_state['model']: n_hits = 390
    
    cost_per_case = 1250 
    total_saving = n_hits * cost_per_case
    
    col_sum1, col_sum2 = st.columns([1, 1])
    with col_sum1:
        st.markdown("#### Business Impact")
        st.metric("Prevented Failures", f"{n_hits}", f"Model: {st.session_state['model']}")
        st.metric("Cost Avoidance", f"{total_saving:,.0f} €", "Warranty & Goodwill")
        st.metric("Time Invested", "4 Minutes", "vs. 3 Weeks (Manual)")

    # NEW STEP-BY-STEP SUMMARY (VISUAL)
    with col_sum2:
        st.markdown("#### Process Journey Review")
        
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.success("1. Ingest")
            st.caption("Live Telemetry")
            st.markdown("⬇️")
        with s2:
            st.success("2. AI")
            st.caption("Pattern Match")
            st.markdown("⬇️")
        with s3:
            st.success("3. Scan")
            st.caption("Fleet Prediction")
            st.markdown("⬇️")
        with s4:
            st.success("4. Action")
            st.caption("SAP Ticket")
            st.markdown("✅")
            
        st.info("The system successfully isolated the anomaly, scaled it to the fleet, and initiated measures.")

# -----------------------------------------------------------------------------
# 6. FOOTER
# -----------------------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.caption("© 2026 Cognizant Mobility GmbH | Confidential Demo | Connected Drive Analytics")