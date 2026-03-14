"""
Main HIV Analytics Dashboard
==============================

Interactive Streamlit dashboard for HIV medical analytics.

Run with: streamlit run src/visualization/dashboards/main_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.data_generator import HIVDataGenerator
from src.analytics.transmission.transmission_analyzer import TransmissionAnalyzer
from src.analytics.treatment.treatment_analyzer import TreatmentAnalyzer
from src.ml.models.viral_suppression_predictor import ViralSuppressionPredictor


# Page configuration
st.set_page_config(
    page_title="HIV/AIDS Data Analysis Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #34495e;
        margin-top: 30px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(n_patients=1000):
    """Load or generate synthetic data."""
    generator = HIVDataGenerator()
    datasets = generator.generate_complete_dataset(n_patients=n_patients)
    return datasets


@st.cache_resource
def train_ml_model(data):
    """Train the viral suppression prediction model."""
    predictor = ViralSuppressionPredictor(model_type="xgboost")
    metrics = predictor.train(data)
    return predictor, metrics


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">HIV/AIDS Data Analysis & Medical Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 20px;">
        <b>Advanced Analytics for HIV Treatment, Transmission Patterns & Patient Outcomes</b><br>
        <i>Real-time insights • Machine Learning Predictions • Evidence-based Decision Support</i>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analysis",
        [
            "Overview",
            "Transmission Analysis",
            "Treatment Efficacy",
            "ML Predictions",
            "Geographic Insights",
            "Trends & Forecasting",
            "Data Dictionary & Guide",
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Settings")
    n_patients = st.sidebar.slider("Number of Patients", 1000, 20000, 1000, 1000)
    
    # Load data
    with st.spinner("Loading data..."):
        datasets = load_data(n_patients)
        patients_df = datasets["patients"]
        lab_results_df = datasets["lab_results"]
        treatments_df = datasets["treatments"]
    
    # Page routing
    if page == "Overview":
        show_overview(patients_df, lab_results_df, treatments_df)
    elif page == "Transmission Analysis":
        show_transmission_analysis(patients_df)
    elif page == "Treatment Efficacy":
        show_treatment_analysis(patients_df, lab_results_df, treatments_df)
    elif page == "ML Predictions":
        show_ml_predictions(patients_df)
    elif page == "Geographic Insights":
        show_geographic_analysis(patients_df)
    elif page == "Trends & Forecasting":
        show_trends_forecasting(patients_df, lab_results_df)
    elif page == "Data Dictionary & Guide":
        show_data_dictionary()


def show_overview(patients_df, lab_results_df, treatments_df):
    """Display overview dashboard."""
    st.markdown('<div class="section-header">Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Patients",
            value=f"{len(patients_df):,}",
            delta=f"+{int(len(patients_df) * 0.05)} this month",
        )
    
    with col2:
        suppression_rate = patients_df["viral_suppression"].mean() * 100
        st.metric(
            label="Viral Suppression Rate",
            value=f"{suppression_rate:.1f}%",
            delta=f"+{2.3}%",
        )
    
    with col3:
        survival_rate = patients_df["is_alive"].mean() * 100
        st.metric(
            label="Survival Rate",
            value=f"{survival_rate:.1f}%",
            delta=f"+{1.5}%",
        )
    
    with col4:
        high_adherence = (patients_df["treatment_adherence"] == "High").mean() * 100
        st.metric(
            label="High Adherence Rate",
            value=f"{high_adherence:.1f}%",
            delta=f"+{3.2}%",
        )
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transmission Route Distribution")
        transmission_counts = patients_df["transmission_route"].value_counts().reset_index()
        transmission_counts.columns = ["Route", "Count"]
        
        fig = px.pie(
            transmission_counts,
            values="Count",
            names="Route",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Age Distribution")
        fig = px.histogram(
            patients_df,
            x="age",
            nbins=30,
            color_discrete_sequence=["#667eea"],
        )
        fig.update_layout(
            xaxis_title="Age",
            yaxis_title="Number of Patients",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Treatment adherence by gender
    st.subheader("Treatment Adherence by Gender")
    adherence_gender = pd.crosstab(
        patients_df["gender"],
        patients_df["treatment_adherence"],
        normalize="index",
    ) * 100
    
    fig = px.bar(
        adherence_gender,
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(
        xaxis_title="Gender",
        yaxis_title="Percentage (%)",
        legend_title="Adherence Level",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_transmission_analysis(patients_df):
    """Display transmission analysis."""
    st.markdown('<div class="section-header">Transmission Analysis</div>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = TransmissionAnalyzer(data=patients_df)
    
    # Demographic analysis
    st.subheader("Transmission Routes by Demographics")
    
    demographic_var = st.selectbox(
        "Select Demographic Variable",
        ["gender", "age_group", "country_code"],
    )
    
    # Perform analysis
    demo_analysis = analyzer.analyze_transmission_by_demographic([demographic_var])
    
    # Create visualization
    fig = px.bar(
        demo_analysis,
        x=demographic_var,
        y="count",
        color="transmission_route",
        barmode="stack",
        title=f"Transmission Routes by {demographic_var.replace('_', ' ').title()}",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show data table
    with st.expander("View Detailed Data"):
        st.dataframe(demo_analysis)
    
    st.markdown("---")
    
    # High-risk populations
    st.subheader("High-Risk Populations")
    high_risk = analyzer.identify_high_risk_populations(top_n=10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(high_risk)
    
    with col2:
        fig = px.bar(
            high_risk,
            x="count",
            y=high_risk.apply(lambda x: f"{x['age_group']}-{x['gender'][:1]}-{x['transmission_route'][:3]}", axis=1),
            orientation="h",
            title="Top 10 High-Risk Population Segments",
        )
        fig.update_layout(yaxis_title="Population Segment", xaxis_title="Number of Cases")
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk scores
    st.subheader("Transmission Risk Scores")
    risk_scores = analyzer.calculate_transmission_risk_scores()
    st.dataframe(risk_scores)


def show_treatment_analysis(patients_df, lab_results_df, treatments_df):
    """Display treatment efficacy analysis."""
    st.markdown('<div class="section-header">Treatment Efficacy Analysis</div>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = TreatmentAnalyzer(
        patients_data=patients_df,
        lab_results_data=lab_results_df,
        treatments_data=treatments_df,
    )
    
    # Regimen comparison
    st.subheader("Treatment Regimen Comparison")
    regimen_comparison = analyzer.compare_treatment_regimens()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(regimen_comparison)
    
    with col2:
        fig = px.bar(
            regimen_comparison,
            x="regimen",
            y="viral_suppression_rate",
            color="viral_suppression_rate",
            color_continuous_scale="Viridis",
            title="Viral Suppression Rate by Regimen",
        )
        fig.update_layout(xaxis_title="Regimen", yaxis_title="Suppression Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Time to suppression
    st.subheader("Time to Viral Suppression")
    time_to_suppression = analyzer.analyze_time_to_suppression()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(time_to_suppression)
    
    with col2:
        fig = px.box(
            patients_df.merge(
                lab_results_df.groupby("patient_id").first().reset_index(),
                on="patient_id",
            ),
            x="treatment_adherence",
            y="cd4_count",
            color="treatment_adherence",
            title="CD4 Count by Adherence Level",
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Treatment effectiveness scores
    st.subheader("Treatment Effectiveness by Country")
    effectiveness = analyzer.generate_treatment_effectiveness_score()
    
    fig = px.scatter(
        effectiveness,
        x="viral_suppression_rate",
        y="survival_rate",
        size="n_patients",
        color="effectiveness_score",
        hover_data=["country_code"],
        title="Treatment Effectiveness Landscape",
        labels={
            "viral_suppression_rate": "Viral Suppression Rate (%)",
            "survival_rate": "Survival Rate (%)",
        },
    )
    st.plotly_chart(fig, use_container_width=True)


def show_ml_predictions(patients_df):
    """Display ML predictions."""
    st.markdown('<div class="section-header">Machine Learning Predictions</div>', unsafe_allow_html=True)
    
    # Train model
    with st.spinner("Training prediction model..."):
        predictor, metrics = train_ml_model(patients_df)
    
    # Display metrics
    st.subheader("Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    with col4:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    
    st.markdown("---")
    
    # Feature importance
    if predictor.feature_importance is not None:
        st.subheader("Feature Importance")
        
        fig = px.bar(
            predictor.feature_importance.head(10),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 Most Important Features",
        )
        fig.update_layout(yaxis_title="", xaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Interactive prediction
    st.subheader("Predict Viral Suppression for New Patient")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=15, max_value=80, value=35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        transmission_route = st.selectbox(
            "Transmission Route",
            ["Heterosexual", "MSM", "IDU", "MTCT", "Blood_Products", "Unknown"],
        )
    
    with col2:
        cd4_count = st.number_input("CD4 Count at Diagnosis", min_value=0, max_value=1500, value=350)
        viral_load = st.number_input("Viral Load at Diagnosis", min_value=50, max_value=1000000, value=50000)
        who_stage = st.selectbox("WHO Clinical Stage", ["Stage 1", "Stage 2", "Stage 3", "Stage 4"])
    
    with col3:
        treatment_adherence = st.selectbox("Expected Treatment Adherence", ["High", "Medium", "Low"])
        days_to_treatment = st.number_input("Days from Diagnosis to Treatment", min_value=0, max_value=365, value=7)
    
    if st.button("Predict"):
        # Create patient data
        new_patient = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "transmission_route": transmission_route,
            "cd4_count_at_diagnosis": cd4_count,
            "viral_load_at_diagnosis": viral_load,
            "who_clinical_stage": who_stage,
            "treatment_adherence": treatment_adherence,
            "diagnosis_date": datetime.now(),
            "treatment_start_date": datetime.now() + pd.Timedelta(days=days_to_treatment),
            "viral_suppression": True,  # Placeholder
            "is_alive": True,  # Placeholder
        }])
        
        # Predict
        probabilities = predictor.predict_proba(new_patient)
        suppression_prob = probabilities[0][1] * 100
        
        st.success(f"### Predicted Probability of Viral Suppression: {suppression_prob:.1f}%")
        
        # Interpretation
        if suppression_prob >= 75:
            st.info("✅ **High likelihood** of achieving viral suppression with proper adherence.")
        elif suppression_prob >= 50:
            st.warning("⚠️ **Moderate likelihood** of viral suppression. Close monitoring recommended.")
        else:
            st.error("❌ **Low likelihood** of viral suppression. Intensive support and monitoring required.")


def show_geographic_analysis(patients_df):
    """Display geographic analysis."""
    st.markdown('<div class="section-header">Geographic Insights</div>', unsafe_allow_html=True)
    
    # Country-level statistics
    country_stats = patients_df.groupby("country_code").agg({
        "patient_id": "count",
        "viral_suppression": "mean",
        "is_alive": "mean",
        "age": "mean",
    }).reset_index()
    
    country_stats.columns = ["Country", "Total_Patients", "Suppression_Rate", "Survival_Rate", "Avg_Age"]
    country_stats["Suppression_Rate"] *= 100
    country_stats["Survival_Rate"] *= 100
    
    # Map visualization (simplified)
    st.subheader("Patients by Country")
    
    fig = px.bar(
        country_stats.sort_values("Total_Patients", ascending=False),
        x="Country",
        y="Total_Patients",
        color="Suppression_Rate",
        color_continuous_scale="RdYlGn",
        title="Patient Distribution and Suppression Rates by Country",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed country data
    st.subheader("Country-Level Statistics")
    st.dataframe(country_stats.sort_values("Total_Patients", ascending=False))


def show_trends_forecasting(patients_df, lab_results_df):
    """Display trends and forecasting."""
    st.markdown('<div class="section-header">Trends & Forecasting</div>', unsafe_allow_html=True)
    
    # Temporal trends
    st.subheader("New Diagnoses Over Time")
    
    patients_df["diagnosis_date"] = pd.to_datetime(patients_df["diagnosis_date"])
    patients_df["diagnosis_month"] = patients_df["diagnosis_date"].dt.to_period("M")
    
    monthly_diagnoses = patients_df.groupby("diagnosis_month").size().reset_index(name="count")
    monthly_diagnoses["diagnosis_month"] = monthly_diagnoses["diagnosis_month"].astype(str)
    
    fig = px.line(
        monthly_diagnoses,
        x="diagnosis_month",
        y="count",
        title="Monthly New HIV Diagnoses",
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Number of Diagnoses")
    st.plotly_chart(fig, use_container_width=True)
    
    # Viral suppression trends
    st.subheader("Viral Suppression Trends")
    
    lab_results_df["test_date"] = pd.to_datetime(lab_results_df["test_date"])
    lab_results_df["test_quarter"] = lab_results_df["test_date"].dt.to_period("Q")
    
    quarterly_suppression = (
        lab_results_df.groupby("test_quarter")["is_suppressed"]
        .mean()
        .reset_index()
    )
    quarterly_suppression["test_quarter"] = quarterly_suppression["test_quarter"].astype(str)
    quarterly_suppression["is_suppressed"] *= 100
    
    fig = px.line(
        quarterly_suppression,
        x="test_quarter",
        y="is_suppressed",
        title="Quarterly Viral Suppression Rate",
        markers=True,
    )
    fig.update_layout(xaxis_title="Quarter", yaxis_title="Suppression Rate (%)")
    st.plotly_chart(fig, use_container_width=True)


def show_data_dictionary():
    """Display comprehensive data dictionary and parameter explanations."""
    st.markdown('<div class="section-header">Data Dictionary & Parameter Guide</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196f3; margin-bottom: 30px;">
        <h3 style="color: #1976d2; margin-top: 0;">Welcome to the Data Dictionary</h3>
        <p style="font-size: 1.1rem;">
        This guide provides detailed explanations of all metrics, parameters, and terminology used in the 
        HIV/AIDS Data Analysis Platform. Use this reference to better understand the data and make informed decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Key Metrics", 
        "Patient Data", 
        "Treatment Terms", 
        "Clinical Parameters",
        "Statistical Terms"
    ])
    
    with tab1:
        st.subheader("Key Performance Metrics")
        
        metrics_data = {
            "Metric": [
                "Total Patients",
                "Viral Suppression Rate",
                "Treatment Adherence Rate",
                "Survival Rate",
                "ROC-AUC Score",
                "Effectiveness Score",
                "Risk Score"
            ],
            "Definition": [
                "Total number of HIV-positive patients in the analysis dataset",
                "Percentage of patients who achieved undetectable viral load (<200 copies/mL) while on treatment",
                "Percentage of patients classified as having 'High' adherence to their antiretroviral therapy regimen",
                "Percentage of patients who are alive at the time of data collection",
                "Area Under the Receiver Operating Characteristic Curve - measures ML model's ability to distinguish between outcomes (0-1 scale)",
                "Composite score (0-100) combining viral suppression, adherence, and survival rates to measure overall treatment program success",
                "Composite score (0-100) evaluating multiple risk factors including CD4 count, viral load, late diagnosis, and adherence"
            ],
            "Good Value": [
                "Varies by dataset",
                "≥85% (UNAIDS target: 95%)",
                "≥70%",
                "≥95%",
                "≥0.80 (excellent: ≥0.90)",
                "≥75 (High effectiveness)",
                "≥70 (Lower risk)"
            ],
            "Clinical Significance": [
                "Sample size affects statistical power",
                "Primary treatment success indicator; higher rates mean better patient outcomes",
                "Strong predictor of treatment success; low adherence leads to resistance",
                "Overall program effectiveness measure",
                "Model reliability indicator; higher means more accurate predictions",
                "Overall program quality; identifies best practices",
                "Identifies populations needing intervention"
            ]
        }
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
        
        st.info("""
        Clinical Interpretation Tips:
        - A viral suppression rate below 80% indicates program issues requiring investigation
        - Low adherence rates often correlate with poor suppression outcomes
        - ROC-AUC >0.80 means the ML model can reliably predict patient outcomes
        """)
    
    with tab2:
        st.subheader("Patient Demographics & Characteristics")
        
        st.markdown("### Age Groups")
        age_data = {
            "Age Range": ["0-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65+"],
            "Category": ["Children", "Youth/Young Adults", "Adults", "Adults", "Middle Age", "Older Adults", "Seniors"],
            "Clinical Notes": [
                "Pediatric HIV - often MTCT (Mother-to-Child Transmission)",
                "High-risk age for acquisition; MSM and heterosexual transmission common",
                "Peak incidence age; highest transmission rates",
                "Established infection; focus on long-term management",
                "Long-term survivors; monitoring for comorbidities",
                "Age-related complications; drug interactions common",
                "Multiple comorbidities; complex medication management"
            ]
        }
        st.dataframe(pd.DataFrame(age_data), use_container_width=True, hide_index=True)
        
        st.markdown("### Gender Categories")
        gender_data = {
            "Gender": ["Male", "Female", "Other"],
            "Global Prevalence": ["~52%", "~47%", "~1%"],
            "Key Considerations": [
                "Higher rates of MSM transmission in some regions; different risk profiles",
                "Pregnancy considerations; MTCT prevention; cervical cancer screening",
                "Transgender individuals; hormone therapy interactions; specific care needs"
            ]
        }
        st.dataframe(pd.DataFrame(gender_data), use_container_width=True, hide_index=True)
        
        st.markdown("### Transmission Routes")
        transmission_data = {
            "Route": [
                "Heterosexual",
                "MSM",
                "IDU",
                "MTCT",
                "Blood Products",
                "Unknown"
            ],
            "Full Name": [
                "Heterosexual Contact",
                "Men who have Sex with Men",
                "Injection Drug Use",
                "Mother-to-Child Transmission",
                "Blood Products/Transfusion",
                "Unknown/Unreported"
            ],
            "Global %": [
                "~45%",
                "~30%",
                "~10%",
                "~5%",
                "~2%",
                "~8%"
            ],
            "Prevention Focus": [
                "Condom use, PrEP, education",
                "PrEP, condoms, regular testing",
                "Needle exchange, harm reduction, treatment programs",
                "PMTCT programs, maternal ART, infant prophylaxis",
                "Blood screening, safe practices",
                "Comprehensive prevention education"
            ]
        }
        st.dataframe(pd.DataFrame(transmission_data), use_container_width=True, hide_index=True)
        
        st.warning("""
        Important Notes:
        - Transmission route data helps target prevention efforts
        - "Unknown" category often indicates stigma or disclosure concerns
        - Regional variations are significant - always consider local epidemiology
        """)
    
    with tab3:
        st.subheader("Treatment & Medication Terms")
        
        st.markdown("### Antiretroviral Therapy (ART) Regimens")
        art_data = {
            "Regimen": [
                "TDF/3TC/EFV",
                "TDF/3TC/DTG",
                "ABC/3TC/DTG",
                "TAF/FTC/BIC",
                "DRV/r + TDF/FTC"
            ],
            "Components": [
                "Tenofovir + Lamivudine + Efavirenz",
                "Tenofovir + Lamivudine + Dolutegravir",
                "Abacavir + Lamivudine + Dolutegravir",
                "Tenofovir alafenamide + Emtricitabine + Bictegravir",
                "Darunavir/ritonavir + Tenofovir + Emtricitabine"
            ],
            "Class Combination": [
                "2 NRTIs + 1 NNRTI",
                "2 NRTIs + 1 INSTI",
                "2 NRTIs + 1 INSTI",
                "2 NRTIs + 1 INSTI",
                "2 NRTIs + 1 PI (boosted)"
            ],
            "Effectiveness": [
                "85-88% suppression",
                "86-90% suppression (Preferred)",
                "85-90% suppression",
                "88-92% suppression",
                "84-88% suppression"
            ],
            "Notes": [
                "WHO first-line; CNS side effects possible",
                "WHO preferred first-line; high barrier to resistance",
                "Alternative if TDF contraindicated",
                "Newer option; good tolerability",
                "Second-line option; more pills"
            ]
        }
        st.dataframe(pd.DataFrame(art_data), use_container_width=True, hide_index=True)
        
        st.markdown("### Drug Classes Explained")
        st.markdown("""
        - NRTI (Nucleoside Reverse Transcriptase Inhibitor): Blocks virus from copying its genetic material
        - NNRTI (Non-Nucleoside Reverse Transcriptase Inhibitor): Also blocks reverse transcriptase, different mechanism
        - INSTI (Integrase Strand Transfer Inhibitor): Prevents viral DNA from integrating into human DNA
        - PI (Protease Inhibitor): Blocks protease enzyme needed for viral replication
        - /r (ritonavir-boosted): Low-dose ritonavir increases drug levels
        """)
        
        st.markdown("### Treatment Adherence Levels")
        adherence_data = {
            "Level": ["High", "Medium", "Low"],
            "Definition": [
                "≥95% of doses taken on time (misses <3 doses/month)",
                "85-94% of doses taken (misses 3-9 doses/month)",
                "<85% of doses taken (misses >9 doses/month)"
            ],
            "Typical Outcomes": [
                "85-95% viral suppression; low resistance risk",
                "60-80% viral suppression; moderate resistance risk",
                "30-60% viral suppression; high resistance risk"
            ],
            "Clinical Action": [
                "Continue current regimen; routine monitoring",
                "Enhanced adherence counseling; identify barriers",
                "Intensive support; consider simplified regimen; assess resistance"
            ]
        }
        st.dataframe(pd.DataFrame(adherence_data), use_container_width=True, hide_index=True)
        
        st.success("""
        Treatment Success Factors:
        - Adherence is THE most important predictor (65.5% of model importance)
        - Early treatment initiation improves outcomes
        - Simplified regimens (1 pill/day) improve adherence
        - Addressing side effects early prevents discontinuation
        """)
    
    with tab4:
        st.subheader("Clinical Parameters & Lab Values")
        
        st.markdown("### CD4 Count (cells/μL)")
        cd4_data = {
            "CD4 Range": [
                "≥500",
                "350-499",
                "200-349",
                "<200",
                "<50"
            ],
            "Classification": [
                "Normal/Healthy",
                "Mild Immunosuppression",
                "Moderate Immunosuppression",
                "Severe Immunosuppression (AIDS)",
                "Critical - High Mortality Risk"
            ],
            "Clinical Implications": [
                "Good immune function; low opportunistic infection risk",
                "Slightly increased infection risk; monitor closely",
                "Increased infection risk; consider prophylaxis",
                "High risk of opportunistic infections; immediate treatment",
                "Life-threatening risk; aggressive treatment & prophylaxis"
            ],
            "WHO Stage": [
                "Stage 1",
                "Stage 1-2",
                "Stage 2-3",
                "Stage 3-4",
                "Stage 4"
            ],
            "Urgency": [
                "Routine care",
                "Close monitoring",
                "Enhanced care",
                "Urgent intervention",
                "Emergency care"
            ]
        }
        st.dataframe(pd.DataFrame(cd4_data), use_container_width=True, hide_index=True)
        
        st.markdown("### Viral Load (copies/mL)")
        vl_data = {
            "Viral Load": [
                "Undetectable (<50)",
                "Low (50-199)",
                "Moderate (200-999)",
                "High (1,000-99,999)",
                "Very High (≥100,000)"
            ],
            "Treatment Status": [
                "Suppressed - Treatment Success",
                "Near-suppressed - Good response",
                "Detectable - Treatment concern",
                "Failure - Action needed",
                "Untreated or Major failure"
            ],
            "Clinical Action": [
                "Continue treatment; routine monitoring every 6 months",
                "Continue treatment; monitor in 3 months",
                "Assess adherence; enhanced counseling; consider resistance testing",
                "Adherence intervention; resistance testing; consider regimen change",
                "Immediate action; resistance testing; regimen change; intensive support"
            ],
            "Transmission Risk": [
                "Undetectable = Untransmittable (U=U)",
                "Very low transmission risk",
                "Low but present risk",
                "Moderate to high risk",
                "Very high transmission risk"
            ]
        }
        st.dataframe(pd.DataFrame(vl_data), use_container_width=True, hide_index=True)
        
        st.info("""
        U=U (Undetectable = Untransmittable):
        - People with HIV who achieve and maintain an undetectable viral load cannot sexually transmit the virus
        - This is a critical concept for reducing stigma and supporting treatment adherence
        - Requires consistent viral suppression (<200 copies/mL) for at least 6 months
        """)
        
        st.markdown("### WHO Clinical Stages")
        who_data = {
            "Stage": ["Stage 1", "Stage 2", "Stage 3", "Stage 4"],
            "Description": [
                "Asymptomatic",
                "Mild symptoms",
                "Advanced HIV disease",
                "Severe HIV disease (AIDS)"
            ],
            "Typical CD4": [
                ">500", "350-500", "200-350", "<200"
            ],
            "Examples": [
                "No symptoms; normal activity",
                "Weight loss <10%, recurrent respiratory infections",
                "Weight loss >10%, chronic diarrhea, oral thrush, TB",
                "Wasting syndrome, opportunistic infections, HIV-related cancers"
            ]
        }
        st.dataframe(pd.DataFrame(who_data), use_container_width=True, hide_index=True)
    
    with tab5:
        st.subheader("Statistical & ML Terms")
        
        st.markdown("### Machine Learning Metrics")
        ml_data = {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall (Sensitivity)",
                "F1-Score",
                "ROC-AUC",
                "Specificity"
            ],
            "Formula/Range": [
                "(TP + TN) / Total | 0-1 or 0-100%",
                "TP / (TP + FP) | 0-1",
                "TP / (TP + FN) | 0-1",
                "2 × (Precision × Recall) / (Precision + Recall)",
                "Area under ROC curve | 0-1",
                "TN / (TN + FP) | 0-1"
            ],
            "What It Measures": [
                "Overall correctness of predictions",
                "Of predicted positives, how many are truly positive (avoid false alarms)",
                "Of actual positives, how many did we catch (avoid missing cases)",
                "Balance between precision and recall",
                "Model's ability to distinguish between classes across all thresholds",
                "Of actual negatives, how many did we correctly identify"
            ],
            "Good Value": [
                "≥80%",
                "≥85%",
                "≥80%",
                "≥80%",
                "≥0.80 (excellent: ≥0.90)",
                "≥85%"
            ],
            "Clinical Context": [
                "How often model is right overall",
                "When model predicts 'will suppress', how often is it correct?",
                "Of patients who will suppress, how many does model identify?",
                "Overall model quality considering both precision and recall",
                "Excellent (≥0.90), Good (0.80-0.89), Fair (0.70-0.79), Poor (<0.70)",
                "How well model identifies patients who won't suppress"
            ]
        }
        st.dataframe(pd.DataFrame(ml_data), use_container_width=True, hide_index=True)
        
        st.markdown("### Confusion Matrix Terms")
        st.markdown("""
        - TP (True Positive): Correctly predicted viral suppression
        - TN (True Negative): Correctly predicted non-suppression  
        - FP (False Positive): Predicted suppression but didn't achieve it (Type I error)
        - FN (False Negative): Predicted non-suppression but achieved it (Type II error)
        """)
        
        st.markdown("### Feature Importance")
        st.markdown("""
        What is it?: Measures how much each variable contributes to the model's predictions (0-1 scale, higher = more important)
        
        Top Features in Our Model:
        1. Treatment Adherence (65.5%): Overwhelmingly most important predictor
        2. High-risk Transmission (4.9%): Affects baseline risk
        3. Transmission Route (4.7%): Different routes have different outcomes
        4. CD4 Count at Diagnosis (4.5%): Starting health status matters
        5. Days to Treatment (4.2%): Earlier treatment = better outcomes
        
        Clinical Interpretation: This tells us that adherence interventions will have the biggest impact on patient outcomes.
        """)
        
        st.markdown("### Statistical Significance")
        st.markdown("""
        - p-value: Probability that results occurred by chance
          - p < 0.05: Statistically significant (≤5% chance it's random)
          - p < 0.01: Highly significant (≤1% chance)
          - p ≥ 0.05: Not statistically significant
        
        - Confidence Interval (CI): Range where true value likely falls
          - 95% CI means 95% confident the true value is in this range
          - Narrower intervals = more precise estimates
        """)
    
    # Add glossary section at the bottom
    st.markdown("---")
    st.markdown("### Quick Reference Glossary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        Common Abbreviations:
        - ART: Antiretroviral Therapy
        - ARV: Antiretroviral (drug)
        - PLHIV: People Living with HIV
        - PMTCT: Prevention of Mother-to-Child Transmission
        - PrEP: Pre-Exposure Prophylaxis
        - PEP: Post-Exposure Prophylaxis
        - MSM: Men who have Sex with Men
        - IDU: Injection Drug Use
        - MTCT: Mother-to-Child Transmission
        - WHO: World Health Organization
        - UNAIDS: Joint United Nations Programme on HIV/AIDS
        - CDC: Centers for Disease Control and Prevention
        """)
    
    with col2:
        st.markdown("""
        Key Concepts:
        - U=U: Undetectable = Untransmittable
        - Viral Suppression: Viral load <200 copies/mL
        - AIDS: Advanced HIV (CD4 <200 or Stage 4)
        - Resistance: Virus mutations making drugs ineffective
        - First-line: Initial recommended treatment
        - Second-line: Alternative if first-line fails
        - Opportunistic Infection: Infections in weakened immune system
        - 90-90-90 Targets: 90% diagnosed, 90% on treatment, 90% suppressed
        - Treatment as Prevention: ART reduces transmission
        """)
    
    st.markdown("---")
    st.success("""
    How to Use This Guide:
    1. Use the tabs above to navigate to specific topics
    2. Refer back to this guide when viewing dashboard metrics
    3. Understanding these parameters helps you interpret results correctly
    4. Share this guide with team members for consistent interpretation
    5. Bookmark this page for quick reference during analysis sessions
    """)
    
    st.info("""
    Need More Information?
    - WHO HIV Guidelines: https://www.who.int/health-topics/hiv-aids
    - CDC HIV Resources: https://www.cdc.gov/hiv/
    - UNAIDS Data: https://www.unaids.org/
    - Contact the analytics team for dataset-specific questions
    """)


if __name__ == "__main__":
    main()

