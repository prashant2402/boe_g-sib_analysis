import base64
import io
import re
from datetime import datetime, timedelta
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from wordcloud import WordCloud

# Set page configuration - simplified
st.set_page_config(
    page_title="G-SIB Analysis Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simplified CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7f9;
        color: #1a2e44;
    }
    h1, h2, h3 {
        color: #1a2e44;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #1a2e44;
        color: white;
    }
    /* Add a special risk warning class */
    .risk-warning {
        background-color: #ffe8e8;
        border-left: 4px solid #d62728;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    /* Add a risk insight class */
    .risk-insight {
        background-color: #f0f8ff;
        border-left: 4px solid #1a2e44;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    /* Risk indicator badges */
    .risk-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 5px;
    }
    .risk-badge-high {
        background-color: #d62728;
        color: white;
    }
    .risk-badge-medium {
        background-color: #ff7f0e;
        color: white;
    }
    .risk-badge-low {
        background-color: #2ca02c;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Academic disclaimer function - simplified
def display_academic_disclaimer():
    """Display a disclaimer indicating this is an academic project"""
    st.markdown(
        """
        <div style="background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 10px 0; font-size: 0.8em;">
            <strong>Academic Project:</strong> This dashboard is created for academic purposes only with simulated data.
        </div>
        """,
        unsafe_allow_html=True,
    )


# Footer function - simplified
def display_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>G-SIB Analysis Dashboard Simulation | Academic Project</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Dashboard Header and Logo - simplified
def display_header():
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(
            "https://codeitgroup.com/wp-content/uploads/2023/08/bank-of-england-logo.png",
            width=150,
        )

    with col2:
        st.title("G-SIB Quarterly Announcements Analysis")
        st.markdown(
            """
            <div style="margin-top: -15px;">
                <p>Academic Project</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Simple layout structure - single page with sections
def create_layout():
    # Display header with logo and title
    display_header()

    # Add academic disclaimer at the top
    display_academic_disclaimer()

    return None


# Sidebar for filters - simplified for single bank focus
def create_sidebar_filters():
    with st.sidebar:
        st.header("Analysis Filters")

        # Add academic project note
        st.markdown(
            """
            <div style="font-size: 0.8em; font-style: italic; margin-bottom: 15px; color: #666;">
            Academic project using simulated data
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Bank Selection - Just one bank at a time
        st.subheader("Bank Selection")
        all_banks = ["JPMorgan", "UBS"]
        selected_bank = st.selectbox(
            "Select Bank to Analyze",
            options=all_banks,
            key="bank_selection_single",
        )

        # Quarter Selection
        st.subheader("Quarter Selection")
        all_quarters = ["2024 Q1", "2024 Q2", "2024 Q3", "2024 Q4"]
        selected_quarter = st.selectbox(
            "Select Quarter to Analyze",
            options=all_quarters,
            index=3,  # Default to latest quarter (Q4)
            key="quarter_selection_single",
        )

        # Add comparison bank option only for Advanced Analytics
        st.subheader("Peer Comparison")
        show_comparison = st.checkbox("Enable peer comparison", value=False)

        comparison_bank = None
        if show_comparison:
            # Filter out the already selected bank
            comparison_options = [bank for bank in all_banks if bank != selected_bank]
            if comparison_options:
                comparison_bank = st.selectbox(
                    "Select bank to compare with",
                    options=comparison_options,
                    key="comparison_bank_selection",
                )

        # Date of Analysis - simplified
        st.subheader("Report Information")
        st.markdown(f"**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}")

        # Export Report Section - simplified
        st.header("Export Report")
        if st.button("Generate PDF Report", key="pdf_button"):
            with st.spinner("Generating PDF report..."):
                # Will be implemented in a later function
                b64_pdf = export_report_as_pdf(
                    selected_bank, selected_quarter, comparison_bank
                )

                # Create download link
                pdf_filename = f"G-SIB_Analysis_{selected_bank}_{selected_quarter.replace(' ', '')}.pdf"
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{pdf_filename}">Click here to download PDF report</a>'

                # Show success message and download link
                st.success("PDF generated successfully!")
                st.markdown(href, unsafe_allow_html=True)

        # Add a quick risk overview for the selected bank
        st.header("Risk Overview")

        # Risk score for selected bank only
        risk_summary = {
            "JPMorgan": {"score": 2.8, "level": "LOW", "class": "risk-badge-low"},
            "UBS": {"score": 4.5, "level": "MEDIUM", "class": "risk-badge-medium"},
        }

        # Display risk info for selected bank
        if selected_bank in risk_summary:
            data = risk_summary[selected_bank]
            risk_html = f"""
            <div style="padding: 10px; background-color: #f5f5f5; border-radius: 5px; margin-top: 10px;">
                <div style="font-weight: bold; margin-bottom: 5px;">{selected_bank}</div>
                <div class="risk-badge {data["class"]}">{data["level"]}</div> Credit Risk Score: {data["score"]}/10
            </div>
            """
            st.markdown(risk_html, unsafe_allow_html=True)

        # Help section - simplified
        with st.expander("Dashboard Help"):
            st.markdown(
                """
                **How to use this dashboard:**
                
                * Select a bank to analyze
                * Choose a quarter to analyze
                * Enable peer comparison if needed
                * Generate a PDF report for documentation
                """
            )

    return selected_bank, selected_quarter, comparison_bank, show_comparison


# Function to get filtered data based on single bank and quarter selection
def get_filtered_data(data, selected_bank, selected_quarter):
    # Convert quarter format from "2024 Q1" to datetime for filtering
    quarter_mapping = {
        "2024 Q1": datetime(2024, 1, 1),
        "2024 Q2": datetime(2024, 4, 1),
        "2024 Q3": datetime(2024, 7, 1),
        "2024 Q4": datetime(2024, 10, 1),
    }

    selected_date = quarter_mapping.get(selected_quarter)

    # Filter data for single bank and quarter
    filtered_data = data[
        (data["Bank"] == selected_bank) & (data["Date"] == selected_date)
    ]

    return filtered_data


# Function to get historical data for a single bank (all quarters)
def get_historical_data(data, selected_bank):
    # Filter data for all quarters of the selected bank
    filtered_data = data[data["Bank"] == selected_bank]

    return filtered_data


# Function to get comparison data for two banks in a specific quarter
def get_comparison_data(data, selected_bank, comparison_bank, selected_quarter):
    # Convert quarter format from "2024 Q1" to datetime for filtering
    quarter_mapping = {
        "2024 Q1": datetime(2024, 1, 1),
        "2024 Q2": datetime(2024, 4, 1),
        "2024 Q3": datetime(2024, 7, 1),
        "2024 Q4": datetime(2024, 10, 1),
    }

    selected_date = quarter_mapping.get(selected_quarter)

    # Filter data for both banks in the selected quarter
    banks = [selected_bank, comparison_bank]
    filtered_data = data[(data["Bank"].isin(banks)) & (data["Date"] == selected_date)]

    return filtered_data


# Cache the data loading to improve performance - Simplified for cleaner dataset
@st.cache_data
def load_sample_data():
    """Load simplified sample data for Credit Risk analysis of UBS and JPMorgan"""
    # Generate data for all 4 quarters of 2024
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=91 * i) for i in range(4)]
    banks = ["JPMorgan", "UBS"]

    # Credit Risk related topics
    topics = [
        "Loan Loss Provisions",
        "NPL Ratio",
        "Credit Quality",
        "Lending Standards",
    ]

    # Generate sentiment data - simplified patterns
    sentiment_data = []

    # Predefined patterns to make data more consistent
    sentiment_patterns = {
        "JPMorgan": {
            "Loan Loss Provisions": [-0.1, -0.15, -0.2, -0.3],  # Worsening trend
            "NPL Ratio": [-0.05, -0.1, -0.15, -0.2],  # Slight worsening
            "Credit Quality": [0.3, 0.25, 0.2, 0.15],  # Declining but positive
            "Lending Standards": [0.2, 0.15, 0.1, 0.05],  # Declining trend
        },
        "UBS": {
            "Loan Loss Provisions": [-0.15, -0.2, -0.3, -0.4],  # Worse than JPM
            "NPL Ratio": [-0.1, -0.2, -0.25, -0.3],  # Worse than JPM
            "Credit Quality": [0.2, 0.15, 0.05, 0.0],  # Declining to neutral
            "Lending Standards": [0.15, 0.1, 0.0, -0.05],  # Declining to negative
        },
    }

    for bank in banks:
        for i, date in enumerate(dates):
            for topic in topics:
                # Get predefined sentiment score to create clear patterns
                sentiment_score = sentiment_patterns[bank][topic][i]

                # Generate mentions count
                mentions = np.random.randint(8, 20)  # Simplified range

                sentiment_data.append(
                    {
                        "Bank": bank,
                        "Date": date,
                        "Topic": topic,
                        "Sentiment_Score": sentiment_score,
                        "Sentiment_Category": "Positive"
                        if sentiment_score > 0.1
                        else "Negative"
                        if sentiment_score < -0.1
                        else "Neutral",
                        "Mentions": mentions,
                    }
                )

    sentiment_df = pd.DataFrame(sentiment_data)

    # Generate Credit Risk-related metrics with clear risk patterns
    metrics_data = []

    # Predefined metrics to create clearer patterns for risk identification
    metrics_patterns = {
        "JPMorgan": {
            "NPL_Ratio": [0.9, 1.0, 1.2, 1.3],  # Worsening trend
            "Coverage_Ratio": [170, 165, 160, 155],  # Declining trend
            "Provisions_Bn": [4.2, 4.4, 4.6, 4.8],  # Increasing provisions
            "CET1_Ratio": [13.5, 13.4, 13.2, 13.0],  # Declining trend
        },
        "UBS": {
            "NPL_Ratio": [1.1, 1.2, 1.4, 1.6],  # Worse than JPM
            "Coverage_Ratio": [150, 145, 140, 135],  # Worse than JPM
            "Provisions_Bn": [3.5, 3.8, 4.1, 4.5],  # Increasing more rapidly
            "CET1_Ratio": [13.0, 12.8, 12.5, 12.2],  # Declining faster than JPM
        },
    }

    for bank in banks:
        for i, date in enumerate(dates):
            metrics_data.append(
                {
                    "Bank": bank,
                    "Date": date,
                    "NPL_Ratio": metrics_patterns[bank]["NPL_Ratio"][i],
                    "Coverage_Ratio": metrics_patterns[bank]["Coverage_Ratio"][i],
                    "Provisions_Bn": metrics_patterns[bank]["Provisions_Bn"][i],
                    "CET1_Ratio": metrics_patterns[bank]["CET1_Ratio"][i],
                }
            )

    metrics_df = pd.DataFrame(metrics_data)

    # Generate speaker analysis data - simplified for essential roles only
    roles = ["CEO", "CFO", "CRO", "Analyst"]
    speaker_data = []

    # Simplify to just the most recent quarter
    recent_date = dates[-1]  # Q4 2024

    for bank in banks:
        for role in roles:
            # Set different speaking patterns by role
            if role == "CEO":
                speaking_time = 12
                statements = 8
                sentiment_score = (
                    0.3 if bank == "JPMorgan" else 0.2
                )  # CEOs are positive
            elif role == "CFO":
                speaking_time = 15
                statements = 12
                sentiment_score = 0.1 if bank == "JPMorgan" else 0.0  # CFOs are neutral
            elif role == "CRO":
                speaking_time = 10
                statements = 6
                sentiment_score = (
                    -0.1 if bank == "JPMorgan" else -0.2
                )  # CROs are cautious
            else:  # Analyst
                speaking_time = 3
                questions = 5
                sentiment_score = (
                    -0.2 if bank == "JPMorgan" else -0.3
                )  # Analysts ask challenging questions

            # Add data point
            if role == "Analyst":
                speaker_data.append(
                    {
                        "Bank": bank,
                        "Date": recent_date,
                        "Role": role,
                        "Speaking_Time": speaking_time,
                        "Questions": questions,
                        "Sentiment_Score": sentiment_score,
                    }
                )
            else:
                speaker_data.append(
                    {
                        "Bank": bank,
                        "Date": recent_date,
                        "Role": role,
                        "Speaking_Time": speaking_time,
                        "Statements": statements,
                        "Sentiment_Score": sentiment_score,
                    }
                )

    speaker_df = pd.DataFrame(speaker_data)

    # Create wordcloud text data for the selected topics
    wordcloud_data = {
        "Loan Loss Provisions": "provisions reserves allowances impairments coverage charge write-offs loss models IFRS9 CECL expected-credit-loss forecasting deterioration macroeconomic stress scenarios",
        "NPL Ratio": "non-performing loans NPL delinquency default past-due classified watchlist underperforming deteriorating criticized special-mention impaired restructured forbearance",
        "Credit Quality": "rating migration downgrade upgrade credit-quality borrower assessment risk-profile performance collateral recovery secured unsecured scoring monitoring",
        "Lending Standards": "underwriting criteria covenants tightening loosening origination standards documentation LTV debt-service-coverage ratios approval terms conditions",
    }

    # Simplified topic importance data
    topic_counts = {
        "Loan Loss Provisions": 120,
        "NPL Ratio": 105,
        "Credit Quality": 95,
        "Lending Standards": 80,
    }

    # Create a simpler topic DataFrame
    topic_df = pd.DataFrame(
        {
            "Name": list(topic_counts.keys()),
            "Count": list(topic_counts.values()),
        }
    )

    # Generate representative quotes for each topic - one per topic per bank
    quotes_data = {
        "JPMorgan": {
            "Loan Loss Provisions": "We've increased our loan loss provisions by 15% this quarter due to macroeconomic uncertainties.",
            "NPL Ratio": "Our NPL ratio increased to 1.3%, primarily driven by the commercial real estate portfolio.",
            "Credit Quality": "Overall credit quality remains acceptable, with increased rating downgrades across the portfolio.",
            "Lending Standards": "We've tightened our lending standards in response to economic uncertainties.",
        },
        "UBS": {
            "Loan Loss Provisions": "Our CECL models indicate a need for higher reserves in the commercial real estate sector.",
            "NPL Ratio": "We've seen continued deterioration in our non-performing loans, with the ratio increasing to 1.6%.",
            "Credit Quality": "We're seeing concerning signs of stress in certain consumer segments that require monitoring.",
            "Lending Standards": "Our LTV requirements have been adjusted upward for commercial real estate loans.",
        },
    }

    # LLM-generated insights for each bank and quarter - simplified for bank-centric approach
    llm_insights = {
        # JPMorgan insights by quarter
        (
            "JPMorgan",
            "2024 Q1",
        ): """JPMorgan's credit risk profile appears stable with NPL ratios at 0.9%. The bank maintains a strong coverage ratio of 170%, providing adequate buffer against potential losses. Commercial real estate exposure requires ongoing monitoring, with specific stress testing for office properties showing potential for slight increases in delinquencies.""",
        (
            "JPMorgan",
            "2024 Q2",
        ): """JPMorgan shows early signs of credit deterioration with NPL ratio increasing to 1.0% and coverage ratio declining to 165%. The C&I lending sector remains stable, but increasing provisions (up to $4.4bn) suggest management expects further weakening in credit conditions. Key risk indicator: loan demand remains muted except in the Card business.""",
        (
            "JPMorgan",
            "2024 Q3",
        ): """JPMorgan's credit metrics continue to weaken with NPL ratio now at 1.2% and coverage ratio dropping to 160%. Rising provisions and yield curve impacts are negatively affecting profitability. Risk alert: Declining CET1 ratio (13.2%) alongside higher NPLs indicates potential capital pressure if trends continue.""",
        (
            "JPMorgan",
            "2024 Q4",
        ): """JPMorgan shows clear credit deterioration with NPL ratio reaching 1.3% and coverage ratio at 155%. Critical risk indicator: Provisions increased 14.3% year-over-year to $4.8bn, signaling management expects further credit losses. Real estate sector particularly concerning, with rising interest rates increasing borrower default risk.""",
        # UBS insights by quarter
        (
            "UBS",
            "2024 Q1",
        ): """UBS's credit risk profile shows elevated metrics with an NPL ratio of 1.1% and lower coverage ratio of 150% compared to peers. The Credit Suisse integration remains a concern. Commercial real estate exposures, particularly office properties in urban centers, require enhanced monitoring as occupancy rates face pressure.""",
        (
            "UBS",
            "2024 Q2",
        ): """UBS shows continued credit quality deterioration with NPL ratio at 1.2% and declining coverage ratio at 145%. Risk alert: The bank's conservative approach to loan growth and increased focus on the Non-Core and Legacy division suggests internal concern about asset quality. Credit Suisse integration continues to present challenges.""",
        (
            "UBS",
            "2024 Q3",
        ): """UBS credit metrics show accelerating deterioration with NPL ratio reaching 1.4% and coverage ratio dropping to 140%. Despite derisking efforts and reduction in Risk-Weighted Assets, credit provisions continue to increase. Critical concern: CET1 ratio declining to 12.5% indicates weakening capital position against rising credit risk.""",
        (
            "UBS",
            "2024 Q4",
        ): """UBS shows significant credit risk with NPL ratio at 1.6% and coverage ratio falling to 135%. High risk alert: Provisions have increased 28.6% year-over-year to $4.5bn. The flat yield curve is limiting profitability while credit costs rise. Basel III compliance efforts and balance sheet optimization suggest management is actively trying to mitigate deteriorating fundamentals.""",
    }

    return (
        topic_df,
        sentiment_df,
        metrics_df,
        speaker_df,
        wordcloud_data,
        quotes_data,
        llm_insights,
    )


# Function to format quarter and year from datetime
def format_quarter_year(date):
    quarter = f"Q{(date.month - 1) // 3 + 1}"
    year = date.year
    return f"{year} {quarter}"


# Function to create word cloud from text data - simplified to return figure directly
def create_wordcloud(text_data, title):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="Blues",
        max_words=100,
    ).generate(text_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.set_title(title, fontsize=18)
    ax.axis("off")
    return fig


# Function to calculate risk score based on metrics
def calculate_risk_score(metrics_data):
    """Calculate a risk score based on key metrics"""
    if metrics_data.empty:
        return 0, "Low"

    # Extract key metrics
    npl_ratio = metrics_data["NPL_Ratio"].values[0]
    coverage_ratio = metrics_data["Coverage_Ratio"].values[0]
    cet1_ratio = metrics_data["CET1_Ratio"].values[0]

    # Simple weighted formula for risk score (0-10 scale)
    # Higher NPL is worse, lower coverage is worse, lower CET1 is worse
    risk_score = (npl_ratio * 4) + (200 - coverage_ratio) / 10 + (14 - cet1_ratio) * 2

    # Scale to 0-10 range
    risk_score = max(min(risk_score / 10, 10), 0)

    # Determine risk category
    if risk_score > 6:
        risk_category = "High"
    elif risk_score > 3.5:
        risk_category = "Medium"
    else:
        risk_category = "Low"

    return risk_score, risk_category


# Redesigned Executive Summary function - focused on a single bank
def display_executive_summary(
    selected_bank, selected_quarter, metrics_df, sentiment_df, llm_insights, quotes_data
):
    st.header(f"{selected_bank} - Executive Summary")

    # Convert quarter format to datetime
    quarter_mapping = {
        "2024 Q1": datetime(2024, 1, 1),
        "2024 Q2": datetime(2024, 4, 1),
        "2024 Q3": datetime(2024, 7, 1),
        "2024 Q4": datetime(2024, 10, 1),
    }
    selected_date = quarter_mapping.get(selected_quarter)

    # Get current quarter data
    current_data = metrics_df[
        (metrics_df["Bank"] == selected_bank) & (metrics_df["Date"] == selected_date)
    ]

    if current_data.empty:
        st.warning(f"No data available for {selected_bank} in {selected_quarter}")
        return

    # Get historical data for trend calculation
    historical_data = metrics_df[metrics_df["Bank"] == selected_bank].sort_values(
        "Date"
    )

    # Calculate risk score
    risk_score, risk_category = calculate_risk_score(current_data)

    # Create risk color based on category
    if risk_category == "High":
        risk_color = "#d62728"
        risk_badge_color = "risk-badge-high"
    elif risk_category == "Medium":
        risk_color = "#ff7f0e"
        risk_badge_color = "risk-badge-medium"
    else:
        risk_color = "#2ca02c"
        risk_badge_color = "risk-badge-low"

    # Create 3-column layout for key metrics
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Create a card with bank name and risk badge
        st.markdown(
            f"""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 20px; border-top: 5px solid {risk_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h3 style="margin: 0;">{selected_bank} - {selected_quarter}</h3>
                    <div class="risk-badge {risk_badge_color}">{risk_category.upper()} RISK</div>
                </div>
                <p>Risk Score: {risk_score:.1f}/10</p>
                <div style="height: 10px; background-color: #f0f2f6; border-radius: 5px; margin: 10px 0;">
                    <div style="height: 100%; width: {risk_score * 10}%; background-color: {risk_color}; border-radius: 5px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display AI-generated insight
        insight_key = (selected_bank, selected_quarter)
        if insight_key in llm_insights:
            st.subheader("AI Analysis")
            st.markdown(llm_insights[insight_key])

    with col2:
        # Display key metrics
        st.subheader("Key Metrics")

        # Get current values
        npl_ratio = current_data["NPL_Ratio"].values[0]
        coverage_ratio = current_data["Coverage_Ratio"].values[0]
        provisions_bn = current_data["Provisions_Bn"].values[0]
        cet1_ratio = current_data["CET1_Ratio"].values[0]

        # Get first quarter values for trend (if available)
        if len(historical_data) > 1:
            first_data = historical_data.iloc[0]
            first_npl = first_data["NPL_Ratio"]
            first_coverage = first_data["Coverage_Ratio"]
            first_provisions = first_data["Provisions_Bn"]
            first_cet1 = first_data["CET1_Ratio"]

            # Calculate percent changes
            npl_change = ((npl_ratio - first_npl) / first_npl) * 100
            coverage_change = ((coverage_ratio - first_coverage) / first_coverage) * 100
            provisions_change = (
                (provisions_bn - first_provisions) / first_provisions
            ) * 100
            cet1_change = ((cet1_ratio - first_cet1) / first_cet1) * 100

            # Create metrics with delta values
            st.metric("NPL Ratio", f"{npl_ratio:.2f}%", f"{npl_change:.1f}%")
            st.metric(
                "Coverage Ratio", f"{coverage_ratio:.0f}%", f"{coverage_change:.1f}%"
            )
            st.metric(
                "Provisions", f"${provisions_bn:.1f}Bn", f"{provisions_change:.1f}%"
            )
            st.metric("CET1 Ratio", f"{cet1_ratio:.2f}%", f"{cet1_change:.1f}%")
        else:
            # Just show current values without trends
            st.metric("NPL Ratio", f"{npl_ratio:.2f}%")
            st.metric("Coverage Ratio", f"{coverage_ratio:.0f}%")
            st.metric("Provisions", f"${provisions_bn:.1f}Bn")
            st.metric("CET1 Ratio", f"{cet1_ratio:.2f}%")

    with col3:
        # Show key quote from the bank
        st.subheader("Key Statement")
        if selected_bank in quotes_data:
            bank_quotes = quotes_data[selected_bank]
            # Show NPL ratio quote as most representative
            if "NPL Ratio" in bank_quotes:
                quote = bank_quotes["NPL Ratio"]
                st.markdown(
                    f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; 
                         border-left: 3px solid {risk_color}; font-style: italic; margin-top: 15px;">
                        "{quote}"
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Display trend chart for key metrics
    st.subheader("Quarterly Trend")

    if len(historical_data) > 1:
        # Create a line chart for NPL Ratio and Coverage Ratio
        fig = go.Figure()

        # Add NPL Ratio line
        fig.add_trace(
            go.Scatter(
                x=historical_data["Date"],
                y=historical_data["NPL_Ratio"],
                mode="lines+markers",
                name="NPL Ratio (%)",
                line=dict(color="#1a2e44", width=3),
            )
        )

        # Add Coverage Ratio on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=historical_data["Date"],
                y=historical_data["Coverage_Ratio"],
                mode="lines+markers",
                name="Coverage Ratio (%)",
                line=dict(color="#ff7f0e", width=3, dash="dot"),
                yaxis="y2",
            )
        )

        # Update layout with dual y-axes
        fig.update_layout(
            title=f"{selected_bank} - Quarterly Trend Analysis",
            xaxis_title="Quarter",
            yaxis=dict(
                title="NPL Ratio (%)",
                titlefont=dict(color="#1a2e44"),
                tickfont=dict(color="#1a2e44"),
            ),
            yaxis2=dict(
                title="Coverage Ratio (%)",
                titlefont=dict(color="#ff7f0e"),
                tickfont=dict(color="#ff7f0e"),
                anchor="x",
                overlaying="y",
                side="right",
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )

        # Add quarter labels to x-axis
        quarter_labels = [format_quarter_year(date) for date in historical_data["Date"]]
        fig.update_xaxes(tickvals=historical_data["Date"], ticktext=quarter_labels)

        st.plotly_chart(fig, use_container_width=True)


# Streamlined Topic Analysis function focused on a single bank
def display_topic_analysis(
    selected_bank, selected_quarter, topic_df, sentiment_df, wordcloud_data, quotes_data
):
    st.header(f"{selected_bank} - Topic Analysis")

    # Convert quarter format to datetime
    quarter_mapping = {
        "2024 Q1": datetime(2024, 1, 1),
        "2024 Q2": datetime(2024, 4, 1),
        "2024 Q3": datetime(2024, 7, 1),
        "2024 Q4": datetime(2024, 10, 1),
    }
    selected_date = quarter_mapping.get(selected_quarter)

    # Filter sentiment data for the selected bank and quarter
    filtered_sentiment = sentiment_df[
        (sentiment_df["Bank"] == selected_bank)
        & (sentiment_df["Date"] == selected_date)
    ]

    if filtered_sentiment.empty:
        st.warning(f"No topic data available for {selected_bank} in {selected_quarter}")
        return

    # Create a horizontal bar chart for topic counts
    fig_counts = px.bar(
        topic_df,
        y="Name",
        x="Count",
        orientation="h",
        labels={"Name": "Topic", "Count": "Mention Count"},
        color="Count",
        color_continuous_scale="Blues",
        title=f"{selected_bank} - Topic Mention Frequency",
    )

    fig_counts.update_layout(height=300)
    st.plotly_chart(fig_counts, use_container_width=True)

    # Create grid for wordclouds of all topics
    topics = topic_df["Name"].tolist()

    # Create two columns for wordclouds
    col1, col2 = st.columns(2)

    # Display wordclouds for all topics in grid layout
    for i, topic in enumerate(topics):
        # Alternate between columns
        column = col1 if i % 2 == 0 else col2

        with column:
            st.subheader(topic)

            # Generate and display wordcloud
            if topic in wordcloud_data:
                wordcloud_fig = create_wordcloud(wordcloud_data[topic], "")
                st.pyplot(wordcloud_fig)

            # Calculate topic sentiment for this bank and quarter
            topic_sentiment = filtered_sentiment[filtered_sentiment["Topic"] == topic]
            if not topic_sentiment.empty:
                avg_sentiment = topic_sentiment["Sentiment_Score"].mean()
                sentiment_color = (
                    "#2ca02c"
                    if avg_sentiment > 0.1
                    else "#d62728"
                    if avg_sentiment < -0.1
                    else "#1f77b4"
                )

                # Display sentiment score
                st.markdown(
                    f"""
                    <div style="margin-top: -20px; margin-bottom: 20px;">
                        <span style="font-weight: bold;">Sentiment Score:</span> 
                        <span style="color: {sentiment_color};">{avg_sentiment:.2f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Display bank's quote for this topic
                if selected_bank in quotes_data and topic in quotes_data[selected_bank]:
                    quote = quotes_data[selected_bank][topic]
                    st.markdown(
                        f"""
                        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; 
                             border-left: 3px solid {sentiment_color}; font-style: italic; margin-bottom: 20px;">
                            "{quote}"
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Display sentiment comparison across topics
    st.subheader(f"{selected_bank} - Topic Sentiment Comparison")

    # Calculate average sentiment for each topic
    topic_sentiments = (
        filtered_sentiment.groupby("Topic")["Sentiment_Score"].mean().reset_index()
    )

    # Create a horizontal bar chart for sentiment scores
    fig_sentiment = px.bar(
        topic_sentiments,
        y="Topic",
        x="Sentiment_Score",
        orientation="h",
        color="Sentiment_Score",
        color_continuous_scale="RdBu",
        range_color=[-0.5, 0.5],
        title=f"{selected_bank} - {selected_quarter} Topic Sentiment Analysis",
        labels={"Sentiment_Score": "Average Sentiment Score", "Topic": ""},
    )

    # Add reference lines
    fig_sentiment.add_vline(x=0.1, line_dash="dash", line_color="green")
    fig_sentiment.add_vline(x=-0.1, line_dash="dash", line_color="red")
    fig_sentiment.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1)

    fig_sentiment.update_layout(height=300)
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # Display historical sentiment trends for this bank
    st.subheader(f"{selected_bank} - Topic Sentiment Trends")

    # Get all sentiment data for this bank (across all quarters)
    bank_sentiment = sentiment_df[sentiment_df["Bank"] == selected_bank]

    if len(bank_sentiment["Date"].unique()) > 1:
        # Create line chart showing sentiment over time for all topics
        fig_trends = px.line(
            bank_sentiment,
            x="Date",
            y="Sentiment_Score",
            color="Topic",
            markers=True,
            title=f"{selected_bank} - Topic Sentiment Trends Over Time",
            labels={"Sentiment_Score": "Average Sentiment", "Date": "Quarter"},
        )

        # Add reference lines
        fig_trends.add_hline(y=0.1, line_dash="dash", line_color="green")
        fig_trends.add_hline(y=-0.1, line_dash="dash", line_color="red")
        fig_trends.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)

        # Add quarter labels to x-axis
        quarter_labels = [
            format_quarter_year(date)
            for date in sorted(bank_sentiment["Date"].unique())
        ]
        quarter_dates = sorted(bank_sentiment["Date"].unique())
        fig_trends.update_xaxes(tickvals=quarter_dates, ticktext=quarter_labels)

        fig_trends.update_layout(height=400)
        st.plotly_chart(fig_trends, use_container_width=True)


# Simplified Sentiment Analysis function focused on a single bank
def display_sentiment_analysis(
    selected_bank, selected_quarter, sentiment_df, speaker_df
):
    st.header(f"{selected_bank} - Sentiment Analysis")

    # Convert quarter format to datetime
    quarter_mapping = {
        "2024 Q1": datetime(2024, 1, 1),
        "2024 Q2": datetime(2024, 4, 1),
        "2024 Q3": datetime(2024, 7, 1),
        "2024 Q4": datetime(2024, 10, 1),
    }
    selected_date = quarter_mapping.get(selected_quarter)

    # Filter sentiment data for the selected bank
    bank_sentiment = sentiment_df[sentiment_df["Bank"] == selected_bank]
    current_sentiment = bank_sentiment[bank_sentiment["Date"] == selected_date]

    if bank_sentiment.empty:
        st.warning(f"No sentiment data available for {selected_bank}")
        return

    # Create two columns layout
    col1, col2 = st.columns(2)

    with col1:
        # Display overall sentiment score
        st.subheader("Overall Sentiment")

        # Calculate average sentiment for the current quarter
        current_avg_sentiment = current_sentiment["Sentiment_Score"].mean()

        # Determine sentiment category and color
        if current_avg_sentiment > 0.1:
            sentiment_category = "Positive"
            sentiment_color = "#2ca02c"
        elif current_avg_sentiment < -0.1:
            sentiment_category = "Negative"
            sentiment_color = "#d62728"
        else:
            sentiment_category = "Neutral"
            sentiment_color = "#1f77b4"

        # Display sentiment score with gauge chart
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=current_avg_sentiment,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Sentiment Score"},
                gauge={
                    "axis": {"range": [-0.5, 0.5], "tickwidth": 1},
                    "bar": {"color": sentiment_color},
                    "steps": [
                        {"range": [-0.5, -0.1], "color": "#ffcccb"},
                        {"range": [-0.1, 0.1], "color": "#f0f0f0"},
                        {"range": [0.1, 0.5], "color": "#d4f1d4"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 2},
                        "thickness": 0.75,
                        "value": current_avg_sentiment,
                    },
                },
            )
        )

        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

        # Display sentiment trend over time
        if len(bank_sentiment["Date"].unique()) > 1:
            st.subheader("Sentiment Trend")

            # Calculate average sentiment by date
            sentiment_over_time = (
                bank_sentiment.groupby("Date")["Sentiment_Score"].mean().reset_index()
            )

            # Create line chart
            fig_trend = px.line(
                sentiment_over_time,
                x="Date",
                y="Sentiment_Score",
                markers=True,
                title=f"{selected_bank} - Sentiment Trend",
                labels={"Sentiment_Score": "Average Sentiment", "Date": "Quarter"},
            )

            # Add reference lines
            fig_trend.add_hline(y=0.1, line_dash="dash", line_color="green")
            fig_trend.add_hline(y=-0.1, line_dash="dash", line_color="red")
            fig_trend.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)

            # Add quarter labels to x-axis
            quarter_labels = [
                format_quarter_year(date)
                for date in sorted(bank_sentiment["Date"].unique())
            ]
            quarter_dates = sorted(bank_sentiment["Date"].unique())
            fig_trend.update_xaxes(tickvals=quarter_dates, ticktext=quarter_labels)

            fig_trend.update_layout(height=250)
            st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        # Display sentiment by role (for most recent quarter)
        st.subheader("Sentiment by Role")

        # Filter speaker data for the current bank
        bank_speakers = speaker_df[speaker_df["Bank"] == selected_bank]

        if not bank_speakers.empty:
            # Create bar chart for speaker roles
            fig_roles = px.bar(
                bank_speakers,
                x="Role",
                y="Sentiment_Score",
                color="Sentiment_Score",
                color_continuous_scale="RdBu",
                range_color=[-0.5, 0.5],
                title=f"{selected_bank} - Sentiment by Role ({selected_quarter})",
                labels={"Sentiment_Score": "Sentiment Score", "Role": ""},
            )

            # Add reference lines
            fig_roles.add_hline(y=0.1, line_dash="dash", line_color="green")
            fig_roles.add_hline(y=-0.1, line_dash="dash", line_color="red")
            fig_roles.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)

            fig_roles.update_layout(height=250)
            st.plotly_chart(fig_roles, use_container_width=True)

            # Calculate CEO-CRO gap
            if (
                "CEO" in bank_speakers["Role"].values
                and "CRO" in bank_speakers["Role"].values
            ):
                ceo_sentiment = bank_speakers[bank_speakers["Role"] == "CEO"][
                    "Sentiment_Score"
                ].values[0]
                cro_sentiment = bank_speakers[bank_speakers["Role"] == "CRO"][
                    "Sentiment_Score"
                ].values[0]
                sentiment_gap = ceo_sentiment - cro_sentiment

                # Determine risk level based on gap
                if sentiment_gap > 0.3:
                    gap_risk = "High"
                    gap_color = "#d62728"
                elif sentiment_gap > 0.2:
                    gap_risk = "Medium"
                    gap_color = "#ff7f0e"
                else:
                    gap_risk = "Low"
                    gap_color = "#2ca02c"

                # Display gap information
                st.subheader("CEO-CRO Sentiment Gap")
                st.markdown(
                    f"""
                    <div style="background-color: white; border-radius: 5px; padding: 15px; 
                         box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid {gap_color};">
                        <p>CEO Sentiment: {ceo_sentiment:.2f}</p>
                        <p>CRO Sentiment: {cro_sentiment:.2f}</p>
                        <p>Gap: <strong>{sentiment_gap:.2f}</strong></p>
                        <div style="margin-top: 10px;">
                            <span style="background-color: {gap_color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px;">
                                {gap_risk} RISK
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No speaker role data available for sentiment analysis.")

    # Display sentiment distribution for the selected bank
    st.subheader("Sentiment Distribution")

    # Create a histogram of sentiment scores
    fig_dist = px.histogram(
        current_sentiment,
        x="Sentiment_Score",
        nbins=20,
        color_discrete_sequence=["#1a2e44"],
        title=f"{selected_bank} - {selected_quarter} Sentiment Distribution",
        labels={"Sentiment_Score": "Sentiment Score", "count": "Frequency"},
    )

    # Add reference lines
    fig_dist.add_vline(x=0.1, line_dash="dash", line_color="green")
    fig_dist.add_vline(x=-0.1, line_dash="dash", line_color="red")
    fig_dist.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1)

    # Add mean line
    fig_dist.add_vline(
        x=current_avg_sentiment,
        line_dash="solid",
        line_width=2,
        line_color="orange",
        annotation_text=f"Mean: {current_avg_sentiment:.2f}",
        annotation_position="top right",
    )

    fig_dist.update_layout(height=300)
    st.plotly_chart(fig_dist, use_container_width=True)


# Revised Advanced Analytics function with optional peer comparison
def display_advanced_analytics(
    selected_bank, selected_quarter, comparison_bank, metrics_df, sentiment_df
):
    st.header("Advanced Analytics")

    # Convert quarter format to datetime
    quarter_mapping = {
        "2024 Q1": datetime(2024, 1, 1),
        "2024 Q2": datetime(2024, 4, 1),
        "2024 Q3": datetime(2024, 7, 1),
        "2024 Q4": datetime(2024, 10, 1),
    }
    selected_date = quarter_mapping.get(selected_quarter)

    # Get data for selected bank
    bank_data = metrics_df[metrics_df["Bank"] == selected_bank]
    current_bank_data = bank_data[bank_data["Date"] == selected_date]

    if current_bank_data.empty:
        st.warning(f"No data available for {selected_bank} in {selected_quarter}")
        return

    # Display peer comparison section only if comparison_bank is provided
    if comparison_bank:
        st.subheader(f"Peer Comparison: {selected_bank} vs {comparison_bank}")

        # Get data for comparison bank
        comparison_data = metrics_df[metrics_df["Bank"] == comparison_bank]
        current_comparison_data = comparison_data[
            comparison_data["Date"] == selected_date
        ]

        if current_comparison_data.empty:
            st.warning(f"No data available for {comparison_bank} in {selected_quarter}")
        else:
            # Create side-by-side metrics comparison
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {selected_bank}")
                npl_ratio = current_bank_data["NPL_Ratio"].values[0]
                coverage_ratio = current_bank_data["Coverage_Ratio"].values[0]
                provisions_bn = current_bank_data["Provisions_Bn"].values[0]
                cet1_ratio = current_bank_data["CET1_Ratio"].values[0]

                st.metric("NPL Ratio", f"{npl_ratio:.2f}%")
                st.metric("Coverage Ratio", f"{coverage_ratio:.0f}%")
                st.metric("Provisions", f"${provisions_bn:.1f}Bn")
                st.metric("CET1 Ratio", f"{cet1_ratio:.2f}%")

            with col2:
                st.markdown(f"### {comparison_bank}")
                comp_npl_ratio = current_comparison_data["NPL_Ratio"].values[0]
                comp_coverage_ratio = current_comparison_data["Coverage_Ratio"].values[
                    0
                ]
                comp_provisions_bn = current_comparison_data["Provisions_Bn"].values[0]
                comp_cet1_ratio = current_comparison_data["CET1_Ratio"].values[0]

                # Calculate differences for delta values
                npl_diff = comp_npl_ratio - npl_ratio
                coverage_diff = comp_coverage_ratio - coverage_ratio
                provisions_diff = comp_provisions_bn - provisions_bn
                cet1_diff = comp_cet1_ratio - cet1_ratio

                # Display metrics with comparison deltas
                st.metric("NPL Ratio", f"{comp_npl_ratio:.2f}%", f"{npl_diff:+.2f}%")
                st.metric(
                    "Coverage Ratio",
                    f"{comp_coverage_ratio:.0f}%",
                    f"{coverage_diff:+.0f}%",
                )
                st.metric(
                    "Provisions",
                    f"${comp_provisions_bn:.1f}Bn",
                    f"{provisions_diff:+.1f}Bn",
                )
                st.metric("CET1 Ratio", f"{comp_cet1_ratio:.2f}%", f"{cet1_diff:+.2f}%")

            # Create radar chart for comparing all metrics
            st.subheader("Metrics Comparison")

            # Define metrics for radar chart and their normalization ranges
            metrics = {
                "NPL_Ratio": {
                    "title": "NPL Ratio",
                    "worst": 2.0,
                    "best": 0.5,
                    "inverse": True,
                },
                "Coverage_Ratio": {
                    "title": "Coverage Ratio",
                    "worst": 120,
                    "best": 180,
                    "inverse": False,
                },
                "Provisions_Bn": {
                    "title": "Provisions",
                    "worst": 3.0,
                    "best": 5.0,
                    "inverse": False,
                },
                "CET1_Ratio": {
                    "title": "CET1 Ratio",
                    "worst": 12.0,
                    "best": 14.0,
                    "inverse": False,
                },
            }

            # Create radar chart
            fig = go.Figure()

            # Function to normalize metrics to 0-1 scale (1 is better)
            def normalize_metrics(data, metrics_config):
                radar_values = []
                radar_labels = []

                for metric, config in metrics_config.items():
                    value = data[metric].values[0]
                    # Normalize to 0-1 scale where 1 is better
                    if config["inverse"]:  # Lower is better (like NPL)
                        normalized = 1 - (value - config["best"]) / (
                            config["worst"] - config["best"]
                        )
                    else:  # Higher is better (like Coverage)
                        normalized = (value - config["worst"]) / (
                            config["best"] - config["worst"]
                        )

                    # Clip to 0-1 range
                    normalized = max(0, min(1, normalized))
                    radar_values.append(normalized)
                    radar_labels.append(config["title"])

                # Add an extra point to close the loop
                radar_values.append(radar_values[0])
                radar_labels.append(radar_labels[0])

                return radar_values, radar_labels

            # Add selected bank to radar chart
            selected_values, radar_labels = normalize_metrics(
                current_bank_data, metrics
            )
            fig.add_trace(
                go.Scatterpolar(
                    r=selected_values,
                    theta=radar_labels,
                    fill="toself",
                    name=selected_bank,
                    line_color="#1a2e44",
                    fillcolor="#1a2e44",
                    opacity=0.6,
                )
            )

            # Add comparison bank to radar chart
            comparison_values, _ = normalize_metrics(current_comparison_data, metrics)
            fig.add_trace(
                go.Scatterpolar(
                    r=comparison_values,
                    theta=radar_labels,
                    fill="toself",
                    name=comparison_bank,
                    line_color="#8B0000",
                    fillcolor="#8B0000",
                    opacity=0.6,
                )
            )

            # Update layout
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=500,
                title=f"Credit Risk Profile Comparison ({selected_quarter})",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Compare sentiment across both banks
            st.subheader("Sentiment Comparison")

            # Get sentiment data for both banks
            selected_sentiment = sentiment_df[
                (sentiment_df["Bank"] == selected_bank)
                & (sentiment_df["Date"] == selected_date)
            ]
            comparison_sentiment = sentiment_df[
                (sentiment_df["Bank"] == comparison_bank)
                & (sentiment_df["Date"] == selected_date)
            ]

            if not selected_sentiment.empty and not comparison_sentiment.empty:
                # Calculate average sentiment by topic for both banks
                selected_topic_sent = (
                    selected_sentiment.groupby("Topic")["Sentiment_Score"]
                    .mean()
                    .reset_index()
                )
                selected_topic_sent["Bank"] = selected_bank

                comparison_topic_sent = (
                    comparison_sentiment.groupby("Topic")["Sentiment_Score"]
                    .mean()
                    .reset_index()
                )
                comparison_topic_sent["Bank"] = comparison_bank

                # Combine data
                combined_sentiment = pd.concat(
                    [selected_topic_sent, comparison_topic_sent]
                )

                # Create grouped bar chart
                fig_sent = px.bar(
                    combined_sentiment,
                    x="Topic",
                    y="Sentiment_Score",
                    color="Bank",
                    barmode="group",
                    title=f"Topic Sentiment Comparison ({selected_quarter})",
                    color_discrete_map={
                        selected_bank: "#1a2e44",
                        comparison_bank: "#8B0000",
                    },
                    labels={"Sentiment_Score": "Average Sentiment", "Topic": ""},
                )

                # Add reference lines
                fig_sent.add_hline(y=0.1, line_dash="dash", line_color="green")
                fig_sent.add_hline(y=-0.1, line_dash="dash", line_color="red")
                fig_sent.add_hline(
                    y=0, line_dash="solid", line_color="gray", line_width=1
                )

                fig_sent.update_layout(height=400)
                st.plotly_chart(fig_sent, use_container_width=True)

                # Calculate overall sentiment difference
                selected_avg = selected_sentiment["Sentiment_Score"].mean()
                comparison_avg = comparison_sentiment["Sentiment_Score"].mean()
                sentiment_diff = abs(selected_avg - comparison_avg)

                # Show significant differences if they exist
                if sentiment_diff > 0.1:
                    more_negative = (
                        selected_bank
                        if selected_avg < comparison_avg
                        else comparison_bank
                    )
                    st.markdown(
                        f"""
                        <div class="risk-warning">
                        <strong>Significant Sentiment Difference:</strong> {more_negative} shows notably more
                        negative sentiment ({sentiment_diff:.2f} difference), which may indicate higher
                        internal concern about credit conditions.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Temporal Analysis section - Always shown regardless of comparison mode
    st.subheader(f"{selected_bank} - Temporal Analysis")

    # Get historical data for selected bank
    historical_data = bank_data.sort_values("Date")

    if len(historical_data) > 1:
        # Create tabs for different metrics
        metric_tabs = st.tabs(
            ["NPL Ratio", "Coverage Ratio", "Provisions", "CET1 Ratio"]
        )

        with metric_tabs[0]:
            # NPL Ratio trend analysis
            fig_npl = px.line(
                historical_data,
                x="Date",
                y="NPL_Ratio",
                markers=True,
                title=f"{selected_bank} - NPL Ratio Trend",
                labels={"NPL_Ratio": "NPL Ratio (%)", "Date": "Quarter"},
            )

            # Add reference lines for NPL
            fig_npl.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="#ff7f0e",
                annotation_text="Medium Risk",
            )
            fig_npl.add_hline(
                y=1.2,
                line_dash="dash",
                line_color="#d62728",
                annotation_text="High Risk",
            )

            # Format x-axis with quarter labels
            quarter_labels = [
                format_quarter_year(date) for date in historical_data["Date"]
            ]
            fig_npl.update_xaxes(
                tickvals=historical_data["Date"].tolist(), ticktext=quarter_labels
            )

            fig_npl.update_layout(height=350)
            st.plotly_chart(fig_npl, use_container_width=True)

        with metric_tabs[1]:
            # Coverage Ratio trend analysis
            fig_cov = px.line(
                historical_data,
                x="Date",
                y="Coverage_Ratio",
                markers=True,
                title=f"{selected_bank} - Coverage Ratio Trend",
                labels={"Coverage_Ratio": "Coverage Ratio (%)", "Date": "Quarter"},
            )

            # Add reference lines for Coverage
            fig_cov.add_hline(
                y=150,
                line_dash="dash",
                line_color="#2ca02c",
                annotation_text="Adequate",
            )
            fig_cov.add_hline(
                y=130,
                line_dash="dash",
                line_color="#d62728",
                annotation_text="Insufficient",
            )

            # Format x-axis with quarter labels
            fig_cov.update_xaxes(
                tickvals=historical_data["Date"].tolist(), ticktext=quarter_labels
            )

            fig_cov.update_layout(height=350)
            st.plotly_chart(fig_cov, use_container_width=True)

        with metric_tabs[2]:
            # Provisions trend analysis
            fig_prov = px.line(
                historical_data,
                x="Date",
                y="Provisions_Bn",
                markers=True,
                title=f"{selected_bank} - Provisions Trend",
                labels={"Provisions_Bn": "Provisions ($ Bn)", "Date": "Quarter"},
            )

            # Format x-axis with quarter labels
            fig_prov.update_xaxes(
                tickvals=historical_data["Date"].tolist(), ticktext=quarter_labels
            )

            fig_prov.update_layout(height=350)
            st.plotly_chart(fig_prov, use_container_width=True)

        with metric_tabs[3]:
            # CET1 Ratio trend analysis
            fig_cet1 = px.line(
                historical_data,
                x="Date",
                y="CET1_Ratio",
                markers=True,
                title=f"{selected_bank} - CET1 Ratio Trend",
                labels={"CET1_Ratio": "CET1 Ratio (%)", "Date": "Quarter"},
            )

            # Add reference lines for CET1
            fig_cet1.add_hline(
                y=13.0,
                line_dash="dash",
                line_color="#2ca02c",
                annotation_text="Well Capitalized",
            )
            fig_cet1.add_hline(
                y=11.0,
                line_dash="dash",
                line_color="#d62728",
                annotation_text="Regulatory Minimum + Buffer",
            )

            # Format x-axis with quarter labels
            fig_cet1.update_xaxes(
                tickvals=historical_data["Date"].tolist(), ticktext=quarter_labels
            )

            fig_cet1.update_layout(height=350)
            st.plotly_chart(fig_cet1, use_container_width=True)

        # Add trend analysis summary
        st.subheader("Trend Analysis Summary")

        # Calculate changes from first to last quarter
        first_data = historical_data.iloc[0]
        last_data = historical_data.iloc[-1]

        npl_change = (
            (last_data["NPL_Ratio"] - first_data["NPL_Ratio"]) / first_data["NPL_Ratio"]
        ) * 100
        coverage_change = (
            (last_data["Coverage_Ratio"] - first_data["Coverage_Ratio"])
            / first_data["Coverage_Ratio"]
        ) * 100
        provisions_change = (
            (last_data["Provisions_Bn"] - first_data["Provisions_Bn"])
            / first_data["Provisions_Bn"]
        ) * 100
        cet1_change = (
            (last_data["CET1_Ratio"] - first_data["CET1_Ratio"])
            / first_data["CET1_Ratio"]
        ) * 100

        # Determine trend direction and risk for each metric
        npl_risk = "High" if npl_change > 20 else "Medium" if npl_change > 10 else "Low"
        npl_color = (
            "#d62728"
            if npl_risk == "High"
            else "#ff7f0e"
            if npl_risk == "Medium"
            else "#2ca02c"
        )

        coverage_risk = (
            "High"
            if coverage_change < -10
            else "Medium"
            if coverage_change < -5
            else "Low"
        )
        coverage_color = (
            "#d62728"
            if coverage_risk == "High"
            else "#ff7f0e"
            if coverage_risk == "Medium"
            else "#2ca02c"
        )

        cet1_risk = (
            "High" if cet1_change < -5 else "Medium" if cet1_change < -2 else "Low"
        )
        cet1_color = (
            "#d62728"
            if cet1_risk == "High"
            else "#ff7f0e"
            if cet1_risk == "Medium"
            else "#2ca02c"
        )

        # Display trend summary in a clean format
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div style="background-color: white; border-radius: 5px; padding: 15px; 
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px;">
                    <h4>NPL Ratio</h4>
                    <p>Change: <span style="color: {npl_color};">{npl_change:.1f}%</span></p>
                    <p>Initial: {first_data["NPL_Ratio"]:.2f}% → Current: {last_data["NPL_Ratio"]:.2f}%</p>
                    <div style="margin-top: 5px;">
                        <span style="background-color: {npl_color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px;">
                            {npl_risk} RISK
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="background-color: white; border-radius: 5px; padding: 15px; 
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h4>CET1 Ratio</h4>
                    <p>Change: <span style="color: {cet1_color};">{cet1_change:.1f}%</span></p>
                    <p>Initial: {first_data["CET1_Ratio"]:.2f}% → Current: {last_data["CET1_Ratio"]:.2f}%</p>
                    <div style="margin-top: 5px;">
                        <span style="background-color: {cet1_color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px;">
                            {cet1_risk} RISK
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="background-color: white; border-radius: 5px; padding: 15px; 
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px;">
                    <h4>Coverage Ratio</h4>
                    <p>Change: <span style="color: {coverage_color};">{coverage_change:.1f}%</span></p>
                    <p>Initial: {first_data["Coverage_Ratio"]:.0f}% → Current: {last_data["Coverage_Ratio"]:.0f}%</p>
                    <div style="margin-top: 5px;">
                        <span style="background-color: {coverage_color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px;">
                            {coverage_risk} RISK
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="background-color: white; border-radius: 5px; padding: 15px; 
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h4>Provisions</h4>
                    <p>Change: <span style="color: {"#1f77b4"};">{provisions_change:.1f}%</span></p>
                    <p>Initial: ${first_data["Provisions_Bn"]:.1f}Bn → Current: ${last_data["Provisions_Bn"]:.1f}Bn</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Not enough historical data available for trend analysis.")


# Updated export report function to focus on a single bank with optional comparison
def export_report_as_pdf(selected_bank, selected_quarter, comparison_bank=None):
    """Create a simplified PDF report focused on a single bank"""
    pdf_buffer = BytesIO()

    # Convert quarter format to datetime
    quarter_mapping = {
        "2024 Q1": datetime(2024, 1, 1),
        "2024 Q2": datetime(2024, 4, 1),
        "2024 Q3": datetime(2024, 7, 1),
        "2024 Q4": datetime(2024, 10, 1),
    }
    selected_date = quarter_mapping.get(selected_quarter)

    # Set up report title
    report_title = f"{selected_bank} Credit Risk Analysis - {selected_quarter}"
    if comparison_bank:
        report_title += f" (with {comparison_bank} comparison)"

    # Create PDF
    with PdfPages(pdf_buffer) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(
            0.5,
            0.85,
            "G-SIB Credit Risk Analysis",
            fontsize=24,
            ha="center",
            fontweight="bold",
        )
        ax.text(0.5, 0.75, report_title, fontsize=20, ha="center")
        ax.text(
            0.5,
            0.65,
            "Academic Project - Quarterly Analysis Report",
            fontsize=16,
            ha="center",
        )
        ax.text(
            0.5,
            0.60,
            "(Simulated Data)",
            fontsize=14,
            ha="center",
            style="italic",
            color="gray",
        )
        ax.text(
            0.5,
            0.55,
            f"Report Date: {datetime.now().strftime('%B %d, %Y')}",
            fontsize=14,
            ha="center",
        )

        # Add academic disclaimer
        ax.text(
            0.5,
            0.18,
            "ACADEMIC PROJECT DISCLAIMER",
            fontsize=10,
            ha="center",
            color="darkblue",
            fontweight="bold",
        )

        ax.text(
            0.5,
            0.14,
            "This report contains simulated data and is created for educational purposes only.",
            fontsize=8,
            ha="center",
            style="italic",
        )

        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # Executive Summary page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(
            0.5, 0.95, "Executive Summary", fontsize=20, ha="center", fontweight="bold"
        )

        # Create simulated metrics data for the report
        metrics_patterns = {
            "JPMorgan": {
                "NPL_Ratio": [0.9, 1.0, 1.2, 1.3],  # Worsening trend
                "Coverage_Ratio": [170, 165, 160, 155],  # Declining trend
                "Provisions_Bn": [4.2, 4.4, 4.6, 4.8],  # Increasing provisions
                "CET1_Ratio": [13.5, 13.4, 13.2, 13.0],  # Declining trend
            },
            "UBS": {
                "NPL_Ratio": [1.1, 1.2, 1.4, 1.6],  # Worse than JPM
                "Coverage_Ratio": [150, 145, 140, 135],  # Worse than JPM
                "Provisions_Bn": [3.5, 3.8, 4.1, 4.5],  # Increasing more rapidly
                "CET1_Ratio": [13.0, 12.8, 12.5, 12.2],  # Declining faster than JPM
            },
        }

        # Get quarter index (0-3) for metrics data
        quarter_index = {"2024 Q1": 0, "2024 Q2": 1, "2024 Q3": 2, "2024 Q4": 3}.get(
            selected_quarter, 0
        )

        # Check if we have data for this bank
        if selected_bank in metrics_patterns:
            bank_data = metrics_patterns[selected_bank]

            # Get current values
            npl_ratio = bank_data["NPL_Ratio"][quarter_index]
            coverage_ratio = bank_data["Coverage_Ratio"][quarter_index]
            provisions_bn = bank_data["Provisions_Bn"][quarter_index]
            cet1_ratio = bank_data["CET1_Ratio"][quarter_index]

            # Get first quarter values for trend
            first_npl = bank_data["NPL_Ratio"][0]
            first_coverage = bank_data["Coverage_Ratio"][0]

            # Calculate changes
            npl_change = ((npl_ratio - first_npl) / first_npl) * 100
            coverage_change = ((coverage_ratio - first_coverage) / first_coverage) * 100

            # Calculate risk score
            # Simple formula for risk score (0-10 scale)
            risk_score = (
                (npl_ratio * 4) + (200 - coverage_ratio) / 10 + (14 - cet1_ratio) * 2
            )
            risk_score = max(min(risk_score / 10, 10), 0)

            # Determine risk category
            if risk_score > 6:
                risk_category = "High"
                risk_color = "red"
            elif risk_score > 3.5:
                risk_category = "Medium"
                risk_color = "orange"
            else:
                risk_category = "Low"
                risk_color = "green"

            # Add key metrics summary
            ax.text(
                0.1,
                0.85,
                f"{selected_bank} - Key Credit Risk Metrics ({selected_quarter})",
                fontsize=14,
                ha="left",
                fontweight="bold",
            )

            metrics_text = f"""
            NPL Ratio: {npl_ratio:.2f}% ({npl_change:+.1f}% YTD)
            Coverage Ratio: {coverage_ratio:.0f}% ({coverage_change:+.1f}% YTD)
            Provisions: ${provisions_bn:.1f}Bn
            CET1 Ratio: {cet1_ratio:.2f}%
            
            Overall Risk Assessment: {risk_category} ({risk_score:.1f}/10)
            """

            # Add multi-line metrics text with proper formatting
            ax.text(
                0.1,
                0.80,
                metrics_text,
                fontsize=12,
                ha="left",
                va="top",
                linespacing=1.8,
            )

            # Add bank-specific insights
            insights = {
                "JPMorgan": {
                    "2024 Q1": "JPMorgan's credit risk profile appears stable with NPL ratios at 0.9% and strong coverage.",
                    "2024 Q2": "Early signs of credit deterioration with NPL ratio increasing to 1.0% and declining coverage.",
                    "2024 Q3": "Credit metrics continue to weaken with NPL ratio at 1.2% and coverage ratio dropping to 160%.",
                    "2024 Q4": "Clear credit deterioration with NPL ratio at 1.3% and provisions up 14.3% year-over-year.",
                },
                "UBS": {
                    "2024 Q1": "UBS shows elevated credit risk with NPL ratio at 1.1% and lower coverage ratio of 150%.",
                    "2024 Q2": "Continued credit quality deterioration with NPL ratio at 1.2% and coverage ratio at 145%.",
                    "2024 Q3": "Accelerating deterioration with NPL ratio at 1.4% and coverage ratio dropping to 140%.",
                    "2024 Q4": "Significant credit risk with NPL ratio at 1.6% and provisions up 28.6% year-over-year.",
                },
            }

            if (
                selected_bank in insights
                and selected_quarter in insights[selected_bank]
            ):
                insight_text = insights[selected_bank][selected_quarter]

                ax.text(
                    0.1,
                    0.60,
                    "Key Insights:",
                    fontsize=14,
                    ha="left",
                    fontweight="bold",
                )
                ax.text(
                    0.1,
                    0.55,
                    insight_text,
                    fontsize=12,
                    ha="left",
                    va="top",
                    linespacing=1.5,
                )

            # Create a simple bar chart for NPL Ratio
            ax_npl = fig.add_axes([0.1, 0.30, 0.35, 0.15])
            quarters = ["Q1", "Q2", "Q3", "Q4"]
            npl_values = bank_data["NPL_Ratio"]

            ax_npl.bar(quarters, npl_values, color="#1a2e44")
            ax_npl.set_title("NPL Ratio by Quarter (%)", fontsize=10)
            ax_npl.tick_params(axis="both", which="major", labelsize=8)
            ax_npl.set_ylim(0, max(npl_values) * 1.2)

            # Create a simple bar chart for Coverage Ratio
            ax_cov = fig.add_axes([0.55, 0.30, 0.35, 0.15])
            coverage_values = bank_data["Coverage_Ratio"]

            ax_cov.bar(quarters, coverage_values, color="#ff7f0e")
            ax_cov.set_title("Coverage Ratio by Quarter (%)", fontsize=10)
            ax_cov.tick_params(axis="both", which="major", labelsize=8)
            ax_cov.set_ylim(min(coverage_values) * 0.9, max(coverage_values) * 1.1)

            # Add comparison data if available
            if comparison_bank and comparison_bank in metrics_patterns:
                comp_data = metrics_patterns[comparison_bank]

                # Get comparison values
                comp_npl = comp_data["NPL_Ratio"][quarter_index]
                comp_coverage = comp_data["Coverage_Ratio"][quarter_index]
                comp_provisions = comp_data["Provisions_Bn"][quarter_index]
                comp_cet1 = comp_data["CET1_Ratio"][quarter_index]

                # Add comparison title
                ax.text(
                    0.1,
                    0.20,
                    f"Comparison with {comparison_bank}:",
                    fontsize=14,
                    ha="left",
                    fontweight="bold",
                )

                # Add comparison text
                comparison_text = f"""
                NPL Ratio: {selected_bank} {npl_ratio:.2f}% vs {comparison_bank} {comp_npl:.2f}% ({comp_npl - npl_ratio:+.2f}%)
                Coverage Ratio: {selected_bank} {coverage_ratio:.0f}% vs {comparison_bank} {comp_coverage:.0f}% ({comp_coverage - coverage_ratio:+.0f}%)
                Provisions: {selected_bank} ${provisions_bn:.1f}Bn vs {comparison_bank} ${comp_provisions:.1f}Bn
                CET1 Ratio: {selected_bank} {cet1_ratio:.2f}% vs {comparison_bank} {comp_cet1:.2f}%
                """

                ax.text(
                    0.1,
                    0.18,
                    comparison_text,
                    fontsize=10,
                    ha="left",
                    va="top",
                    linespacing=1.5,
                )
        else:
            # If no data available
            ax.text(
                0.1,
                0.7,
                f"No data available for {selected_bank}",
                fontsize=12,
                ha="left",
            )

        # Add footer with page number
        ax.text(
            0.5,
            0.02,
            "G-SIB Credit Risk Analysis - Academic Project",
            fontsize=8,
            ha="center",
            style="italic",
        )
        ax.text(0.95, 0.02, "Page 2", fontsize=8, ha="right")

        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # Topic and Sentiment Analysis page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(
            0.5,
            0.95,
            "Topic & Sentiment Analysis",
            fontsize=20,
            ha="center",
            fontweight="bold",
        )

        # Simulated sentiment data for the report
        sentiment_patterns = {
            "JPMorgan": {
                "Loan Loss Provisions": -0.3,
                "NPL Ratio": -0.2,
                "Credit Quality": 0.15,
                "Lending Standards": 0.05,
            },
            "UBS": {
                "Loan Loss Provisions": -0.4,
                "NPL Ratio": -0.3,
                "Credit Quality": 0.0,
                "Lending Standards": -0.05,
            },
        }

        if selected_bank in sentiment_patterns:
            # Create a horizontal bar chart for topic sentiment
            ax_sent = fig.add_axes([0.1, 0.70, 0.8, 0.2])

            topics = list(sentiment_patterns[selected_bank].keys())
            sentiments = list(sentiment_patterns[selected_bank].values())

            # Set colors based on sentiment
            colors = [
                "#d62728" if s < -0.1 else "#2ca02c" if s > 0.1 else "#1f77b4"
                for s in sentiments
            ]

            bars = ax_sent.barh(topics, sentiments, color=colors)
            ax_sent.set_title(
                f"{selected_bank} - Topic Sentiment ({selected_quarter})", fontsize=12
            )
            ax_sent.axvline(x=0, color="gray", linestyle="-", linewidth=1)
            ax_sent.axvline(x=0.1, color="green", linestyle="--", linewidth=1)
            ax_sent.axvline(x=-0.1, color="red", linestyle="--", linewidth=1)
            ax_sent.set_xlim(-0.5, 0.5)
            ax_sent.set_xlabel("Sentiment Score", fontsize=10)

            # Add text for sentiment interpretation
            ax.text(
                0.1,
                0.65,
                "Sentiment Analysis Interpretation:",
                fontsize=14,
                ha="left",
                fontweight="bold",
            )

            sentiment_text = f"""
            • Most negative sentiment relates to Loan Loss Provisions (-{abs(sentiment_patterns[selected_bank]["Loan Loss Provisions"]):.1f}), 
              suggesting concerns about future credit losses
              
            • NPL Ratio sentiment is also negative (-{abs(sentiment_patterns[selected_bank]["NPL Ratio"]):.1f}), 
              indicating acknowledgment of deteriorating credit quality
              
            • Credit Quality sentiment remains {"positive" if sentiment_patterns[selected_bank]["Credit Quality"] > 0 else "neutral"} 
              ({sentiment_patterns[selected_bank]["Credit Quality"]:.2f}), suggesting confidence in overall portfolio quality
              
            • The {"positive" if sentiment_patterns[selected_bank]["Lending Standards"] > 0 else "negative"} sentiment on 
              Lending Standards ({sentiment_patterns[selected_bank]["Lending Standards"]:.2f}) indicates 
              {"prudent approach to risk management" if sentiment_patterns[selected_bank]["Lending Standards"] > 0 else "potential concerns about underwriting quality"}
            """

            ax.text(
                0.1,
                0.60,
                sentiment_text,
                fontsize=10,
                ha="left",
                va="top",
                linespacing=1.8,
            )

            # Create word cloud examples for key topics
            wordcloud_data = {
                "Loan Loss Provisions": "provisions reserves allowances impairments coverage charge write-offs loss models IFRS9 CECL expected-credit-loss",
                "NPL Ratio": "non-performing loans NPL delinquency default past-due classified watchlist underperforming deteriorating criticized",
            }

            # Create word clouds for the top 2 topics
            wc_topics = ["Loan Loss Provisions", "NPL Ratio"]

            for i, topic in enumerate(wc_topics):
                if topic in wordcloud_data:
                    # Create word cloud
                    wordcloud = WordCloud(
                        width=400,
                        height=200,
                        background_color="white",
                        colormap="Blues",
                        max_words=50,
                    ).generate(wordcloud_data[topic])

                    # Position word clouds side by side
                    ax_wc = fig.add_axes([0.1 + (i * 0.45), 0.30, 0.4, 0.2])
                    ax_wc.imshow(wordcloud, interpolation="bilinear")
                    ax_wc.set_title(f"Key Terms: {topic}", fontsize=10)
                    ax_wc.axis("off")

            # Add comparison if available
            if comparison_bank and comparison_bank in sentiment_patterns:
                ax.text(
                    0.1,
                    0.20,
                    f"Sentiment Comparison with {comparison_bank}:",
                    fontsize=14,
                    ha="left",
                    fontweight="bold",
                )

                comparison_text = ""
                for topic in topics:
                    bank_sentiment = sentiment_patterns[selected_bank][topic]
                    comp_sentiment = sentiment_patterns[comparison_bank][topic]
                    diff = comp_sentiment - bank_sentiment

                    comparison_text += f"• {topic}: {selected_bank} {bank_sentiment:.2f} vs {comparison_bank} {comp_sentiment:.2f} (diff: {diff:+.2f})\n"

                ax.text(
                    0.1,
                    0.18,
                    comparison_text,
                    fontsize=10,
                    ha="left",
                    va="top",
                    linespacing=1.5,
                )
        else:
            # If no data available
            ax.text(
                0.1,
                0.7,
                f"No topic data available for {selected_bank}",
                fontsize=12,
                ha="left",
            )

        # Add footer
        ax.text(
            0.5,
            0.02,
            "G-SIB Credit Risk Analysis - Academic Project",
            fontsize=8,
            ha="center",
            style="italic",
        )
        ax.text(0.95, 0.02, "Page 3", fontsize=8, ha="right")

        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

    # Return the PDF as base64 encoded string
    pdf_data = pdf_buffer.getvalue()
    b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
    return b64_pdf


# Modified main function to support the bank-centric approach
def main():
    # Create the main layout
    create_layout()

    # Load data
    (
        topic_df,
        sentiment_df,
        metrics_df,
        speaker_df,
        wordcloud_data,
        quotes_data,
        llm_insights,
    ) = load_sample_data()

    # Get selections from the sidebar
    selected_bank, selected_quarter, comparison_bank, show_comparison = (
        create_sidebar_filters()
    )

    # Convert quarter from "2024 Q1" format to datetime for filtering
    quarter_mapping = {
        "2024 Q1": datetime(2024, 1, 1),
        "2024 Q2": datetime(2024, 4, 1),
        "2024 Q3": datetime(2024, 7, 1),
        "2024 Q4": datetime(2024, 10, 1),
    }
    selected_date = quarter_mapping.get(selected_quarter)

    # Divider line
    st.markdown(
        "<hr style='margin-top: 0; margin-bottom: 30px;'>", unsafe_allow_html=True
    )

    # Display executive summary as the first section
    display_executive_summary(
        selected_bank,
        selected_quarter,
        metrics_df,
        sentiment_df,
        llm_insights,
        quotes_data,
    )

    # Add divider between sections
    st.markdown(
        "<hr style='margin-top: 30px; margin-bottom: 30px;'>", unsafe_allow_html=True
    )

    # Display topic analysis section
    display_topic_analysis(
        selected_bank,
        selected_quarter,
        topic_df,
        sentiment_df,
        wordcloud_data,
        quotes_data,
    )

    # Add divider between sections
    st.markdown(
        "<hr style='margin-top: 30px; margin-bottom: 30px;'>", unsafe_allow_html=True
    )

    # Display sentiment analysis section
    display_sentiment_analysis(
        selected_bank, selected_quarter, sentiment_df, speaker_df
    )

    # Add divider between sections
    st.markdown(
        "<hr style='margin-top: 30px; margin-bottom: 30px;'>", unsafe_allow_html=True
    )

    # Display advanced analytics section with optional comparison
    display_advanced_analytics(
        selected_bank,
        selected_quarter,
        comparison_bank if show_comparison else None,
        metrics_df,
        sentiment_df,
    )

    # Display footer
    display_footer()


# Function to format datetime for quarter display
def format_quarter(date):
    """Convert datetime to quarter string format (e.g., '2024 Q1')"""
    quarter = (date.month - 1) // 3 + 1
    return f"{date.year} Q{quarter}"


# Utility function to show a simple loading state
def display_loading(message="Loading data..."):
    """Display a loading message while processing"""
    with st.spinner(message):
        # Add a slight delay to ensure UI updates
        time.sleep(0.5)


# Helper function to determine risk color
def get_risk_color(value, thresholds, inverse=False):
    """Return color based on value and thresholds

    Parameters:
    - value: The metric value
    - thresholds: Tuple of (low_threshold, high_threshold)
    - inverse: If True, lower values are better; if False, higher values are better

    Returns:
    - Color string
    """
    low, high = thresholds

    if inverse:
        # Lower values are better (e.g., NPL ratio)
        if value > high:
            return "#d62728"  # Red - high risk
        elif value > low:
            return "#ff7f0e"  # Orange - medium risk
        else:
            return "#2ca02c"  # Green - low risk
    else:
        # Higher values are better (e.g., Coverage ratio)
        if value < low:
            return "#d62728"  # Red - high risk
        elif value < high:
            return "#ff7f0e"  # Orange - medium risk
        else:
            return "#2ca02c"  # Green - low risk


# Helper function to create risk badge HTML
def create_risk_badge(risk_category, label=None):
    """Create HTML for a risk badge

    Parameters:
    - risk_category: 'Low', 'Medium', or 'High'
    - label: Optional text to display (defaults to risk_category)

    Returns:
    - HTML string for the badge
    """
    if label is None:
        label = risk_category

    if risk_category.lower() == "high":
        badge_class = "risk-badge-high"
    elif risk_category.lower() == "medium":
        badge_class = "risk-badge-medium"
    else:
        badge_class = "risk-badge-low"

    return f'<span class="risk-badge {badge_class}">{label}</span>'


# Application entry point
if __name__ == "__main__":
    import time  # Import time for loading effects

    main()
