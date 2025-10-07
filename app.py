import streamlit as st
import pandas as pd
import numpy as np

# Configure page
st.set_page_config(
    page_title=" Data Cleaning Assistant",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    from modules.utils import initialize_session_state
    initialize_session_state()
    st.title("🧹 Intelligent Data Cleaning Assistant")
    
    st.markdown("""
    Welcome to the Survey Data Cleaning Assistant - an AI-powered tool designed specifically for statistical agencies.
    This application analyzes each column individually and provides context-specific cleaning recommendations.
    
    ### Key Features:
    - **Individual Column Analysis**: Each column is analyzed separately with tailored recommendations
    - **AI-Powered Assistance**: Context-aware guidance using advanced language models
    - **Multiple Cleaning Strategies**: Various methods for handling missing values, outliers, and inconsistencies
    - **Comprehensive Audit Trail**: Track all cleaning operations with undo/redo functionality
    - **Statistical Rigor**: Maintain methodological consistency for survey data
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Use the pages in the sidebar to navigate through the application:")
    
    st.sidebar.markdown("""
    1. **Data Upload** - Upload and configure your dataset
    2. **Column Analysis** - Detailed analysis of individual columns  
    3. **Cleaning Wizard** - Apply cleaning methods with integrated weights
    4. **AI Assistant** - Get expert advice and explanations
    5. **Reports** - Generate comprehensive cleaning reports
    """)
    
    # Display current dataset info if available
    if 'dataset' in st.session_state and st.session_state.dataset is not None:
        st.subheader("📊 Current Dataset Overview")
        df = st.session_state.dataset
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            cleaned_cols = len(st.session_state.get('cleaning_history', {}))
            st.metric("Cleaned Columns", cleaned_cols)
        
        st.dataframe(df.head(), width='stretch')
    else:
        st.info("👆 Please upload a dataset using the **Data Upload** page to get started.")
        
        # Quick start section
        st.subheader("🚀 Quick Start Guide")
        st.markdown("""
        1. **Upload Your Data**: Go to the Data Upload page and select your CSV or Excel file
        2. **Review Column Types**: The system will automatically detect column types - review and adjust as needed
        3. **Analyze Columns**: Use the Column Analysis page to examine each column individually
        4. **Clean Your Data**: Apply appropriate cleaning methods using the Cleaning Wizard
        5. **Get AI Help**: Use the AI Assistant for expert guidance and explanations
        6. **Generate Reports**: Create comprehensive documentation of your cleaning process
        """)

if __name__ == "__main__":
    main()
