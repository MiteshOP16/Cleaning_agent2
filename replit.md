# Intelligent Data Cleaning Assistant

## Overview
A Streamlit-based web application for statistical agencies, designed to clean and analyze survey data. It provides AI-powered guidance, comprehensive analysis tools, and detailed reporting capabilities for data quality assessment and cleaning operations. The application aims to help data analysts and statisticians upload, analyze, detect, handle, and clean survey datasets, apply various cleaning methods with survey weight support, get AI-powered recommendations, and generate comprehensive reports.

## User Preferences
None configured yet.

## System Architecture
The application is built using **Streamlit** (Python web framework) and leverages **Pandas** and **NumPy** for data processing. For analysis, it uses **Scikit-learn** and **SciPy**, while **Plotly**, **Seaborn**, and **Matplotlib** handle visualization. AI assistance is integrated via the **Groq API** (llama-3.1-8b-instant model). Reporting is managed with **Jinja2** templates and **ReportLab** for PDF generation.

The core structure includes:
- **Streamlit Pages**: Dedicated pages for Data Upload, Anomaly Detection, Column Analysis, Cleaning Wizard, Visualization, Hypothesis Analysis, AI Assistant, and Reports.
- **Modular Design**: Functionality is organized into `modules/` for AI integration, anomaly detection, cleaning engine, data analysis, hypothesis testing, report generation, survey weights, and utility functions.
- **UI/UX Decisions**: Features include interactive visualization builders with multi-column selection and 9 chart types, enhanced distribution analysis with statistical explanations and visual interpretations, and multi-method statistical outlier detection.
- **Reporting**: Professional PDF reports with modern styling, color palettes, improved table styling, enhanced typography, and color-coded messages are generated using ReportLab, alongside Markdown, HTML, and JSON export options. Reports include executive summaries, anomaly detection results, column analysis summaries, embedded high-resolution visualizations, and a cleaning operations audit trail.
- **Key Features**:
    - **Data Upload & Configuration**: Supports CSV and Excel with automatic column type detection.
    - **Column Analysis**: Individual column analysis including missing data patterns, outlier detection, and quality assessment.
    - **Cleaning Wizard**: Multiple cleaning methods (imputation, outlier handling, standardization) with survey weight support.
    - **AI Assistant**: Context-aware guidance for cleaning recommendations.
    - **Undo/Redo**: Full operation history.
    - **Survey Weights**: Integrated support for survey design weights.
    - **Data Type Anomaly Detection**: Dedicated page for type mismatch detection, clear display of anomalous values, and flexible correction options.
    - **Hypothesis Testing**: Comprehensive statistical testing with 15 test types, intelligent recommendations based on data characteristics, and detailed output with interpretations.
    - **Performance Optimizations**: Implemented deterministic caching, vectorized operations, optimized imputation and outlier detection, and memory optimizations for large datasets.

## External Dependencies
- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Analysis**: Scikit-learn, SciPy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **AI Assistant**: Groq API (llama-3.1-8b-instant model)
- **Reporting**: Jinja2, ReportLab