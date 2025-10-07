# Intelligent Data Cleaning Assistant

## Overview
A Streamlit-based web application designed for statistical agencies to clean and analyze survey data. The application provides AI-powered guidance, comprehensive analysis tools, and detailed reporting capabilities for data quality assessment and cleaning operations.

## Purpose
This application helps data analysts and statisticians:
- Upload and analyze survey datasets
- Detect and handle missing values, outliers, and data quality issues
- Apply various cleaning methods with survey weight support
- Get AI-powered recommendations for cleaning strategies
- Generate comprehensive reports of all cleaning operations

## Tech Stack
- **Framework**: Streamlit (Python web framework)
- **Data Processing**: Pandas, NumPy
- **Analysis**: Scikit-learn, SciPy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **AI Assistant**: Groq API (llama-3.1-8b-instant model)
- **Reporting**: Jinja2 templates, ReportLab (PDF generation)

## Project Structure
```
.
├── app.py                          # Main application entry point
├── pages/                          # Streamlit pages
│   ├── 1_Data_Upload.py           # Dataset upload and configuration
│   ├── 2_Column_Analysis.py       # Individual column analysis
│   ├── 3_Cleaning_Wizard.py       # Data cleaning operations with validation
│   ├── 4_Visualization.py         # Custom visualizations with KDE & Q-Q plots
│   ├── 5_Hypothesis_Analysis.py   # Statistical hypothesis testing
│   ├── 6_AI_Assistant.py          # AI-powered guidance
│   └── 7_Reports.py               # Report generation with exports
├── modules/                        # Core functionality modules
│   ├── ai_assistant.py            # AI assistant integration
│   ├── cleaning_engine.py         # Data cleaning methods + validation
│   ├── data_analyzer.py           # Column analysis engine
│   ├── hypothesis_analysis.py     # Statistical hypothesis testing
│   ├── report_generator.py        # Report generation
│   ├── survey_weights.py          # Survey weights management
│   ├── utils.py                   # Utility functions
│   └── visualization.py           # Visualization components
├── pyproject.toml                 # Python dependencies
└── .gitignore                     # Git ignore rules
```

## Key Features

### Phase 1 (Core Features)
1. **Data Upload & Configuration**: Support for CSV and Excel files with automatic column type detection
2. **Column Analysis**: Individual column analysis with missing data patterns, outlier detection, and quality assessment
3. **Cleaning Wizard**: Multiple cleaning methods (imputation, outlier handling, standardization) with survey weight support
4. **AI Assistant**: Context-aware guidance using Groq API for cleaning recommendations
5. **Undo/Redo**: Full operation history with undo/redo functionality
6. **Survey Weights**: Integrated support for survey design weights in all analyses

### Phase 2 (Enhanced Features) ✅ COMPLETED
7. **Custom Visualizations**: Interactive visualization builder with:
   - Multi-column selection support (1-4 columns)
   - 9 chart types: bar, line, scatter, box, violin, histogram, pie, heatmap, correlation
   - Real-time updates reflecting cleaned data
   - Static PNG image generation for PDF reports
   - Save/download functionality
   - Chart customization (title, height, legend)
   
8. **Enhanced Distribution Analysis**: User-friendly statistical explanations with:
   - Visual interpretations with icons for skewness, kurtosis, normality
   - Quartile analysis with 5-point summary
   - Category entropy analysis for categorical variables
   - Collapsible help sections explaining concepts
   
9. **Comprehensive Anomaly Detection**: Multi-method anomaly detection with:
   - IQR, Z-score, Modified Z-score, Isolation Forest methods
   - Visual representation and severity assessment
   - Results stored for PDF report inclusion
   - Interactive visualizations
   
10. **Professional PDF Reports**: Export comprehensive reports with:
    - Executive summary with dataset statistics
    - Anomaly detection results with tables
    - Column analysis summaries
    - Embedded visualizations as high-resolution images
    - Cleaning operations audit trail
    - Professional formatting using ReportLab
    
11. **Multiple Export Formats**: 
    - PDF (with anomalies + visualizations)
    - Markdown
    - HTML
    - JSON

## Environment Setup
The application is configured to run on Replit with:
- Python 3.11
- All dependencies managed via uv (see pyproject.toml)
- Streamlit server on port 5000 with proper CORS configuration

## Configuration
The Streamlit app is configured to:
- Listen on 0.0.0.0:5000 (required for Replit proxy)
- Disable CORS and XSRF protection for Replit environment
- Run in headless mode for server deployment

## AI Assistant Configuration
The AI assistant requires a GROQ_API_KEY environment variable to be set. Without this key, the AI features will not be available, but all other functionality will work normally.

## Recent Changes
- **2025-10-07 Hypothesis Testing & Enhanced Visualizations**: Added comprehensive statistical testing capabilities
  - ✅ **New Visualization Types**: Added KDE (Kernel Density Estimate) and Q-Q (Quantile-Quantile) plots to the Visualization page
    * All visualizations including KDE and Q-Q plots properly save to PDF reports
  - ✅ **Hypothesis Testing Module**: Comprehensive statistical hypothesis testing with:
    * 15 statistical tests: One-sample t-test, Welch's t-test, Mann-Whitney U, Pearson/Spearman correlations
    * Chi-square, Fisher's exact test, One-way ANOVA, Kruskal-Wallis
    * Tukey's HSD post-hoc, Two-proportion z-test, Paired t-test, Independent t-test
    * Simple linear regression, Logistic regression (with proper target/predictor selection)
  - ✅ **Intelligent Test Recommendations**: Automatic test selection based on:
    * Data types (numeric/categorical)
    * Number of groups
    * Distribution characteristics (normality testing)
    * Sample sizes
  - ✅ **Comprehensive Test Output**: Each test returns:
    * Test statistic, p-value, degrees of freedom
    * Effect size (Cohen's d, Cramér's V, R-squared, etc.)
    * Confidence intervals
    * Assumption checks with warnings
    * Plain-English interpretations
  - ✅ **Hypothesis Analysis Page**: New dedicated page (5_Hypothesis_Analysis.py) with:
    * Intuitive variable selection interface
    * Test category filtering (Basic Tests, ANOVA, Categorical, Regression, Advanced)
    * Real-time recommendations
    * Results storage and visualization suggestions
    * Export to PDF reports
    * Proper parameter handling for all test types
  - ✅ **Page Structure Update**: Reorganized navigation to include Hypothesis Analysis between Visualization and AI Assistant
    * Pages renumbered: AI Assistant (5→6), Reports (6→7)
  

- **2025-10-03 Performance Optimizations**: Comprehensive performance improvements across all features
  - ✅ **Deterministic Caching System**: Implemented SHA256-based, order-aware caching in ColumnAnalyzer
    * Full-column hash that detects any data value or order changes
    * Deterministic across sessions for stable cache reuse
    * Cache invalidation on data modifications ensures accuracy
  
  - ✅ **Optimized Data Analysis**:
    * Vectorized missing pattern detection using NumPy (replaces loop-based approach)
    * Optimized IQR outlier detection with vectorized quantile calculations
    * Instance-level correlation matrix caching in DataVisualizer
  
  - ✅ **Enhanced Cleaning Engine**:
    * KNN imputation now uses top 10 most correlated columns when >10 features available
    * Isolation Forest with smart sampling (>50K rows) and parallel processing (n_jobs=-1)
    * Optimized neighbor selection based on available data
  
  - ✅ **Memory Optimizations**:
    * Missing value heatmaps use int8 instead of int for 87.5% memory reduction
    * Large dataset sampling (>10K rows) for visualization performance
    * Capped visualization heights to prevent excessive memory usage
  
  - ✅ **Rendering Improvements**:
    * Correlation matrix calculations cached with instance-level storage
    * Progress indicators via st.spinner for long-running operations
    * Efficient batch processing for large datasets

- **2025-10-03 Phase 2 Enhancements**: Advanced visualization, anomaly detection, and PDF reporting
  - ✅ **Enhanced Distribution Analysis**: Improved Column Analysis distribution graphs with:
    * Detailed explanations for skewness, kurtosis, and normality tests with visual icons
    * Quartile analysis with 5-point summary (Min, Q1, Median, Q3, Max)
    * Category distribution insights for categorical variables with entropy analysis
    * Collapsible help sections explaining statistical concepts in plain language
    * User-friendly interpretations for all statistical measures
  
  - ✅ **Comprehensive Anomaly Detection**: Added dedicated anomaly detection section in Column Analysis with:
    * Multi-method detection: IQR, Z-score, Modified Z-score, Isolation Forest
    * Visual representation of detected anomalies
    * Severity assessment (Low, Medium, High, Critical)
    * Results stored in session state for PDF report inclusion
    * Interactive visualization with Plotly charts
  
  - ✅ **Redesigned Visualization Page**: Complete rewrite with interactive visualization builder:
    * Multi-column selection (1-4 columns depending on chart type)
    * 9 chart types: bar, line, scatter, box, violin, histogram, pie, heatmap, correlation matrix
    * Real-time data updates reflecting cleaned data
    * Static PNG image generation (1200px width, high resolution)
    * Save visualizations to PDF reports with metadata
    * Data quality indicators showing cleaning impact
    * Download individual charts or save to report
    * Chart configuration (title, height, legend)
  
  - ✅ **PDF Report Generation**: Professional PDF export using reportlab:
    * Comprehensive executive summary with dataset statistics
    * Anomaly detection results with detailed tables
    * Column analysis summaries with quality scores
    * Embedded static visualizations from saved charts
    * Cleaning operations audit trail with timestamps
    * Professional formatting with styled tables and headers
    * Export alongside existing Markdown, HTML, and JSON formats
  
  - ✅ **Data Quality Dashboard**: Added overview metrics in Visualization page showing:
    * Missing data percentage across dataset
    * Analyzed vs total columns
    * Cleaned columns count
    * Average quality score

- **2025-10-03 Initial Setup**: Application refinement and UX improvements
  - ✅ **Groq API Integration**: Configured AI Assistant with secure API key management via Replit Secrets
  - ✅ **Fixed Plotly Warnings**: Replaced all deprecated `width='stretch'` with `use_container_width=True` in st.plotly_chart() calls
  - ✅ **Professional Report Templates**: Improved executive summary with formal structure, tables, and professional language suitable for statistical agencies
  - ✅ **Improved Navigation**: Clean 6-page structure (Data Upload, Column Analysis, Cleaning Wizard, Visualization, AI Assistant, Reports)

- **2025-10-03**: GitHub Import to Replit Environment ✅ COMPLETED
  - Successfully imported project from GitHub
  - Installed Python 3.11 and all dependencies using uv package manager
  - All 61 Python packages installed successfully (streamlit, pandas, numpy, scikit-learn, plotly, seaborn, matplotlib, groq, jinja2, scipy)
  - Verified Streamlit configuration (.streamlit/config.toml) is properly set up for Replit
  - Configured workflow to run on port 5000 with 0.0.0.0 binding
  - Set up autoscale deployment configuration for production
  - Application running successfully and tested via screenshot
  - ⚠️ Note: GROQ_API_KEY environment variable is not set - AI Assistant features will not be available until configured
  - To enable AI features: Add GROQ_API_KEY in the Secrets tab (Tools → Secrets) in the Replit workspace

## Deployment
The application is configured for autoscale deployment, suitable for stateless web applications. It automatically scales based on traffic and stops when not in use.

## User Preferences
None configured yet.

## Notes
- The application uses a virtual environment (.pythonlibs/) for dependency isolation
- Data files (CSV, Excel) are gitignored to protect sensitive survey data
- The app supports weighted and unweighted statistical analysis
- All cleaning operations are logged with timestamps for audit trails
