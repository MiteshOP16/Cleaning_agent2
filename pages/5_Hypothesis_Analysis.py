import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import initialize_session_state
from modules.hypothesis_analysis import HypothesisAnalyzer
import plotly.graph_objects as go
import plotly.express as px

initialize_session_state()

st.title("🔬 Hypothesis Testing & Statistical Analysis")

st.markdown("""
Perform comprehensive hypothesis tests to analyze relationships and differences in your data. 
The system automatically recommends the most suitable tests based on your data types and distribution characteristics.
""")

if st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset in the Data Upload page first.")
    st.stop()

df = st.session_state.dataset
analyzer = HypothesisAnalyzer()

# Initialize session state for hypothesis results
if 'hypothesis_results' not in st.session_state:
    st.session_state.hypothesis_results = []

# ===== TEST SELECTION =====
st.subheader("📊 1. Select Variables for Analysis")

col_select1, col_select2 = st.columns(2)

with col_select1:
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    all_cols = df.columns.tolist()
    
    selected_columns = st.multiselect(
        "Select columns (1-3):",
        options=all_cols,
        max_selections=3,
        help="Choose 1-3 columns for hypothesis testing"
    )

with col_select2:
    if selected_columns:
        alpha = st.slider(
            "Significance level (α):",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            help="Typically 0.05 for 95% confidence"
        )
        analyzer.set_alpha(alpha)

# ===== TEST RECOMMENDATION =====
if selected_columns:
    st.divider()
    st.subheader("💡 2. Recommended Tests")
    
    # Get column types
    data_types = {}
    for col in selected_columns:
        if col in numeric_cols:
            data_types[col] = 'numeric'
        else:
            data_types[col] = 'categorical'
    
    # Get recommendations
    recommendations = analyzer.recommend_test(df, selected_columns, data_types)
    
    if 'error' in recommendations:
        st.error(recommendations['error'])
    else:
        # Display recommendations
        st.markdown(f"**Column types:** {', '.join([f'{col} ({dtype})' for col, dtype in recommendations['column_types'].items()])}")
        
        if recommendations['recommendations']:
            st.markdown("**Recommended tests based on your data:**")
            
            for rec in recommendations['recommendations']:
                priority_color = '🔴' if rec['priority'] == 'high' else '🟡' if rec['priority'] == 'medium' else '🟢'
                st.info(f"{priority_color} **{rec['test'].replace('_', ' ').title()}**: {rec['reason']}")
            
            # Set default test
            default_test = recommendations['recommendations'][0]['test'] if recommendations['recommendations'] else None
        else:
            st.warning("No automatic recommendations available for this combination. Please select a test manually.")
            default_test = None
        
        st.divider()
        st.subheader("🧪 3. Select and Run Test")
        
        # Test selection
        test_options = {
            'one_sample_ttest': 'One-sample t-test (1 group vs value)',
            'welch_ttest': "Welch's t-test (2 groups)",
            'independent_ttest': "Independent t-test (2 groups, equal variance)",
            'mann_whitney': 'Mann-Whitney U test (2 groups, non-parametric)',
            'paired_ttest': 'Paired t-test (matched samples)',
            'one_way_anova': 'One-way ANOVA (3+ groups)',
            'kruskal_wallis': 'Kruskal-Wallis test (3+ groups, non-parametric)',
            'tukey_hsd': "Tukey's HSD (post-hoc for ANOVA)",
            'pearson_correlation': 'Pearson Correlation',
            'spearman_correlation': 'Spearman Correlation',
            'chi_square': 'Chi-square test of independence',
            'fisher_exact': "Fisher's exact test (2x2)",
            'two_proportion_ztest': 'Two-proportion z-test',
            'simple_linear_regression': 'Simple Linear Regression',
            'logistic_regression': 'Logistic Regression'
        }
        
        # Group tests
        basic_tests = ['one_sample_ttest', 'welch_ttest', 'independent_ttest', 'mann_whitney', 'paired_ttest', 
                       'pearson_correlation', 'spearman_correlation']
        anova_tests = ['one_way_anova', 'kruskal_wallis', 'tukey_hsd']
        categorical_tests = ['chi_square', 'fisher_exact', 'two_proportion_ztest']
        regression_tests = ['simple_linear_regression', 'logistic_regression']
        
        test_category = st.selectbox(
            "Test category:",
            options=['Basic Tests', 'ANOVA & Post-hoc', 'Categorical Data', 'Regression', 'Advanced Tests'],
            help="Filter tests by category"
        )
        
        # Filter test options based on category
        if test_category == 'Basic Tests':
            available_tests = {k: v for k, v in test_options.items() if k in basic_tests}
        elif test_category == 'ANOVA & Post-hoc':
            available_tests = {k: v for k, v in test_options.items() if k in anova_tests}
        elif test_category == 'Categorical Data':
            available_tests = {k: v for k, v in test_options.items() if k in categorical_tests}
        elif test_category == 'Regression':
            available_tests = {k: v for k, v in test_options.items() if k in regression_tests}
        else:
            # Advanced tests toggle
            available_tests = test_options
        
        selected_test = st.selectbox(
            "Choose hypothesis test:",
            options=list(available_tests.keys()),
            format_func=lambda x: available_tests[x],
            index=list(available_tests.keys()).index(default_test) if default_test and default_test in available_tests else 0
        )
        
        # Test-specific parameters
        test_params = {}
        
        if selected_test == 'one_sample_ttest':
            if len([c for c in selected_columns if c in numeric_cols]) >= 1:
                param_col1, param_col2 = st.columns(2)
                with param_col1:
                    column = st.selectbox("Numeric variable:", options=[c for c in selected_columns if c in numeric_cols])
                with param_col2:
                    test_value = st.number_input("Test value (H0: μ =):", value=0.0, format="%.4f")
                test_params = {'column': column, 'test_value': test_value}
            else:
                st.error("One-sample t-test requires a numeric column.")
        
        elif selected_test in ['welch_ttest', 'independent_ttest', 'mann_whitney', 'one_way_anova', 'kruskal_wallis', 'tukey_hsd']:
            if len(selected_columns) >= 2:
                param_col1, param_col2 = st.columns(2)
                with param_col1:
                    numeric_col = st.selectbox("Numeric variable:", options=[c for c in selected_columns if c in numeric_cols])
                with param_col2:
                    group_col = st.selectbox("Grouping variable:", options=[c for c in selected_columns if c in categorical_cols])
                test_params = {'numeric_col': numeric_col, 'group_col': group_col}
            else:
                st.error("This test requires at least one numeric and one categorical column.")
        
        elif selected_test == 'paired_ttest':
            if len([c for c in selected_columns if c in numeric_cols]) >= 2:
                param_col1, param_col2 = st.columns(2)
                with param_col1:
                    col1 = st.selectbox("First measurement:", options=[c for c in selected_columns if c in numeric_cols])
                with param_col2:
                    col2 = st.selectbox("Second measurement:", options=[c for c in selected_columns if c in numeric_cols and c != col1])
                test_params = {'col1': col1, 'col2': col2}
            else:
                st.error("Paired t-test requires two numeric columns.")
        
        elif selected_test in ['pearson_correlation', 'spearman_correlation', 'simple_linear_regression', 'logistic_regression']:
            if len([c for c in selected_columns if c in numeric_cols]) >= 2 or (selected_test == 'logistic_regression' and len(selected_columns) >= 2):
                param_col1, param_col2 = st.columns(2)
                with param_col1:
                    if selected_test == 'logistic_regression':
                        x_col = st.selectbox("Predictor variable:", options=[c for c in selected_columns if c in numeric_cols])
                    else:
                        x_col = st.selectbox("First variable (X):", options=[c for c in selected_columns if c in numeric_cols])
                with param_col2:
                    if selected_test == 'logistic_regression':
                        y_col = st.selectbox("Outcome variable (binary):", options=[c for c in selected_columns if c not in [x_col]])
                    else:
                        y_col = st.selectbox("Second variable (Y):", options=[c for c in selected_columns if c in numeric_cols and c != x_col])
                test_params = {'col1': x_col, 'col2': y_col} if selected_test in ['pearson_correlation', 'spearman_correlation', 'paired_ttest'] else {'x_col': x_col, 'y_col': y_col}
            else:
                st.error(f"This test requires appropriate column types.")
        
        elif selected_test in ['chi_square', 'fisher_exact']:
            if len([c for c in selected_columns if c in categorical_cols]) >= 2:
                param_col1, param_col2 = st.columns(2)
                with param_col1:
                    col1 = st.selectbox("First categorical variable:", options=[c for c in selected_columns if c in categorical_cols])
                with param_col2:
                    col2 = st.selectbox("Second categorical variable:", options=[c for c in selected_columns if c in categorical_cols and c != col1])
                test_params = {'col1': col1, 'col2': col2}
            else:
                st.error("This test requires two categorical columns.")
        
        elif selected_test == 'two_proportion_ztest':
            st.markdown("**Enter counts for two-proportion test:**")
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                successes1 = st.number_input("Successes in group 1:", min_value=0, value=0, step=1)
                total1 = st.number_input("Total in group 1:", min_value=1, value=100, step=1)
            with param_col2:
                successes2 = st.number_input("Successes in group 2:", min_value=0, value=0, step=1)
                total2 = st.number_input("Total in group 2:", min_value=1, value=100, step=1)
            test_params = {'successes': [successes1, successes2], 'totals': [total1, total2]}
        
        # Run test button
        if st.button("🔬 Run Hypothesis Test", type="primary", use_container_width=True):
            if test_params or selected_test == 'two_proportion_ztest':
                with st.spinner(f"Running {test_options[selected_test]}..."):
                    try:
                        # Execute test
                        if selected_test == 'one_sample_ttest':
                            result = analyzer.one_sample_ttest(df, test_params['column'], test_params['test_value'])
                        elif selected_test == 'welch_ttest':
                            result = analyzer.welch_ttest(df, test_params['numeric_col'], test_params['group_col'])
                        elif selected_test == 'mann_whitney':
                            result = analyzer.mann_whitney(df, test_params['numeric_col'], test_params['group_col'])
                        elif selected_test == 'pearson_correlation':
                            result = analyzer.pearson_correlation(df, test_params['col1'], test_params['col2'])
                        elif selected_test == 'spearman_correlation':
                            result = analyzer.spearman_correlation(df, test_params['col1'], test_params['col2'])
                        elif selected_test == 'chi_square':
                            result = analyzer.chi_square(df, test_params['col1'], test_params['col2'])
                        elif selected_test == 'fisher_exact':
                            result = analyzer.fisher_exact(df, test_params['col1'], test_params['col2'])
                        elif selected_test == 'one_way_anova':
                            result = analyzer.one_way_anova(df, test_params['numeric_col'], test_params['group_col'])
                        elif selected_test == 'kruskal_wallis':
                            result = analyzer.kruskal_wallis(df, test_params['numeric_col'], test_params['group_col'])
                        elif selected_test == 'tukey_hsd':
                            result = analyzer.tukey_hsd(df, test_params['numeric_col'], test_params['group_col'])
                        elif selected_test == 'two_proportion_ztest':
                            result = analyzer.two_proportion_ztest(test_params['successes'], test_params['totals'])
                        elif selected_test == 'paired_ttest':
                            result = analyzer.paired_ttest(df, test_params['col1'], test_params['col2'])
                        elif selected_test == 'simple_linear_regression':
                            result = analyzer.simple_linear_regression(df, test_params['x_col'], test_params['y_col'])
                        elif selected_test == 'logistic_regression':
                            result = analyzer.logistic_regression(df, test_params['x_col'], test_params['y_col'])
                        else:
                            result = {'error': 'Test not implemented'}
                        
                        if 'error' in result:
                            st.error(f"❌ Error: {result['error']}")
                        else:
                            # Save result
                            result['test_type'] = selected_test
                            result['columns_used'] = test_params if selected_test != 'two_proportion_ztest' else {'manual_input': True}
                            st.session_state.hypothesis_results.append(result)
                            st.success("✅ Test completed successfully!")
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error running test: {str(e)}")
                        st.exception(e)
            else:
                st.error("Please configure test parameters.")

# ===== DISPLAY RESULTS =====
if st.session_state.hypothesis_results:
    st.divider()
    st.subheader("📊 4. Test Results")
    
    # Show most recent result first
    for idx, result in enumerate(reversed(st.session_state.hypothesis_results)):
        result_idx = len(st.session_state.hypothesis_results) - idx - 1
        
        with st.expander(f"🔬 {result['test_name']}", expanded=(idx == 0)):
            # Main results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Test Statistic", f"{result['statistic']:.4f}" if isinstance(result['statistic'], (int, float)) else str(result['statistic']))
            
            with col2:
                p_val = result['p_value']
                st.metric("p-value", f"{p_val:.4f}" if isinstance(p_val, (int, float)) else str(p_val))
            
            with col3:
                decision_color = "🟢" if "Reject" in result['decision'] or "Significant" in result['decision'] else "🔴"
                st.metric("Decision", f"{decision_color} {result['decision']}")
            
            # Effect size and CI
            if result.get('effect_size'):
                st.markdown(f"**Effect Size:** {result['effect_size']['type']} = {result['effect_size']['value']:.4f}" if isinstance(result['effect_size']['value'], (int, float)) else f"**Effect Size:** {result['effect_size']['type']} = {result['effect_size']['value']}")
            
            if result.get('confidence_interval') and result['confidence_interval'].get('interval') != 'N/A':
                ci = result['confidence_interval']['interval']
                if isinstance(ci, (list, tuple)) and len(ci) == 2:
                    st.markdown(f"**{result['confidence_interval']['level']} Confidence Interval:** ({ci[0]:.4f}, {ci[1]:.4f})")
            
            # Degrees of freedom
            if result.get('df'):
                if isinstance(result['df'], dict):
                    st.markdown(f"**Degrees of Freedom:** {', '.join([f'{k}: {v}' for k, v in result['df'].items()])}")
                else:
                    st.markdown(f"**Degrees of Freedom:** {result['df']}")
            
            # Interpretation
            if result.get('interpretation'):
                st.info(f"**Interpretation:** {result['interpretation']}")
            
            # Sample sizes
            if result.get('sample_sizes'):
                st.markdown(f"**Sample Sizes:** {', '.join([f'{k}: {v}' for k, v in result['sample_sizes'].items()])}")
            
            # Additional stats (for specific tests)
            if result.get('group_stats'):
                st.markdown("**Group Statistics:**")
                stats_df = pd.DataFrame(result['group_stats']).T
                st.dataframe(stats_df, use_container_width=True)
            
            if result.get('coefficients'):
                st.markdown("**Coefficients:**")
                coef_data = result['coefficients']
                st.write(coef_data)
                
                if result.get('equation'):
                    st.markdown(f"**Equation:** `{result['equation']}`")
            
            if result.get('comparisons'):
                st.markdown("**Pairwise Comparisons:**")
                comp_df = pd.DataFrame(result['comparisons'])
                st.dataframe(comp_df, use_container_width=True)
            
            # Assumption checks
            if result.get('assumption_checks'):
                with st.expander("🔍 Assumption Checks"):
                    for assumption, check in result['assumption_checks'].items():
                        if isinstance(check, dict):
                            st.write(f"**{assumption.replace('_', ' ').title()}:** {check}")
                        elif isinstance(check, bool):
                            status = "✅ Passed" if check else "❌ Failed"
                            st.write(f"**{assumption.replace('_', ' ').title()}:** {status}")
                        else:
                            st.write(f"**{assumption.replace('_', ' ').title()}:** {check}")
            
            # Warnings
            if result.get('warnings'):
                st.warning("⚠️ **Warnings:**\n" + "\n".join([f"- {w}" for w in result['warnings']]))
            
            # Notes
            if result.get('notes'):
                st.caption(f"ℹ️ {result['notes']}")
            
            # Action buttons
            action_cols = st.columns([2, 1, 1])
            
            with action_cols[1]:
                if st.button("📊 Visualize", key=f"viz_{result_idx}", use_container_width=True):
                    st.info("Visualization will be shown below")
            
            with action_cols[2]:
                if st.button("🗑️ Remove", key=f"remove_{result_idx}", use_container_width=True):
                    st.session_state.hypothesis_results.pop(result_idx)
                    st.rerun()
    
    # Bulk actions
    st.divider()
    action_cols = st.columns([2, 1, 1])
    
    with action_cols[0]:
        st.markdown(f"**Total tests performed:** {len(st.session_state.hypothesis_results)}")
    
    with action_cols[1]:
        if st.button("🗑️ Clear All Results", use_container_width=True):
            st.session_state.hypothesis_results = []
            st.success("All results cleared")
            st.rerun()
    
    with action_cols[2]:
        if st.button("📊 Generate Report", type="primary", use_container_width=True):
            # Save to session for report generation
            st.session_state.hypothesis_test_results = st.session_state.hypothesis_results
            st.switch_page("pages/7_Reports.py")

else:
    st.info("No hypothesis tests have been run yet. Configure and run a test above to see results.")

# ===== QUICK GUIDE =====
st.divider()

with st.expander("📚 Quick Guide to Hypothesis Testing"):
    st.markdown("""
    ### Choosing the Right Test
    
    **Comparing Two Groups:**
    - **Welch's t-test**: Numeric outcome, 2 groups (default - robust to unequal variances)
    - **Mann-Whitney U**: Numeric outcome, 2 groups, non-normal distribution
    - **Paired t-test**: Numeric outcome, matched/paired samples
    
    **Comparing 3+ Groups:**
    - **One-way ANOVA**: Numeric outcome, 3+ groups, normal distribution
    - **Kruskal-Wallis**: Numeric outcome, 3+ groups, non-normal distribution
    - **Tukey HSD**: Post-hoc test after significant ANOVA
    
    **Relationships Between Variables:**
    - **Pearson Correlation**: Two numeric variables, linear relationship
    - **Spearman Correlation**: Two numeric variables, monotonic relationship
    - **Simple Linear Regression**: Predict one numeric variable from another
    
    **Categorical Data:**
    - **Chi-square**: Independence of two categorical variables
    - **Fisher's Exact**: 2x2 tables, small sample sizes
    - **Two-proportion z-test**: Compare proportions between two groups
    
    **Binary Outcomes:**
    - **Logistic Regression**: Predict binary outcome from numeric predictor
    
    ### Interpreting Results
    
    - **p-value < α (typically 0.05)**: Reject null hypothesis, result is statistically significant
    - **Effect Size**: Magnitude of the difference/relationship (more important than p-value!)
    - **Confidence Interval**: Range of plausible values for the true effect
    
    ### Assumptions
    
    Always check test assumptions before interpreting results. The system will warn you when assumptions are violated and suggest alternatives.
    """)

# Navigation
st.divider()
st.markdown("### 📋 Next Steps")

nav_cols = st.columns(3)

with nav_cols[0]:
    if st.button("📊 Visualize Data", use_container_width=True):
        st.switch_page("pages/4_Visualization.py")

with nav_cols[1]:
    if st.button("🤖 Ask AI Assistant", use_container_width=True):
        st.switch_page("pages/6_AI_Assistant.py")

with nav_cols[2]:
    if st.button("📄 Generate Report", use_container_width=True, type="primary"):
        st.switch_page("pages/7_Reports.py")
