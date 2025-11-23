import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.utils import initialize_session_state
from modules.data_balancer import DataBalancer
import io

st.title("⚖️ Data Balancer")

st.markdown("""
Balance your dataset for machine learning by handling class imbalance in your target variable.
Choose from multiple sampling techniques to create a balanced dataset suitable for model training.
""")

initialize_session_state()

if st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset on the Home page first.")
    st.stop()

df = st.session_state.dataset

balancer = DataBalancer()

if 'balanced_data' not in st.session_state:
    st.session_state.balanced_data = None
if 'balancing_result' not in st.session_state:
    st.session_state.balancing_result = None

st.divider()

st.info("""
⚠️ **Important:** Balanced data will have a different number of rows than your cleaned dataset. 
Use balanced columns directly for model training. Do not mix balanced data with your original cleaned dataset.
""")

st.divider()
st.subheader("📋 Step 1: Select Columns")

st.markdown("""
**Select feature columns** (must be numeric) and a **target column** (the class you want to balance).
""")

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
all_cols = df.columns.tolist()

col1, col2 = st.columns(2)

with col1:
    feature_cols = st.multiselect(
        "Select feature columns (numeric only):",
        options=numeric_cols,
        help="These columns will be used as input features. Only numeric columns can be selected.",
        key='feature_selection'
    )

with col2:
    target_col = st.selectbox(
        "Select target column (class to balance):",
        options=[''] + all_cols,
        help="This column contains the class labels you want to balance. Can be numeric or categorical.",
        key='target_selection'
    )

if target_col and target_col != '':
    st.divider()
    st.subheader("📊 Current Class Distribution")
    
    dist = df[target_col].value_counts().sort_index()
    
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        st.markdown("**Class Counts:**")
        dist_df = pd.DataFrame({
            'Class': dist.index.astype(str),
            'Count': dist.values,
            'Percentage': (dist.values / len(df) * 100).round(2)
        })
        st.dataframe(dist_df, use_container_width=True, hide_index=True)
    
    with col_dist2:
        fig = go.Figure(data=[
            go.Bar(
                x=dist.index.astype(str),
                y=dist.values,
                text=dist.values,
                textposition='auto',
                marker_color='lightblue'
            )
        ])
        fig.update_layout(
            title="Class Distribution",
            xaxis_title="Class",
            yaxis_title="Count",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    imbalance_ratio = dist.max() / dist.min() if dist.min() > 0 else float('inf')
    if imbalance_ratio > 1.5:
        st.warning(f"⚠️ Dataset is imbalanced. Ratio: {imbalance_ratio:.2f}:1 (majority:minority)")
    else:
        st.success(f"✅ Dataset is relatively balanced. Ratio: {imbalance_ratio:.2f}:1")

st.divider()
st.subheader("⚙️ Step 2: Choose Balancing Method")

methods_dict = balancer.get_available_methods()

tab_os, tab_us, tab_hybrid, tab_advanced = st.tabs([
    "📈 Oversampling", 
    "📉 Undersampling", 
    "🔄 Hybrid", 
    "🚀 Advanced"
])

with tab_os:
    st.markdown("""
    **Oversampling** increases the number of minority class samples.
    
    - **Random Oversampling**: Randomly duplicates minority class samples
    - **SMOTE**: Creates synthetic samples using k-nearest neighbors
    """)
    
    for method in methods_dict['Oversampling']:
        if st.button(f"Select {method}", key=f"btn_os_{method}", use_container_width=True):
            st.session_state.selected_method = method

with tab_us:
    st.markdown("""
    **Undersampling** reduces the number of majority class samples.
    
    - **Random Undersampling**: Randomly removes majority class samples
    - **Tomek Links**: Removes borderline majority samples
    - **NearMiss-1/2/3**: Selects majority samples based on distance to minority samples
    - **ENN**: Removes samples whose class differs from majority of neighbors
    - **CNN**: Finds consistent subset of majority class
    - **OSS**: Combines Tomek Links and CNN
    - **Cluster Centroids**: Replaces majority samples with cluster centroids
    - **NCR**: Cleans data by removing noisy samples
    """)
    
    for method in methods_dict['Undersampling']:
        if st.button(f"Select {method}", key=f"btn_us_{method}", use_container_width=True):
            st.session_state.selected_method = method

with tab_hybrid:
    st.markdown("""
    **Hybrid Methods** combine oversampling and undersampling.
    
    - **SMOTE + Tomek Links**: First applies SMOTE, then removes Tomek links
    - **SMOTE + ENN**: First applies SMOTE, then cleans with ENN
    """)
    
    for method in methods_dict['Hybrid']:
        if st.button(f"Select {method}", key=f"btn_hybrid_{method}", use_container_width=True):
            st.session_state.selected_method = method

with tab_advanced:
    st.markdown("""
    **Advanced Methods** use sophisticated techniques (coming soon).
    
    - **GAN Oversampling**: Uses Generative Adversarial Networks
    - **VAE Oversampling**: Uses Variational Autoencoders
    - **Cost-Sensitive Learning**: Adjusts learning algorithm to penalize misclassifications
    """)
    
    st.info("🚧 Advanced methods are not yet implemented. These require additional deep learning dependencies.")
    
    for method in methods_dict['Advanced']:
        if st.button(f"Select {method} (Not Available)", key=f"btn_adv_{method}", use_container_width=True, disabled=True):
            pass

if 'selected_method' in st.session_state:
    st.success(f"✅ Selected method: **{st.session_state.selected_method}**")

st.divider()
st.subheader("🚀 Step 3: Apply Balancing")

col_apply1, col_apply2 = st.columns([2, 1])

with col_apply1:
    if st.button("⚖️ Apply Balancing", type="primary", use_container_width=True, 
                 disabled=not (feature_cols and target_col and target_col != '' and 'selected_method' in st.session_state)):
        
        validation = balancer.validate_data(df, feature_cols, target_col)
        
        if not validation['valid']:
            st.error("❌ **Data Validation Failed:**")
            for error in validation['errors']:
                st.error(f"• {error}")
            if validation['warnings']:
                st.warning("⚠️ **Warnings:**")
                for warning in validation['warnings']:
                    st.warning(f"• {warning}")
        else:
            if validation['warnings']:
                st.warning("⚠️ **Warnings:**")
                for warning in validation['warnings']:
                    st.warning(f"• {warning}")
            
            with st.spinner(f"Applying {st.session_state.selected_method}..."):
                result = balancer.balance_data(
                    df=df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    method=st.session_state.selected_method,
                    random_state=42
                )
                
                if result['success']:
                    st.session_state.balanced_data = result['balanced_data']
                    st.session_state.balancing_result = result
                    st.success(f"✅ Balancing completed successfully using {result['method']}!")
                    st.rerun()
                else:
                    st.error(f"❌ Balancing failed: {result['error']}")

with col_apply2:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.balanced_data = None
        st.session_state.balancing_result = None
        if 'selected_method' in st.session_state:
            del st.session_state.selected_method
        st.success("Reset successful")
        st.rerun()

if st.session_state.balancing_result:
    st.divider()
    st.subheader("📊 Step 4: Results")
    
    result = st.session_state.balancing_result
    
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    
    with col_metrics1:
        st.metric("Original Size", f"{result['original_size']:,} rows")
    
    with col_metrics2:
        st.metric("Balanced Size", f"{result['balanced_size']:,} rows")
    
    with col_metrics3:
        size_change = ((result['balanced_size'] - result['original_size']) / result['original_size'] * 100)
        st.metric("Size Change", f"{size_change:+.1f}%")
    
    st.divider()
    st.markdown("### 📊 Before vs After Class Distribution")
    
    col_before, col_after = st.columns(2)
    
    with col_before:
        st.markdown("**Before Balancing:**")
        before_df = pd.DataFrame({
            'Class': result['original_distribution'].index.astype(str),
            'Count': result['original_distribution'].values,
            'Percentage': (result['original_distribution'].values / result['original_size'] * 100).round(2)
        })
        st.dataframe(before_df, use_container_width=True, hide_index=True)
        
        fig_before = go.Figure(data=[
            go.Bar(
                x=result['original_distribution'].index.astype(str),
                y=result['original_distribution'].values,
                text=result['original_distribution'].values,
                textposition='auto',
                marker_color='lightcoral',
                name='Before'
            )
        ])
        fig_before.update_layout(
            title="Original Distribution",
            xaxis_title="Class",
            yaxis_title="Count",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_before, use_container_width=True)
    
    with col_after:
        st.markdown("**After Balancing:**")
        after_df = pd.DataFrame({
            'Class': result['balanced_distribution'].index.astype(str),
            'Count': result['balanced_distribution'].values,
            'Percentage': (result['balanced_distribution'].values / result['balanced_size'] * 100).round(2)
        })
        st.dataframe(after_df, use_container_width=True, hide_index=True)
        
        fig_after = go.Figure(data=[
            go.Bar(
                x=result['balanced_distribution'].index.astype(str),
                y=result['balanced_distribution'].values,
                text=result['balanced_distribution'].values,
                textposition='auto',
                marker_color='lightgreen',
                name='After'
            )
        ])
        fig_after.update_layout(
            title="Balanced Distribution",
            xaxis_title="Class",
            yaxis_title="Count",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_after, use_container_width=True)
    
    st.divider()
    st.subheader("💾 Step 5: Download Balanced Data")
    
    st.warning("""
    ⚠️ **Important Warning:**
    
    Balanced data will have a different number of rows than your cleaned dataset. 
    Use balanced columns directly for model training. Do not attempt to merge or join 
    this balanced data back with your original cleaned dataset.
    
    The balanced dataset contains only the feature columns and target column you selected.
    """)
    
    col_download1, col_download2, col_download3 = st.columns(3)
    
    with col_download1:
        csv_buffer = io.StringIO()
        st.session_state.balanced_data.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"balanced_data_{result['method'].replace(' ', '_').lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_download2:
        excel_buffer = io.BytesIO()
        st.session_state.balanced_data.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_data = excel_buffer.getvalue()
        
        st.download_button(
            label="📥 Download as Excel",
            data=excel_data,
            file_name=f"balanced_data_{result['method'].replace(' ', '_').lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col_download3:
        if st.button("👀 Preview Balanced Data", use_container_width=True):
            st.session_state.show_balanced_preview = not st.session_state.get('show_balanced_preview', False)
    
    if st.session_state.get('show_balanced_preview', False):
        st.divider()
        st.markdown("### 🔍 Balanced Data Preview")
        st.dataframe(st.session_state.balanced_data.head(100), use_container_width=True)
        
        st.caption(f"Showing first 100 rows of {len(st.session_state.balanced_data)} total rows")

st.divider()

with st.expander("📚 Guide to Data Balancing"):
    st.markdown("""
    ### Why Balance Data?
    
    Machine learning algorithms often perform poorly on imbalanced datasets where one class 
    significantly outnumbers others. Balancing helps create better models by:
    
    - Preventing bias toward majority class
    - Improving model performance on minority classes
    - Creating more robust predictions
    
    ### Which Method to Choose?
    
    **For Small Datasets:**
    - Use **SMOTE** (creates synthetic samples)
    - Avoid heavy undersampling (loses too much data)
    
    **For Large Datasets:**
    - **Random Undersampling** works well
    - **Cluster Centroids** for efficient reduction
    
    **For Best Results:**
    - Try **Hybrid methods** (SMOTE + Tomek Links or SMOTE + ENN)
    - These combine benefits of both approaches
    
    **For Borderline Cases:**
    - **Tomek Links** removes ambiguous samples
    - **ENN** cleans noisy data
    
    ### Important Considerations
    
    1. **Data Loss**: Undersampling removes data, which may lose important information
    2. **Overfitting**: Oversampling may cause overfitting if not used carefully
    3. **Validation**: Always use cross-validation to evaluate balanced models
    4. **Original Data**: Keep your original cleaned data separate from balanced data
    5. **Model Training**: Use balanced data ONLY for training, not for final analysis
    
    ### Edge Cases Handled
    
    - **Missing Values**: System blocks balancing if any missing values exist in selected columns
    - **Non-Numeric Features**: Only numeric columns can be used as features
    - **Too Few Classes**: Target must have at least 2 classes
    - **Too Many Classes**: Warning shown if more than 10 classes (may not balance well)
    - **Insufficient Data**: At least 10 rows required for balancing
    """)

st.divider()
st.markdown("### 📋 Next Steps")

nav_cols = st.columns(3)
with nav_cols[0]:
    if st.button("🤖 Get AI Assistance", use_container_width=True):
        st.switch_page("pages/7_AI_Assistant.py")

with nav_cols[1]:
    if st.button("📊 Visualize Data", use_container_width=True):
        st.switch_page("pages/4_Visualization.py")

with nav_cols[2]:
    if st.button("📄 Generate Report", use_container_width=True, type="primary"):
        st.switch_page("pages/8_Reports.py")
