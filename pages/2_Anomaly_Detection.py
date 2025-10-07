import streamlit as st
import pandas as pd
from modules.utils import initialize_session_state, create_backup, save_cleaning_operation
from datetime import datetime

initialize_session_state()

st.title("🔍 Data Type Anomaly Detection")

if st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset in the Data Upload page first.")
    st.stop()

st.markdown("""
This section detects values that don't match their declared data type. For example:
- String values (like 'thirthy') in numeric columns declared as integers
- Invalid date formats in datetime columns
- Unexpected values in binary or categorical columns

Detecting these anomalies early prevents errors in later analysis steps.
""")

df = st.session_state.dataset
column_types = st.session_state.column_types

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("1. Select Column to Check")
    
    selected_column = st.selectbox(
        "Choose a column to analyze for anomalies",
        options=list(df.columns),
        help="Select which column you want to check for data type mismatches"
    )

with col2:
    st.subheader("Column Information")
    if selected_column:
        expected_type = column_types.get(selected_column, 'unknown')
        st.metric("Expected Type", expected_type.title())
        st.metric("Total Values", len(df[selected_column]))
        st.metric("Null Values", df[selected_column].isnull().sum())

st.divider()

if selected_column:
    detector = st.session_state.anomaly_detector
    
    col_scan, col_refresh = st.columns([3, 1])
    with col_scan:
        st.subheader("2. Scan for Anomalies")
    with col_refresh:
        if st.button("🔄 Re-scan Column", use_container_width=True):
            if selected_column in st.session_state.anomaly_results:
                del st.session_state.anomaly_results[selected_column]
            st.rerun()
    
    if selected_column not in st.session_state.anomaly_results:
        with st.spinner(f"Scanning {selected_column} for anomalies..."):
            expected_type = column_types.get(selected_column, 'unknown')
            anomaly_data = detector.detect_column_anomalies(df, selected_column, expected_type)
            st.session_state.anomaly_results[selected_column] = anomaly_data
            st.session_state.anomaly_last_updated = datetime.now().isoformat()
    
    anomaly_data = st.session_state.anomaly_results[selected_column]
    
    if anomaly_data['anomaly_count'] == 0:
        st.success(f"✅ {anomaly_data['summary']}")
    else:
        st.warning(f"⚠️ {anomaly_data['summary']}")
        
        st.divider()
        
        st.subheader("3. Review Anomalies")
        
        anomalies_df = pd.DataFrame(anomaly_data['anomalies'])
        
        st.dataframe(
            anomalies_df,
            use_container_width=True,
            column_config={
                "row_index": st.column_config.NumberColumn("Row Index", help="Row number in dataset"),
                "value": st.column_config.TextColumn("Anomalous Value", help="The problematic value"),
                "reason": st.column_config.TextColumn("Reason", help="Why this value is anomalous")
            }
        )
        
        st.download_button(
            label="📥 Download Anomalies as CSV",
            data=anomalies_df.to_csv(index=False),
            file_name=f"anomalies_{selected_column}.csv",
            mime="text/csv"
        )
        
        st.divider()
        
        st.subheader("4. Fix Anomalies")
        
        fix_method = st.radio(
            "Choose how to handle these anomalies:",
            options=["Remove All Anomalous Rows", "Replace Values Individually"],
            help="Either remove all rows with anomalies or replace specific values"
        )
        
        if fix_method == "Remove All Anomalous Rows":
            st.warning(f"⚠️ This will remove **{anomaly_data['anomaly_count']} rows** from your dataset.")
            
            col_confirm, col_cancel = st.columns(2)
            
            with col_confirm:
                if st.button("🗑️ Confirm: Remove All Anomalous Rows", type="primary", use_container_width=True):
                    create_backup()
                    
                    anomaly_indices = [a['row_index'] for a in anomaly_data['anomalies']]
                    cleaned_df, summary = detector.remove_anomalies(df, selected_column, anomaly_indices)
                    
                    st.session_state.dataset = cleaned_df
                    
                    save_cleaning_operation({
                        'column': selected_column,
                        'operation': 'remove_anomalies',
                        'details': summary,
                        'rows_affected': summary['rows_removed']
                    })
                    
                    del st.session_state.anomaly_results[selected_column]
                    
                    st.success(f"✅ Successfully removed {summary['rows_removed']} rows with anomalies!")
                    st.rerun()
            
            with col_cancel:
                if st.button("❌ Cancel", use_container_width=True):
                    st.info("Operation cancelled.")
        
        else:
            st.info("Select specific rows to replace values individually.")
            
            for idx, anomaly in enumerate(anomaly_data['anomalies']):
                with st.expander(f"Row {anomaly['row_index']}: {anomaly['value']} - {anomaly['reason']}"):
                    col_show, col_replace = st.columns([2, 1])
                    
                    with col_show:
                        row_data = df.iloc[anomaly['row_index']].to_dict()
                        st.json(row_data)
                    
                    with col_replace:
                        new_value = st.text_input(
                            "New value",
                            key=f"replace_{selected_column}_{anomaly['row_index']}",
                            placeholder="Enter replacement value"
                        )
                        
                        if st.button(f"Replace", key=f"btn_{selected_column}_{anomaly['row_index']}"):
                            if new_value:
                                create_backup()
                                
                                modified_df, summary = detector.replace_anomaly(
                                    df.copy(),
                                    anomaly['row_index'],
                                    selected_column,
                                    new_value
                                )
                                
                                st.session_state.dataset = modified_df
                                
                                save_cleaning_operation({
                                    'column': selected_column,
                                    'operation': 'replace_anomaly',
                                    'details': summary
                                })
                                
                                del st.session_state.anomaly_results[selected_column]
                                
                                st.success(f"✅ Replaced value at row {anomaly['row_index']}")
                                st.rerun()
                            else:
                                st.warning("Please enter a replacement value")
            
            st.divider()
            
            st.subheader("Batch Replace")
            st.markdown("Replace multiple anomalies with the same value:")
            
            batch_value = st.text_input(
                "Enter value to replace all anomalies in this column",
                key=f"batch_{selected_column}",
                placeholder="e.g., 0, NULL, Unknown"
            )
            
            if st.button(f"Replace All {anomaly_data['anomaly_count']} Anomalies with '{batch_value}'", 
                        disabled=not batch_value,
                        type="primary"):
                create_backup()
                
                replacements = {a['row_index']: batch_value for a in anomaly_data['anomalies']}
                modified_df, summary = detector.batch_replace_anomalies(
                    df.copy(),
                    selected_column,
                    replacements
                )
                
                st.session_state.dataset = modified_df
                
                save_cleaning_operation({
                    'column': selected_column,
                    'operation': 'batch_replace_anomalies',
                    'details': summary
                })
                
                del st.session_state.anomaly_results[selected_column]
                
                st.success(f"✅ Replaced {summary['replacements_count']} anomalies!")
                st.rerun()

st.divider()

with st.expander("📊 Scan All Columns for Anomalies", expanded=False):
    st.markdown("Scan all columns at once to get a complete overview of data type issues.")
    
    if st.button("🔍 Scan All Columns", use_container_width=True):
        with st.spinner("Scanning all columns for anomalies..."):
            all_anomalies = detector.detect_all_anomalies(df, column_types)
            
            if not all_anomalies:
                st.success("✅ No anomalies detected in any column!")
            else:
                st.warning(f"⚠️ Found anomalies in {len(all_anomalies)} column(s)")
                
                summary_data = []
                for col, data in all_anomalies.items():
                    summary_data.append({
                        'Column': col,
                        'Expected Type': data['expected_type'],
                        'Anomaly Count': data['anomaly_count'],
                        'Percentage': f"{data['anomaly_percentage']:.2f}%"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                st.download_button(
                    label="📥 Download Full Anomaly Report",
                    data=summary_df.to_csv(index=False),
                    file_name="all_anomalies_summary.csv",
                    mime="text/csv"
                )

st.info("💡 **Tip**: After fixing anomalies, proceed to Column Analysis to examine data quality and patterns.")
