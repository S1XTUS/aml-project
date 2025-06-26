import streamlit as st
import requests
import json
from datetime import datetime, timezone
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="AML Risk Scoring & SAR Generator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .high-risk {
        border-left-color: #ff4444 !important;
        background: #fff5f5;
    }
    .medium-risk {
        border-left-color: #ff9500 !important;
        background: #fff8f0;
    }
    .low-risk {
        border-left-color: #28a745 !important;
        background: #f0fff4;
    }
    .transaction-card {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .transaction-card:hover {
        background: #e9ecef;
        border-color: #1f77b4;
    }
    .sar-display {
        background: #f8f9fa;
        color: #212529;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        max-height: 400px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        line-height: 1.5;
    }
    .sar-display strong {
        color: #0d6efd;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🏦 AML Risk Scoring & SAR Generator</h1></div>', unsafe_allow_html=True)

# Initialize session state
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = None
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []
if 'sar_reports' not in st.session_state:
    st.session_state.sar_reports = {}
if 'selected_transaction' not in st.session_state:
    st.session_state.selected_transaction = None
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "http://127.0.0.1:8000"

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Configuration
    st.subheader("API Settings")
    api_base_url = st.text_input(
        "API Base URL", 
        value=st.session_state.api_base_url,
        help="Base URL for the AML API endpoints"
    )
    st.session_state.api_base_url = api_base_url
    
    # Test API Connection
    if st.button("🔗 Test API Connection"):
        try:
            response = requests.get(f"{api_base_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API Connection Successful")
            else:
                st.error(f"❌ API returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection failed: {str(e)}")
    
    st.divider()
    
    # Risk Thresholds
    st.subheader("Risk Thresholds")
    high_risk_threshold = st.slider("High Risk Threshold", 0.0, 1.0, 0.7, 0.01)
    medium_risk_threshold = st.slider("Medium Risk Threshold", 0.0, 1.0, 0.4, 0.01)
    
    st.divider()
    
    # Clear History
    if st.button("🗑️ Clear Transaction History"):
        st.session_state.transaction_history = []
        st.session_state.sar_reports = {}
        st.session_state.selected_transaction = None
        st.success("Transaction history cleared!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Transaction Input Form
    st.header("📝 Transaction Details")
    
    with st.form("transaction_form"):
        # Form fields in a more organized layout
        col_left, col_right = st.columns(2)
        
        with col_left:
            case_id = st.text_input("Case ID", value=f"CASE-{datetime.now().strftime('%Y%m%d')}-{len(st.session_state.transaction_history) + 1:03d}")
            timestamp = st.text_input("Transaction Time (ISO8601)", value=datetime.now(timezone.utc).isoformat())
            from_bank = st.text_input("From Bank", value="Alpha Bank")
            from_account = st.text_input("From Account", value="1234567890")
            amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
        
        with col_right:
            currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"])
            to_bank = st.text_input("To Bank", value="Beta Bank")
            to_account = st.text_input("To Account", value="9876543210")
            
            transaction_type_options = {
            "Wire Transfer": "wire_transfer",
            "ACH": "ach", 
            "Check": "check",
            "Cash Deposit": "cash_deposit",
            "International Transfer": "international_transfer"
            }

            transaction_type_display = st.selectbox("Transaction Type", list(transaction_type_options.keys()))
            transaction_type = transaction_type_options[transaction_type_display]
            jurisdiction = st.selectbox("Destination Jurisdiction", 
                                      ["Domestic", "EU", "Offshore", "High-Risk Country"])
        
        # Form submission buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            get_risk_score = st.form_submit_button("🎯 Get Risk Score", use_container_width=True)
        
        with col_btn2:
            generate_sar = st.form_submit_button("📄 Generate SAR", use_container_width=True)
        
        with col_btn3:
            save_transaction = st.form_submit_button("💾 Save Transaction", use_container_width=True)

# Risk Score Calculation
if get_risk_score:
    with st.spinner("Calculating risk score..."):
        risk_payload = {
            "case_id": case_id,
            "transaction_time": timestamp,
            "from_bank": from_bank,
            "from_account": from_account,
            "to_bank": to_bank,
            "to_account": to_account,
            "amount": amount,
            "currency": currency,
            "transaction_type": transaction_type,
            "jurisdiction": jurisdiction
        }
        
        try:
            response = requests.post(f"{api_base_url}/predict-risk", json=risk_payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                risk_score = result.get("risk_score", 0)
                st.session_state.risk_score = risk_score
                
                # Determine risk level and styling
                if risk_score >= high_risk_threshold:
                    risk_level = "HIGH"
                    risk_class = "high-risk"
                    risk_color = "#ff4444"
                elif risk_score >= medium_risk_threshold:
                    risk_level = "MEDIUM"
                    risk_class = "medium-risk"
                    risk_color = "#ff9500"
                else:
                    risk_level = "LOW"
                    risk_class = "low-risk"
                    risk_color = "#28a745"
                
                st.success(f"✅ Risk assessment completed successfully!")
                
                # Display risk score with visual indicator
                st.markdown(f"""
                <div class="metric-card {risk_class}">
                    <h3>Risk Score: {risk_score:.4f}</h3>
                    <h4>Risk Level: {risk_level}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk score gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Score"},
                    delta = {'reference': medium_risk_threshold},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, medium_risk_threshold], 'color': "lightgray"},
                            {'range': [medium_risk_threshold, high_risk_threshold], 'color': "yellow"},
                            {'range': [high_risk_threshold, 1], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': high_risk_threshold
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"❌ API Error: {response.status_code}")
                with st.expander("Error Details"):
                    st.code(response.text)
                    
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")

# SAR Generation
if generate_sar:
    if st.session_state.risk_score is None:
        st.warning("⚠️ Please calculate risk score first!")
    else:
        with st.spinner("Generating SAR narrative..."):
            sar_payload = {
                "case_id": case_id,
                "transaction_time": timestamp,
                "from_bank": from_bank,
                "from_account": from_account,
                "to_bank": to_bank,
                "to_account": to_account,
                "amount": amount,
                "currency": currency,
                "risk_score": st.session_state.risk_score,
                "transaction_type": transaction_type,
                "jurisdiction": jurisdiction,
                "anomaly_flag": st.session_state.risk_score > medium_risk_threshold,
                "pattern_summary": f"{transaction_type} to {jurisdiction.lower()} jurisdiction with amount ${amount:,.2f}.",
                "kyc_summary": "Customer profile requires enhanced due diligence review.",
                "regulatory_reference": "FinCEN Advisory FIN-2023-A002"
            }
            
            try:
                # Increased timeout to 60 seconds for SAR generation
                response = requests.post(f"{api_base_url}/generate-sar", json=sar_payload, timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    sar_narrative = result.get("sar_narrative", "")
                    
                    # Store SAR report in session state
                    st.session_state.sar_reports[case_id] = {
                        "narrative": sar_narrative,
                        "generated_at": datetime.now().isoformat(),
                        "case_details": sar_payload
                    }
                    
                    st.success("✅ SAR narrative generated successfully!")
                    
                    # Display SAR with better formatting
                    st.subheader("📄 Generated SAR Narrative")
                    st.text_area("SAR Content", sar_narrative, height=300, disabled=True)
                    
                    # Download button for SAR
                    st.download_button(
                        label="💾 Download SAR",
                        data=sar_narrative,
                        file_name=f"SAR_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
                else:
                    st.error(f"❌ SAR Generation Error: {response.status_code}")
                    with st.expander("Error Details"):
                        st.code(response.text)
                        
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")

# Save Transaction
if save_transaction:
    transaction_data = {
        "case_id": case_id,
        "timestamp": timestamp,
        "from_bank": from_bank,
        "from_account": from_account,
        "to_bank": to_bank,
        "to_account": to_account,
        "amount": amount,
        "currency": currency,
        "transaction_type": transaction_type,
        "jurisdiction": jurisdiction,
        "risk_score": st.session_state.risk_score,
        "saved_at": datetime.now().isoformat()
    }
    
    st.session_state.transaction_history.append(transaction_data)
    st.success(f"✅ Transaction {case_id} saved to history!")

# Right column - Dashboard and History
with col2:
    st.header("📊 Dashboard")
    
    # Current Risk Score Display
    if st.session_state.risk_score is not None:
        risk_score = st.session_state.risk_score
        if risk_score >= high_risk_threshold:
            st.metric("Current Risk Score", f"{risk_score:.4f}", "HIGH RISK", delta_color="inverse")
        elif risk_score >= medium_risk_threshold:
            st.metric("Current Risk Score", f"{risk_score:.4f}", "MEDIUM RISK", delta_color="off")
        else:
            st.metric("Current Risk Score", f"{risk_score:.4f}", "LOW RISK", delta_color="normal")
    else:
        st.metric("Current Risk Score", "N/A", "Not Calculated")
    
    # Transaction History with Clickable Cards
    st.subheader("📈 Transaction History")
    
    if st.session_state.transaction_history:
        # Create DataFrame for history
        df = pd.DataFrame(st.session_state.transaction_history)
        
        # Summary metrics
        total_transactions = len(df)
        total_amount = df['amount'].sum()
        avg_risk_score = df[df['risk_score'].notna()]['risk_score'].mean() if not df[df['risk_score'].notna()].empty else 0
        
        col_met1, col_met2 = st.columns(2)
        with col_met1:
            st.metric("Total Transactions", total_transactions)
        with col_met2:
            st.metric("Total Amount", f"${total_amount:,.2f}")
        
        st.metric("Avg Risk Score", f"{avg_risk_score:.4f}" if avg_risk_score > 0 else "N/A")
        
        # Recent transactions as clickable cards
        st.subheader("Recent Transactions")
        
        # Show last 10 transactions
        recent_transactions = st.session_state.transaction_history[-10:]
        recent_transactions.reverse()  # Show newest first
        
        for i, transaction in enumerate(recent_transactions):
            case_id_display = transaction['case_id']
            risk_score_display = transaction.get('risk_score', 'N/A')
            amount_display = transaction['amount']
            currency_display = transaction['currency']
            
            # Determine risk level for styling
            risk_level = "N/A"
            risk_color = "#6c757d"
            if risk_score_display != 'N/A':
                if risk_score_display >= high_risk_threshold:
                    risk_level = "HIGH"
                    risk_color = "#dc3545"
                elif risk_score_display >= medium_risk_threshold:
                    risk_level = "MEDIUM"
                    risk_color = "#fd7e14"
                else:
                    risk_level = "LOW"
                    risk_color = "#28a745"
            
            # Create clickable transaction card
            card_key = f"transaction_card_{case_id_display}_{i}"
            
            if st.button(
                f"📄 {case_id_display}\n💰 {currency_display} {amount_display:,.2f}\n🎯 Risk: {risk_score_display} ({risk_level})",
                key=card_key,
                use_container_width=True
            ):
                st.session_state.selected_transaction = transaction
        
        # Display selected transaction details and SAR
        if st.session_state.selected_transaction:
            st.divider()
            selected = st.session_state.selected_transaction
            
            st.subheader(f"📋 Transaction Details: {selected['case_id']}")
            
            # Transaction details in columns
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.write(f"**From:** {selected['from_bank']}")
                st.write(f"**Account:** {selected['from_account']}")
                st.write(f"**Amount:** {selected['currency']} {selected['amount']:,.2f}")
                st.write(f"**Type:** {selected.get('transaction_type', 'N/A')}")
            
            with detail_col2:
                st.write(f"**To:** {selected['to_bank']}")
                st.write(f"**Account:** {selected['to_account']}")
                st.write(f"**Risk Score:** {selected.get('risk_score', 'N/A')}")
                st.write(f"**Jurisdiction:** {selected.get('jurisdiction', 'N/A')}")
            
            # Show SAR if available
            case_id_selected = selected['case_id']
            if case_id_selected in st.session_state.sar_reports:
                st.subheader("📄 SAR Report")
                sar_data = st.session_state.sar_reports[case_id_selected]
                
                # Format the narrative for HTML display
                formatted_narrative = sar_data['narrative'].replace('\n', '<br>')
                
                st.markdown(f"""
                <div class="sar-display">
                    <strong>Generated:</strong> {sar_data['generated_at']}<br><br>
                    <strong>SAR Narrative:</strong><br>
                    {formatted_narrative}
                </div>
                """, unsafe_allow_html=True)
                
                # Download button for this SAR
                st.download_button(
                    label="💾 Download This SAR",
                    data=sar_data['narrative'],
                    file_name=f"SAR_{case_id_selected}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            else:
                st.info("No SAR report generated for this transaction yet.")
        
        # Risk distribution chart
        if not df[df['risk_score'].notna()].empty:
            st.subheader("📊 Risk Distribution")
            risk_data = df[df['risk_score'].notna()]
            fig = px.histogram(risk_data, x='risk_score', nbins=10, title="Risk Score Distribution")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Export History to CSV",
            data=csv,
            file_name=f"transaction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No transaction history available. Start by calculating some risk scores!")

# Footer
st.markdown("---")
st.markdown("*AML Risk Scoring Dashboard - Built with Streamlit*")