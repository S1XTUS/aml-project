import html
import os
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
if 'loaded_sample' not in st.session_state:
    st.session_state.loaded_sample = None

# Real labelled transactions built by src/data/build_test_samples.py
SAMPLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "samples", "test_transactions.csv")
SOURCE_LABELS = {
    "ibm_li_small_test": "IBM LI-Small (model's held-out test split)",
    "ibm_hi_small": "IBM HI-Small (unseen, same simulator)",
    "saml_d": "SAML-D (different simulator)",
    "amlsim": "IBM AMLSim (different simulator)",
}
CURRENCY_OPTIONS = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR", "RUB", "BRL",
                    "MXN", "SAR", "ILS", "BTC", "AED", "MAD", "NGN", "PKR", "TRY", "ALL"]
TRANSACTION_TYPE_OPTIONS = {
    "Wire Transfer": "wire_transfer",
    "ACH": "ach",
    "Check": "check",
    "Cash": "cash",
    "Cash Deposit": "cash_deposit",
    "Cash Withdrawal": "cash_withdrawal",
    "Credit Card": "credit_card",
    "Debit Card": "debit_card",
    "Bitcoin": "bitcoin",
    "Reinvestment": "reinvestment",
    "International Transfer": "international_transfer",
}
TRANSACTION_TYPE_LABELS = {value: label for label, value in TRANSACTION_TYPE_OPTIONS.items()}
JURISDICTION_OPTIONS = ["Domestic", "EU", "International", "Offshore", "High-Risk Country"]

# Form field defaults (widgets read and write these session_state keys)
FORM_DEFAULTS = {
    "f_case_id": f"CASE-{datetime.now().strftime('%Y%m%d')}-001",
    "f_timestamp": datetime.now(timezone.utc).isoformat(),
    "f_from_bank": "Alpha Bank",
    "f_from_account": "1234567890",
    "f_amount": 1000.0,
    "f_currency": "USD",
    "f_to_bank": "Beta Bank",
    "f_to_account": "9876543210",
    "f_transaction_type": "Wire Transfer",
    "f_jurisdiction": "Domestic",
}
for key, value in FORM_DEFAULTS.items():
    st.session_state.setdefault(key, value)


@st.cache_data
def load_samples() -> pd.DataFrame:
    text_cols = ["record_id", "from_bank", "from_account", "to_bank", "to_account"]
    df = pd.read_csv(SAMPLES_PATH, dtype={c: str for c in text_cols})
    df["receiver_country"] = df["receiver_country"].fillna("")
    return df


def load_sample_into_form(row: pd.Series):
    st.session_state.f_case_id = f"{row['source'].upper()}-{row['record_id']}"
    st.session_state.f_timestamp = row["timestamp"]
    st.session_state.f_from_bank = row["from_bank"]
    st.session_state.f_from_account = row["from_account"]
    st.session_state.f_amount = float(row["amount"])
    st.session_state.f_currency = row["currency"]
    st.session_state.f_to_bank = row["to_bank"]
    st.session_state.f_to_account = row["to_account"]
    st.session_state.f_transaction_type = TRANSACTION_TYPE_LABELS[row["transaction_type"]]
    st.session_state.f_jurisdiction = row["jurisdiction"]
    st.session_state.loaded_sample = row.to_dict()
    st.session_state.risk_score = None


def format_risk(score) -> str:
    """Risk probability (0-1) shown as a percentage."""
    if score is None or pd.isna(score):
        return "N/A"
    return f"{score * 100:.2f}%"


def case_payload(row: pd.Series) -> dict:
    return {
        "case_id": f"{row['source'].upper()}-{row['record_id']}",
        "transaction_time": row["timestamp"],
        "from_bank": row["from_bank"],
        "from_account": row["from_account"],
        "to_bank": row["to_bank"],
        "to_account": row["to_account"],
        "amount": float(row["amount"]),
        "currency": row["currency"],
        "transaction_type": row["transaction_type"],
        "jurisdiction": row["jurisdiction"],
    }

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

    # Real test data
    st.subheader("📂 Test Data")
    if os.path.exists(SAMPLES_PATH):
        samples = load_samples()
        sample_source = st.selectbox("Dataset", list(SOURCE_LABELS), format_func=SOURCE_LABELS.get)
        sample_label = st.radio("Ground truth", ["Any", "Laundering", "Normal"], horizontal=True)
        if st.button("🎲 Load random transaction", use_container_width=True):
            pool = samples[samples["source"] == sample_source]
            if sample_label != "Any":
                pool = pool[pool["is_laundering"] == int(sample_label == "Laundering")]
            load_sample_into_form(pool.sample(1).iloc[0])
    else:
        samples = None
        st.info("No test data yet. Build it with:\n\n`python src/data/build_test_samples.py`")

    st.divider()

    # Risk Thresholds
    st.subheader("Risk Thresholds")
    high_risk_threshold = st.slider("High Risk Threshold (%)", 0, 100, 70, 1) / 100
    medium_risk_threshold = st.slider("Medium Risk Threshold (%)", 0, 100, 40, 1) / 100

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

    loaded = st.session_state.loaded_sample
    if loaded:
        truth = "🚨 LAUNDERING" if loaded["is_laundering"] else "✅ NORMAL"
        country = f" · receiver in {loaded['receiver_country']}" if loaded["receiver_country"] else ""
        st.info(f"Loaded from **{SOURCE_LABELS[loaded['source']]}**, record {loaded['record_id']}{country}  \n"
                f"Ground truth: **{truth}** ({loaded['typology']}). "
                f"Editing the fields means the label no longer applies.")

    with st.form("transaction_form"):
        # Form fields in a more organized layout
        col_left, col_right = st.columns(2)

        with col_left:
            case_id = st.text_input("Case ID", key="f_case_id")
            timestamp = st.text_input("Transaction Time (ISO8601)", key="f_timestamp")
            from_bank = st.text_input("From Bank", key="f_from_bank",
                                      help="IBM datasets use numeric bank IDs; names are treated as unknown banks")
            from_account = st.text_input("From Account", key="f_from_account")
            amount = st.number_input("Amount", min_value=0.0, step=100.0, key="f_amount")

        with col_right:
            currency = st.selectbox("Currency", CURRENCY_OPTIONS, key="f_currency")
            to_bank = st.text_input("To Bank", key="f_to_bank")
            to_account = st.text_input("To Account", key="f_to_account")

            transaction_type_display = st.selectbox("Transaction Type", list(TRANSACTION_TYPE_OPTIONS),
                                                    key="f_transaction_type")
            transaction_type = TRANSACTION_TYPE_OPTIONS[transaction_type_display]
            jurisdiction = st.selectbox("Destination Jurisdiction", JURISDICTION_OPTIONS, key="f_jurisdiction")

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
                if result.get("scoring_method") == "rules":
                    st.warning("⚠️ Trained model unavailable - score comes from rule-based fallback.")
                if loaded:
                    flagged = result.get("is_suspicious", False)
                    actual = bool(loaded["is_laundering"])
                    verdict = ("correct" if flagged == actual else
                               "missed laundering" if actual else "false alarm")
                    st.caption(f"Model {'flagged' if flagged else 'did not flag'} this transaction "
                               f"(tuned threshold) · ground truth {'laundering' if actual else 'normal'} → **{verdict}**")

                # Display risk score with visual indicator
                st.markdown(f"""
                <div class="metric-card {risk_class}">
                    <h3>Risk: {format_risk(risk_score)}</h3>
                    <h4>Risk Level: {risk_level}</h4>
                </div>
                """, unsafe_allow_html=True)

                # Risk gauge (percent)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_score * 100,
                    number = {'suffix': "%", 'valueformat': ".2f"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk"},
                    delta = {'reference': medium_risk_threshold * 100, 'valueformat': ".2f", 'suffix': " pts"},
                    gauge = {
                        'axis': {'range': [0, 100], 'ticksuffix': "%"},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, medium_risk_threshold * 100], 'color': "lightgray"},
                            {'range': [medium_risk_threshold * 100, high_risk_threshold * 100], 'color': "yellow"},
                            {'range': [high_risk_threshold * 100, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': high_risk_threshold * 100
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
                "transaction_type": transaction_type,
                "jurisdiction": jurisdiction,
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
            st.metric("Current Risk", format_risk(risk_score), "HIGH RISK", delta_color="inverse")
        elif risk_score >= medium_risk_threshold:
            st.metric("Current Risk", format_risk(risk_score), "MEDIUM RISK", delta_color="off")
        else:
            st.metric("Current Risk", format_risk(risk_score), "LOW RISK", delta_color="normal")
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

        st.metric("Avg Risk", format_risk(avg_risk_score) if avg_risk_score > 0 else "N/A")

        # Recent transactions as clickable cards
        st.subheader("Recent Transactions")

        # Show last 10 transactions
        recent_transactions = st.session_state.transaction_history[-10:]
        recent_transactions.reverse()  # Show newest first

        for i, transaction in enumerate(recent_transactions):
            case_id_display = transaction['case_id']
            risk_value = transaction.get('risk_score')
            risk_score_display = format_risk(risk_value)
            amount_display = transaction['amount']
            currency_display = transaction['currency']

            # Determine risk level for styling
            risk_level = "N/A"
            risk_color = "#6c757d"
            if risk_value is not None:
                if risk_value >= high_risk_threshold:
                    risk_level = "HIGH"
                    risk_color = "#dc3545"
                elif risk_value >= medium_risk_threshold:
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
                st.write(f"**Risk:** {format_risk(selected.get('risk_score'))}")
                st.write(f"**Jurisdiction:** {selected.get('jurisdiction', 'N/A')}")

            # Show SAR if available
            case_id_selected = selected['case_id']
            if case_id_selected in st.session_state.sar_reports:
                st.subheader("📄 SAR Report")
                sar_data = st.session_state.sar_reports[case_id_selected]

                # Format the narrative for HTML display
                formatted_narrative = html.escape(sar_data['narrative']).replace('\n', '<br>')

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
            fig = px.histogram(risk_data.assign(risk_pct=risk_data['risk_score'] * 100), x='risk_pct', nbins=10,
                               title="Risk Distribution", labels={'risk_pct': 'Risk (%)'})
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

# Batch test against ground-truth labels
if samples is not None:
    st.markdown("---")
    st.header("🧪 Batch Test Against Labels")
    st.caption("Scores real labelled transactions through the API and compares them with the ground truth. "
               "Samples are 30% laundering (real data is <0.2%), so precision here is far higher than in production. "
               "The API scores each transaction without its account history.")

    bcol1, bcol2 = st.columns([2, 1])
    with bcol1:
        batch_sources = st.multiselect("Datasets", list(SOURCE_LABELS), default=list(SOURCE_LABELS),
                                       format_func=SOURCE_LABELS.get)
    with bcol2:
        per_source = st.slider("Transactions per dataset", 20, 500, 100, 20)

    if st.button("▶️ Run batch test", disabled=not batch_sources):
        batch = pd.concat([
            samples[samples["source"] == src].sample(min(per_source, int((samples["source"] == src).sum())), random_state=0)
            for src in batch_sources
        ])
        results = []
        progress = st.progress(0.0, text="Scoring transactions...")
        try:
            for start in range(0, len(batch), 100):
                chunk = batch.iloc[start:start + 100]
                response = requests.post(f"{api_base_url}/batch-risk-predict",
                                         json=[case_payload(row) for _, row in chunk.iterrows()], timeout=300)
                response.raise_for_status()
                results.extend(response.json()["results"])
                done = min(start + 100, len(batch))
                progress.progress(done / len(batch), text=f"Scored {done}/{len(batch)}")
            st.session_state.batch_results = batch.assign(
                risk_score=[r.get("risk_score") for r in results],
                is_suspicious=[bool(r.get("is_suspicious")) for r in results],
                scoring_method=[r.get("scoring_method") for r in results],
            )
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Batch scoring failed: {e}")
        progress.empty()

    batch_results = st.session_state.get("batch_results")
    if batch_results is not None:
        from sklearn.metrics import roc_auc_score

        summary = []
        for src, group in batch_results.groupby("source"):
            y, flagged = group["is_laundering"] == 1, group["is_suspicious"]
            true_pos = int((y & flagged).sum())
            summary.append({
                "Dataset": SOURCE_LABELS[src],
                "Transactions": len(group),
                "Laundering": int(y.sum()),
                "ROC AUC": round(roc_auc_score(y, group["risk_score"]), 3) if y.nunique() == 2 else None,
                "Flagged": int(flagged.sum()),
                "Precision": round(true_pos / flagged.sum(), 3) if flagged.sum() else None,
                "Recall": round(true_pos / y.sum(), 3) if y.sum() else None,
                "Avg risk (laundering)": format_risk(group.loc[y, "risk_score"].mean()),
                "Avg risk (normal)": format_risk(group.loc[~y, "risk_score"].mean()),
            })
        st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
        if (batch_results["scoring_method"] == "rules").any():
            st.warning("⚠️ Some scores came from the rule-based fallback - check the API has a trained model.")

        plot_df = batch_results.assign(label=batch_results["is_laundering"].map({1: "laundering", 0: "normal"}),
                                       dataset=batch_results["source"].map(SOURCE_LABELS),
                                       risk_pct=batch_results["risk_score"] * 100)
        fig = px.box(plot_df, x="dataset", y="risk_pct", color="label", points="outliers",
                     title="Risk by ground truth", labels={"risk_pct": "Risk (%)"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Scored transactions"):
            st.dataframe(batch_results.assign(risk=batch_results["risk_score"].map(format_risk))[
                             ["source", "record_id", "timestamp", "amount", "currency", "transaction_type",
                              "typology", "is_laundering", "risk", "is_suspicious"]],
                         hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*AML Risk Scoring Dashboard - Built with Streamlit*")