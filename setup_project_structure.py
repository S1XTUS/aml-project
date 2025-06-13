import os

# Define all folders and empty files to be created
structure = {
    "intelligent_aml_assistant": [
        "README.md",
        "requirements.txt",
        "config.yaml",
        "data/raw/",
        "data/processed/",
        "data/kyc_docs/",
        "data/regulations_corpus/",
        "data/external/",
        "notebooks/01_data_exploration.ipynb",
        "notebooks/02_anomaly_detection_model.ipynb",
        "notebooks/03_risk_scoring_model.ipynb",
        "notebooks/04_graph_entity_analysis.ipynb",
        "notebooks/05_llm_sar_generation.ipynb",
        "notebooks/06_kyc_validation_with_llm.ipynb",
        "notebooks/07_rag_regulatory_assistant_demo.ipynb",
        "src/data/load_data.py",
        "src/data/enrich_osint.py",
        "src/data/preprocess_kyc.py",
        "src/models/anomaly_detector.py",
        "src/models/risk_classifier.py",
        "src/models/explainer.py",
        "src/llm/sar_generator.py",
        "src/llm/rag_research_assistant.py",
        "src/llm/kyc_validator.py",
        "src/llm/edd_processor.py",
        "src/realtime/stream_listener.py",
        "src/realtime/realtime_scorer.py",
        "src/utils/config_loader.py",
        "src/utils/logger.py",
        "src/utils/prompts.py",
        "app/dashboard.py",
        "app/api.py",
        "app/sar_submission_form.py",
        "tests/test_models.py",
        "tests/test_llm_outputs.py",
        "tests/test_data_pipeline.py",
        "reports/figures/",
        "reports/sar_samples/",
        "reports/final_report.pdf",
        "demo/sample_alert.json",
        "demo/sample_kyc.txt",
        "demo/sample_sar_output.txt",
        "demo/screencast.mp4"
    ]
}


for root, items in structure.items():
    for path in items:
        full_path = os.path.join(root, path)
        if full_path.endswith("/"):
            os.makedirs(full_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write("")

print("Project structure created successfully!")