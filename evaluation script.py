import pandas as pd
import numpy as np
import os

def calculate_missed_items(text):
    """
    Derives the missed count strictly from the content of the missed_from_raw_text column.
    Since the data appears as list-like strings containing bracketed items (e.g., [{item1}, {item2}]),
    we can count the occurrences of '{' to accurately gauge the number of missed items.
    """
    if pd.isna(text) or str(text).strip() == '':
        return 0
    return str(text).count('{')

def evaluate_extraction_performance_updated(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
    
    # Read the dataset
    df = pd.read_csv(file_path)
    
    # 1. Generate our own reliable count column based ONLY on the text content
    df['reliable_missed_count'] = df['missed_from_raw_text'].apply(calculate_missed_items)
    
    # 2. Total Metrics
    total_cvs = len(df)
    perfect_extractions = df[df['reliable_missed_count'] == 0]
    num_perfect = len(perfect_extractions)
    pct_perfect = (num_perfect / total_cvs) * 100 if total_cvs > 0 else 0
    
    # 3. Statistical Metrics
    avg_missed = df['reliable_missed_count'].mean()
    median_missed = df['reliable_missed_count'].median()
    max_missed = df['reliable_missed_count'].max()
    total_missed_items = df['reliable_missed_count'].sum()
    
    # 4. Identify outliers / high loss cases using the new reliable count
    high_loss_threshold = df['reliable_missed_count'].quantile(0.75)
    high_loss_candidates = df[df['reliable_missed_count'] > high_loss_threshold][['cv_id', 'reliable_missed_count', 'missed_from_raw_text']].to_dict(orient='records')
    
    # Construct Evaluation Report Payload
    report = {
        "summary_metrics": {
            "total_candidates_evaluated": total_cvs,
            "perfect_extraction_count": num_perfect,
            "perfect_extraction_percentage": round(pct_perfect, 2),
            "total_missed_datapoints": int(total_missed_items)
        },
        "error_distribution": {
            "average_missed_items_per_cv": round(avg_missed, 2),
            "median_missed_items_per_cv": int(median_missed),
            "maximum_missed_items_from_single_cv": int(max_missed)
        },
        "critical_review_candidates": [
            {
                "cv_id": str(cand['cv_id']), 
                "missed_count": int(cand['reliable_missed_count']),
                "sample_missed_content": str(cand['missed_from_raw_text'])[:200] + "..."
            } 
            for cand in high_loss_candidates if cand['reliable_missed_count'] > 0
        ]
    }
    
    return report

def generate_markdown_report_updated(report_data, output_file="evaluation_report_v2.md"):
    if not report_data:
        return
    
    sm = report_data["summary_metrics"]
    ed = report_data["error_distribution"]
    
    md_content = f"""# CV Data Extraction Evaluation Report (Content-Derived Metrics)
---

## Executive Summary
This report analyzes data extraction fidelity by dynamically evaluating the missed text content, bypassing any pre-existing (and potentially compromised) count columns.

### Core Performance Indicators
| Metric | Value |
| :--- | :--- |
| **Total Candidates Evaluated** | {sm['total_candidates_evaluated']} |
| **Perfect Extractions (0 Missed)** | {sm['perfect_extraction_count']} ({sm['perfect_extraction_percentage']}%) |
| **Total Information Loss Counts** | {sm['total_missed_datapoints']} data points |

---

## Loss Distribution Analysis
* **Average Information Loss**: {ed['average_missed_items_per_cv']} items missed per CV.
* **Median Loss Score**: {ed['median_missed_items_per_cv']} items.
* **Worst Case Extraction Failure**: {ed['maximum_missed_items_from_single_cv']} items skipped in a single record.

---

## High Risk Candidate Records (Action Required)
The following candidate records exhibited the highest information loss during processing based on content analysis:
"""
    for cand in report_data["critical_review_candidates"]:
        md_content += f"\n* **Candidate CV ID {cand['cv_id']}** — **{cand['missed_count']} missed items**.\n  * *Preview of missed text:* `{cand['sample_missed_content']}`\n"
        
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Report successfully saved to {output_file}")

# Execute
filename = "parsed_vs_raw_with_missed_column.csv"
metrics_report = evaluate_extraction_performance_updated(filename)
generate_markdown_report_updated(metrics_report)