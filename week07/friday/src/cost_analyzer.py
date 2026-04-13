from typing import Dict, Any

def calculate_daily_misclassification_cost(
    precision_neg: float, 
    recall_neg: float, 
    total_reviews_per_day: int = 100000,
    neg_review_percentage: float = 0.10
) -> Dict[str, float]:
    """
    Applies a business cost model to classification errors.
    
    Cost Definitions:
    - False Negative (FN): A negative review missed. 
      Cost: Refund + churn risk + support ticket = $50 per occurrence.
    - False Positive (FP): A positive review flagged as negative. 
      Cost: Support staff manual review time = $2 per occurrence.
    """
    cost_fn = 50.0
    cost_fp = 2.0
    
    num_neg_actual = total_reviews_per_day * neg_review_percentage
    num_pos_actual = total_reviews_per_day * (1 - neg_review_percentage)
    
    # FN = Missing actual negatives
    missed_count = num_neg_actual * (1 - recall_neg)
    daily_fn_cost = missed_count * cost_fn
    
    # FP = Wrongly flagging positives
    # Precision = TP / (TP + FP) -> FP = TP*(1-prec)/prec
    # For simplicity, let's assume predicted negatives = num_neg_actual * recall / precision
    # estimated FP = Predicted negatives - True Positives
    predicted_neg = (num_neg_actual * recall_neg) / precision_neg
    false_pos_count = predicted_neg * (1 - precision_neg)
    daily_fp_cost = false_pos_count * cost_fp
    
    return {
        'daily_fn_cost': daily_fn_cost,
        'daily_fp_cost': daily_fp_cost,
        'total_daily_cost': daily_fn_cost + daily_fp_cost,
        'missed_customers_count': missed_count,
        'false_alarm_count': false_pos_count
    }
