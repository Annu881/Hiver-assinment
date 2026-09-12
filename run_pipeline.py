import pandas as pd
from tqdm import tqdm
import argparse
from src.agent import SupportAgent, ActionEnum, IntentEnum
from src.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Run on a small subset of 5 examples.")
    parser.add_argument("--use-mock", action="store_true", help="Use mock models instead of OpenAI API.")
    args = parser.parse_args()
    
    agent = SupportAgent(use_mock=args.use_mock)
    evaluator = Evaluator(use_mock=args.use_mock)
    
    print("Loading golden set...")
    df = pd.read_csv("golden_set.csv")
    if args.test_mode:
        df = df.head(5)
        
    results = []
    
    print("Running pipeline...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        customer_msg = row['customer_message']
        ground_truth_reply = row['actual_brand_reply']
        
        # 1. Agent processes message
        decision = agent.process_message(customer_msg)
        
        # 2. Evaluator grades drafted reply
        grade = evaluator.grade_reply(
            customer_message=customer_msg,
            generated_reply=decision.draft_reply,
            ground_truth_reply=ground_truth_reply
        )
        
        results.append({
            "conversation_id": row['conversation_id'],
            "customer_message": customer_msg,
            "true_intent": row['true_intent'],
            "pred_intent": decision.intent.value,
            "true_action": row['true_action'],
            "pred_action": decision.action.value,
            "actual_brand_reply": ground_truth_reply,
            "draft_reply": decision.draft_reply,
            "reply_score": grade.score,
            "reply_reasoning": grade.reasoning,
            "agent_reasoning": decision.reason
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv("results.csv", index=False)
    
    metrics = evaluator.compute_metrics(df_results)
    
    print("\n" + "="*40)
    print("FINAL EVALUATION METRICS:")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("="*40)
    
    # Simple baselines
    print("\nBASELINES:")
    # Majority Class Baseline for Intent (usually 'Hardware/Device Issue' in our heuristic rules if most trigger)
    majority_intent = df['true_intent'].mode()[0]
    majority_acc = (df['true_intent'] == majority_intent).mean()
    print(f"Trivial Baseline (Majority Intent Accuracy - '{majority_intent}'): {majority_acc:.4f}")
    
    # Always Auto-handle baseline
    always_auto_prec = 0.0 # because no escalations predicted
    always_auto_recall = 0.0
    print(f"Trivial Baseline (Always Auto-handle - Escalation Recall): {always_auto_recall:.4f}")

if __name__ == "__main__":
    main()
