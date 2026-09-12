import os
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

class ReplyGrade(BaseModel):
    score: int = Field(ge=1, le=5, description="Grade from 1 to 5. 5 is excellent, 1 is terrible.")
    reasoning: str = Field(description="Why this score was given, focusing on tone, helpfulness, and similarity to the ground-truth brand reply.")

class Evaluator:
    def __init__(self, use_mock=False):
        self.use_mock = use_mock
        if not self.use_mock:
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            
    def grade_reply(self, customer_message: str, generated_reply: str, ground_truth_reply: str) -> ReplyGrade:
        if self.use_mock:
            return ReplyGrade(score=3, reasoning="Mock evaluation score.")
            
        sys_prompt = """You are an expert customer service evaluator.
        You will see a Customer Message, a Ground-Truth reply from a human agent, and a Generated reply from an AI.
        Grade the AI's Generated reply on a scale of 1 to 5.
        Criteria:
        - 5: Excellent. Captures the same helpfulness, correct tone, and similar directions as ground-truth.
        - 3: Acceptable but missing some nuance or slightly off-tone compared to ground truth.
        - 1: Harmful, irrelevant, or hallucinates bad instructions."""
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    sys_prompt,
                    f"Customer Message: {customer_message}\nGround-Truth Reply: {ground_truth_reply}\nGenerated Reply: {generated_reply}"
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ReplyGrade,
                )
            )
            return response.parsed
        except Exception as e:
            print(f"Evaluator inference failed (check GEMINI_API_KEY). Falling back to mock. Err: {e}")
            self.use_mock = True
            return self.grade_reply(customer_message, generated_reply, ground_truth_reply)

    def compute_metrics(self, df_results: pd.DataFrame):
        # df_results needs: true_intent, pred_intent, true_action, pred_action, reply_score
        
        y_true_intent = df_results['true_intent'].tolist()
        y_pred_intent = df_results['pred_intent'].tolist()
        
        y_true_action = [1 if a == 'Escalate' else 0 for a in df_results['true_action']]
        y_pred_action = [1 if a == 'Escalate' else 0 for a in df_results['pred_action']]
        
        metrics = {
            "intent_accuracy": accuracy_score(y_true_intent, y_pred_intent),
            "intent_macro_f1": f1_score(y_true_intent, y_pred_intent, average='macro'),
            "escalation_precision": precision_score(y_true_action, y_pred_action, zero_division=0),
            "escalation_recall": recall_score(y_true_action, y_pred_action, zero_division=0),
            "average_reply_score": df_results['reply_score'].mean()
        }
        return metrics

if __name__ == "__main__":
    evaluator = Evaluator(use_mock=True)
    grade = evaluator.grade_reply("Help me", "Restart phone", "Have you tried turning it off and on?")
    print(grade.model_dump_json(indent=2))
