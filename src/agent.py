from pydantic import BaseModel, Field
from enum import Enum
from google import genai
from google.genai import types
import os

class IntentEnum(str, Enum):
    SOFTWARE_UPDATE = "Software/OS Update"
    HARDWARE_ISSUE = "Hardware/Device Issue"
    ACCOUNT_BILLING = "Account & Billing"
    GENERAL = "General Inquiry"

class ActionEnum(str, Enum):
    AUTO_HANDLE = "Auto-handle"
    ESCALATE = "Escalate"

class AgentDecision(BaseModel):
    intent: IntentEnum = Field(description="The classified intent of the customer message.")
    action: ActionEnum = Field(description="Whether the issue should be Auto-handled or Escalated. Hardware issues, billing disputes, or very angry customers should be escalated.")
    reason: str = Field(description="Reason for the action decision.")
    draft_reply: str = Field(description="A polite, helpful draft reply grounded in typical Apple Support responses.")

class SupportAgent:
    def __init__(self, use_mock=False):
        self.use_mock = use_mock
        if not self.use_mock:
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            
    def process_message(self, customer_message: str) -> AgentDecision:
        if self.use_mock:
            # Fallback mock logic when no API key is provided
            c_lower = customer_message.lower()
            if any(w in c_lower for w in ['update', 'ios', 'download', 'firmware']):
                intent = IntentEnum.SOFTWARE_UPDATE
            elif any(w in c_lower for w in ['screen', 'battery', 'charge', 'broken', 'button']):
                intent = IntentEnum.HARDWARE_ISSUE
            elif any(w in c_lower for w in ['bill', 'pay', 'subscription', 'store']):
                intent = IntentEnum.ACCOUNT_BILLING
            else:
                intent = IntentEnum.GENERAL
                
            is_escalate = (intent == IntentEnum.HARDWARE_ISSUE) or (len(customer_message) > 250) or any(w in c_lower for w in ['wtf', 'hate'])
            action = ActionEnum.ESCALATE if is_escalate else ActionEnum.AUTO_HANDLE
            
            return AgentDecision(
                intent=intent,
                action=action,
                reason="Mock reasoning generated.",
                draft_reply="This is a mock draft reply for: " + customer_message[:20] + "..."
            )
            
        sys_prompt = """You are an AppleSupport AI Agent. 
        You analyze incoming customer messages and provide:
        1. An intent classification.
        2. A drafted reply (helpful, Apple-style tone, providing basic troubleshooting links if auto-handling).
        3. A decision to Auto-handle or Escalate, with a reason. 
        Escalate issues that require a physical visit (like broken screens/hardware), complex billing disputes, or highly angry customers.
        Auto-handle basic software/iOS questions, how-tos, and status inquiries."""
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    sys_prompt,
                    f"Customer Message: {customer_message}"
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AgentDecision,
                )
            )
            return response.parsed
        except Exception as e:
            print(f"Agent inference failed (check GEMINI_API_KEY). Falling back to mock. Err: {e}")
            self.use_mock = True
            return self.process_message(customer_message)

if __name__ == "__main__":
    agent = SupportAgent(use_mock=True)
    res = agent.process_message("My iPhone screen is cracked and won't turn on.")
    print(res.model_dump_json(indent=2))
