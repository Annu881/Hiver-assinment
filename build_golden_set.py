import pandas as pd
import json
import re

def build_golden_set():
    print("Loading apple_support_sample.json...")
    df = pd.read_json('apple_support_sample.json')
    
    # We want exactly 150 rows.
    # The dataset currently has 250 rows.
    
    golden_records = []
    
    for idx, row in df.iterrows():
        conv_text = row['conversation']
        if not isinstance(conv_text, str):
            continue
            
        # Parse logic:
        # Example: 
        # Customer: msg1\nSupport: msg2\nCustomer: msg3
        # We need the first Customer message and the first Support message.
        
        parts = re.split(r'(Customer:|Support:)', conv_text)
        
        first_customer = None
        first_support = None
        
        current_speaker = None
        current_msg = []
        
        for p in parts:
            p = p.strip()
            if p == 'Customer:':
                if current_speaker == 'Customer' and current_msg:
                    # Append to previous, maybe multi-line
                    pass
                elif current_speaker == 'Support' and first_support is None:
                    first_support = " ".join(current_msg).strip()
                current_speaker = 'Customer'
                current_msg = []
            elif p == 'Support:':
                if current_speaker == 'Customer' and first_customer is None:
                    first_customer = " ".join(current_msg).strip()
                current_speaker = 'Support'
                current_msg = []
            else:
                if p:
                    current_msg.append(p)
                    
        # Grab final message if it was the last part
        if current_speaker == 'Customer' and first_customer is None:
            first_customer = " ".join(current_msg).strip()
        if current_speaker == 'Support' and first_support is None:
            first_support = " ".join(current_msg).strip()
            
        if not first_customer or not first_support:
            continue
            
        # Heuristic Labeling (acting as human annotator)
        c_lower = first_customer.lower()
        
        if any(w in c_lower for w in ['update', 'ios', 'download', 'firmware']):
            intent = 'Software/OS Update'
        elif any(w in c_lower for w in ['screen', 'battery', 'charge', 'broken', 'button', 'glass']):
            intent = 'Hardware/Device Issue'
        elif any(w in c_lower for w in ['bill', 'pay', 'subscription', 'store', 'money', 'charge']):
            intent = 'Account & Billing'
        else:
            intent = 'General Inquiry'
            
        is_escalate = False
        reason = "Standard issue, can link to docs."
        if intent == 'Hardware/Device Issue':
            is_escalate = True
            reason = "Hardware issue likely needs Genius Bar appointment."
        elif any(w in c_lower for w in ['wtf', 'ridiculous', 'hate', 'terrible']):
            is_escalate = True
            reason = "Customer expressing high frustration."
        elif len(c_lower) > 250:
            is_escalate = True
            reason = "Very long complex message, requires human reading."
            
        action = "Escalate" if is_escalate else "Auto-handle"
        
        golden_records.append({
            'conversation_id': row['conversation_id'],
            'customer_message': first_customer,
            'true_intent': intent,
            'true_action': action,
            'true_reason': reason,
            'actual_brand_reply': first_support
        })
        
        if len(golden_records) == 150:
            break
            
    golden_df = pd.DataFrame(golden_records)
    golden_df.to_csv('golden_set.csv', index=False)
    print(f"Created golden set with {len(golden_df)} examples.")
    
if __name__ == '__main__':
    build_golden_set()
