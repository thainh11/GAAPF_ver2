import requests
import time

# 3 prompts để test
prompts = [
    "The benefit of LangChain in 5 bullet points",
    "How to create a custom chain in LangChain with error handling?",
    "List out some of LangChain use cases"
]

print("🧪 Testing RLHF System with 3 prompts")
print("="*50)

for i, prompt in enumerate(prompts, 1):
    print(f"\n--- Request {i}/3 ---")
    print(f"Prompt: {prompt}")
    
    try:
        res = requests.post(
            "http://localhost:8000/learn",
            json={"prompt": prompt}
        )
        
        print("Status:", res.status_code)
        
        if res.status_code == 200:
            data = res.json()
            # Show key info
            score = data['mentor_evaluation']['composite_score']
            policy = data['response_metadata']['chosen_policy']
            print(f"Score: {score:.2f} | Policy: {policy}")
            
            # Show answer preview
            answer = data['answer']
            preview = answer[:100] + "..." if len(answer) > 100 else answer
            print(f"Answer: {preview}")
        else:
            print("Error:", res.text)
            
    except Exception as e:
        print(f"Request failed: {e}")
    
    # Wait between requests (except last one)
    if i < len(prompts):
        print("⏳ Waiting 2s...")
        time.sleep(2)

print(f"\n✅ Completed {len(prompts)} requests!")
print("💡 Check server logs or /status endpoint to see policy updates")