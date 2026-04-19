import os
from dotenv import load_dotenv
import json
import base64
import anthropic
from google import genai  
from google.genai import types
import time
import random

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client_anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
client_google = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def encode_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_claude(system_prompt, user_prompt, attachments):
    content = []
    
    # Add Attachments
    for file_name in attachments:
        file_path = os.path.join("data", "files", file_name)
        ext = file_name.split('.')[-1].lower()
        
        if ext in ['jpg', 'jpeg', 'png']:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": f"image/{ext}", "data": encode_file(file_path)}
            })
        elif ext == 'pdf':
            content.append({
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": encode_file(file_path)}
            })
            
    content.append({"type": "text", "text": user_prompt})
    
    response = client_anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": content}]
    )
    return response.content[0].text

def call_gemini_with_retry(system_prompt, user_prompt, attachments, retries=3):
    for i in range(retries):
        try:
            return call_gemini(system_prompt, user_prompt, attachments)
        except Exception as e:
            if "503" in str(e) or "high demand" in str(e).lower():
                wait_time = (2 ** i) + random.random() 
                print(f"Server busy. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
                continue
            raise e 
    return "Error: Model remained unavailable after multiple retries."

def call_gemini(system_prompt, user_prompt, attachments):
    prompt_parts = [user_prompt]
    
    for file_name in attachments:
        file_path = os.path.join("data", "files", file_name)
        file_upload = client_google.files.upload(file=file_path)
        prompt_parts.append(file_upload)
    
    response = client_google.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1 
        ),
        contents=prompt_parts
    )
    return response.text
def run_benchmark():
    with open("data/benchmark.json", "r") as f:
        tests = json.load(f)
    
    results = []
    
    for test in tests:
        print(f"Running {test['id']}...")
        
        # Anthropic Run
        """try:
            claude_res = call_claude(test['system_prompt'], test['user_prompt'], test['attachments'])
        except Exception as e:
            claude_res = f"Error: {e}"
        """
        # Google Run
        try:
            gemini_res = call_gemini_with_retry(test['system_prompt'], test['user_prompt'], test['attachments'])
        except Exception as e:
            gemini_res = f"Error: {e}"
            
        results.append({
            "id": test['id'],
            "bucket": test['bucket'],
            "expected": test['expected'],
            "claude_output": claude_res,
            "gemini_output": gemini_res,
            "claude_correct":"",
            "gemini_correct":""
        })
        time.sleep(12)

        
    with open("results/raw_outputs.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Benchmark complete! Results saved to results/raw_outputs.json")

if __name__ == "__main__":
    run_benchmark()