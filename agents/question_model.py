import torch
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "hf_models/Llama-3.1-8B-Instruct"

class QuestionModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )

    def _generate(self, prompt, temperature=0.8):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=temperature,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_question(self, domain="general knowledge"):
        TIME_LIMIT = 9
        start_time = time.time()

        base_prompt = f"""
You are an expert exam creator.

Create ONE very challenging but valid multiple choice question in the domain: {domain}

Rules:
- 4 options (A,B,C,D)
- Exactly ONE correct answer
- No ambiguity
- Avoid trick wording
- Make distractors very strong
- Question should require reasoning

Return ONLY valid JSON:

{{
  "question": "...",
  "choices": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
  }},
  "answer": "A/B/C/D",
  "difficulty": float between 0 and 1
}}
"""

        response = self._generate(base_prompt)

        try:
            data = json.loads(response)
        except:
            # Retry once if invalid
            response = self._generate(base_prompt)
            try:
                data = json.loads(response)
            except:
                return {
                    "question": "Which planet is known as the Red Planet?",
                    "choices": {
                        "A": "Earth",
                        "B": "Mars",
                        "C": "Venus",
                        "D": "Jupiter"
                    },
                    "answer": "B",
                    "difficulty": 0.3
                }

        # Basic validation
        if "question" not in data or "choices" not in data or "answer" not in data:
            return {
                "question": "Which planet is known as the Red Planet?",
                "choices": {
                    "A": "Earth",
                    "B": "Mars",
                    "C": "Venus",
                    "D": "Jupiter"
                },
                "answer": "B",
                "difficulty": 0.3
            }

        if data["answer"] not in ["A","B","C","D"]:
            data["answer"] = "A"

        if time.time() - start_time > TIME_LIMIT:
            data["difficulty"] = min(data.get("difficulty", 0.5), 0.7)

        return data
