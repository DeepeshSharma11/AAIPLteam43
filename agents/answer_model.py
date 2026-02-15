import time
import torch
import json
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "hf_models/Llama-3.1-8B-Instruct"

class AnswerModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )

    def _generate_once(self, prompt, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=temperature,
            do_sample=True if temperature > 0 else False
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _extract_answer(self, text):
        for letter in ["A", "B", "C", "D"]:
            if f'"answer": "{letter}"' in text or f"Answer: {letter}" in text:
                return letter
        return None

    def solve(self, question_text, choices):
        start_time = time.time()
        TIME_LIMIT = 8.5  # keep safe under 9 sec limit

        prompt = f"""
You are a highly logical AI.
Solve step-by-step internally.
Return ONLY valid JSON in this format:
{{
    "answer": "A/B/C/D",
    "confidence": float between 0 and 1
}}

Question:
{question_text}

Choices:
A: {choices.get("A")}
B: {choices.get("B")}
C: {choices.get("C")}
D: {choices.get("D")}
"""

        votes = []

        # -------- Initial Multi-Vote (3 votes) --------
        for _ in range(3):
            if time.time() - start_time > TIME_LIMIT:
                break
            response = self._generate_once(prompt, temperature=0.7)
            ans = self._extract_answer(response)
            if ans:
                votes.append(ans)

        if not votes:
            return {"answer": "A", "confidence": 0.25}

        counter = Counter(votes)
        best_answer, count = counter.most_common(1)[0]
        confidence = count / len(votes)

        # -------- Adaptive Voting --------
        if confidence < 0.7 and time.time() - start_time < TIME_LIMIT:
            for _ in range(2):  # extra votes
                if time.time() - start_time > TIME_LIMIT:
                    break
                response = self._generate_once(prompt, temperature=0.7)
                ans = self._extract_answer(response)
                if ans:
                    votes.append(ans)

            counter = Counter(votes)
            best_answer, count = counter.most_common(1)[0]
            confidence = count / len(votes)

        # -------- Deterministic Fallback --------
        if confidence < 0.6 and time.time() - start_time < TIME_LIMIT:
            response = self._generate_once(prompt, temperature=0.0)
            ans = self._extract_answer(response)
            if ans:
                best_answer = ans
                confidence = max(confidence, 0.75)

        # -------- Final Safe Return --------
        return {
            "answer": best_answer,
            "confidence": round(float(confidence), 2)
        }
