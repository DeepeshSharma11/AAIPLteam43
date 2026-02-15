
import torch
import json
import re
import time
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class QuestionAgent:
    def __init__(self, model_path):
        print(f"--- Force Loading Mistral Architecture ---")

        # 4-bit config for Mistral-7B
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            local_files_only=True
        )

        # We use AutoModelForCausalLM but force the loading via the snapshot path
        # trust_remote_code=True is essential here to allow the config.json 
        # to define the model architecture manually.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
            revision="main"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _extract_json(self, text):
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(match.group(0)) if match else None
        except:
            return None

    def _get_vote(self, domain):
        prompt = f"<s>[INST] Create a difficult MCQ about {domain} in JSON format. Keys: 'question', 'choices', 'answer'. [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.7, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        clean_text = response.split("[/INST]")[-1].strip()
        return self._extract_json(clean_text)

    def generate_consensus(self, domain, num_votes=3):
        votes = []
        for i in range(num_votes):
            print(f"  - Attempt {i+1}...")
            data = self._get_vote(domain)
            if data: votes.append(data)

        if not votes: return None

        answers = [v.get('answer', 'A') for v in votes]
        winner = Counter(answers).most_common(1)[0][0]
        final_data = next((v for v in votes if v.get('answer') == winner), votes[0])
        return final_data
