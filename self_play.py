import time
import json
from agents.question_model import QuestionModel
from agents.answer_model import AnswerModel

class SelfPlayArena:
    def __init__(self, rounds=10):
        self.rounds = rounds
        self.q_model = QuestionModel()
        self.a_model = AnswerModel()
        self.stats = {
            "total": 0,
            "correct": 0,
            "wrong": 0,
            "avg_confidence": 0
        }

    def run(self):
        print("🔥 Starting Self-Play Training...\n")

        total_conf = 0

        for i in range(self.rounds):
            print(f"Round {i+1}/{self.rounds}")

            question = self.q_model.generate_question("logical reasoning")
            answer = self.a_model.solve(question["question"], question["choices"])

            correct_letter = question["answer"]
            predicted_letter = answer["answer"]
            confidence = answer.get("confidence", 0)

            self.stats["total"] += 1
            total_conf += confidence

            if predicted_letter == correct_letter:
                self.stats["correct"] += 1
                print("✅ Correct")
            else:
                self.stats["wrong"] += 1
                print("❌ Wrong")

            print("Correct:", correct_letter)
            print("Predicted:", predicted_letter)
            print("Confidence:", confidence)
            print("-"*40)

        self.stats["avg_confidence"] = total_conf / self.rounds

        return self.stats


if __name__ == "__main__":
    arena = SelfPlayArena(rounds=5)
    results = arena.run()

    print("\n🏆 Self-Play Results")
    print(results)
