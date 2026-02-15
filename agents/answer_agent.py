import argparse
import json
from agents.answer_model import AnswerModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        questions = json.load(f)

    model = AnswerModel()
    answers = []

    for q in questions:
        raw_output = model.answer_question(q)
        try:
            start = raw_output.find('{')
            end = raw_output.rfind('}') + 1
            a_json = json.loads(raw_output[start:end])
            answers.append(a_json)
        except:
            answers.append({"answer": "A", "reasoning": "Fallback"})

    with open(args.output_file, "w") as f:
        json.dump(answers, f, indent=4)

if __name__ == "__main__":
    main()
