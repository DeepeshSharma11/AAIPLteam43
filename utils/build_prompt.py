def build_q_prompt(topic):
    return f"Task: Generate a logic puzzle on {topic}. Format: JSON. Fields: topic, question, choices, answer, explanation."

def build_a_prompt(question, choices):
    return f"Task: Solve this puzzle. Question: {question} Choices: {choices}. Format: JSON. Fields: answer, reasoning."
