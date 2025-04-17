from .data import Dataset
from .cot import CoTModel
import json

def is_answer_valid(answer: float, correct_answer: float, relative_tolerance: float = 0.05):
    abs_diff = abs(round(answer, 3) - round(correct_answer, 3))
    valid = abs_diff < relative_tolerance * abs(round(correct_answer, 3))
    return valid, abs_diff


def generate_dataset(output_json: str = 'data/rft.json', oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()
    dataset = Dataset("train")
    idx = range(len(dataset))

    llm = CoTModel()
    data = []
    count = 0
    for idx in range(len(dataset)):
        question = dataset[idx][0]
        answer = dataset[idx][1]
        prompt = llm.format_prompt(question)
        generations = llm.batched_generate([prompt], oversample, temperature)
        print(generations)
        parsed_answers = [llm.parse_answer(g) for g in generations[0]]
        min_diff = -1
        min_diff_idx = -1
        for i in range(len(parsed_answers)):
            valid, diff = is_answer_valid(parsed_answers[i], answer)
            if valid and (diff < min_diff or min_diff == -1):
                min_diff = diff
                min_diff_idx = i
        
        if min_diff_idx != -1:
            # write this data to file
            data.append([question, parsed_answers[min_diff_idx], generations[min_diff_idx]])
            count += 1
            print("found an entry, count:", count)

    with open(output_json, 'w') as file:
        json.dump(data, file, indent=4)
        
if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
