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
    batch_size = 3

    for r_idx in range(0, len(dataset), batch_size):
        questions = []
        answers = []
        prompts = []
        for idx in range(r_idx, r_idx+batch_size):
            if idx >= len(dataset):
                break
            questions.append(dataset[r_idx][0])
            answers.append(dataset[idx][1])
            prompt = llm.format_prompt(dataset[r_idx][0])
            prompts.append(prompt)
        # generations = llm.batched_generate(prompts, oversample, temperature)
        generations = [["random <answer> 120.0</answer>"],["random <answer> 7200.0</answer>"],["random <answer> 32.0</answer>"]]
        # print(generations)
        for idx in range(len(generations)):
            parsed_answers = [llm.parse_answer(g) for g in generations[idx]]
            min_diff = -1
            min_diff_idx = -1
            for i in range(len(parsed_answers)):
                valid, diff = is_answer_valid(parsed_answers[i], answers[idx])
                if valid and (diff < min_diff or min_diff == -1):
                    min_diff = diff
                    min_diff_idx = i
            
            if min_diff_idx != -1:
                # write this data to file
                data.append([questions[idx], parsed_answers[min_diff_idx], generations[idx][min_diff_idx]])
                count += 1

        print("count:%d, total:%d" % (count, r_idx+batch_size))

    with open(output_json, 'w') as file:
        json.dump(data, file, indent=4)
        
if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
