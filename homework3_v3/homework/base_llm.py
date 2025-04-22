from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# For part 1-3
checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
# For part 4
# checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device
        self.tokenizer.padding_side = "left"

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        return self.batched_generate([prompt])[0]

        # tokenizer.encode returns the tokens only of the input like 
        # inputs=tensor([[22007,  6463,   314]], device='cuda:0')
        # on the other hand tokenizer captues both tokens and attention mask
        # inputs={'input_ids': tensor([[22007,  6463,   314]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1]], device='cuda:0')}

        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # outputs = self.model.generate(**inputs)
        # return self.tokenizer.decode(outputs[0])


    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.

        You will likely get an up to 10x speedup using batched decoding.

        To implement batch decoding you will need to:
        - tokenize the prompts self.tokenizer with padding=True and return_tensors="pt"
        - call self.model.generate
        - decode the outputs with self.tokenizer.batch_decode

        Tip: You need to set self.tokenizer.padding_side = "left" to get the correct padding behavior for generation.
             Left padding makes sure all sequences are aligned to the right (i.e. where tokens are generated).
        Tip: self.model.generate takes a lot of parameters. Here are some relevant ones:
            - max_new_tokens: The maximum number of tokens to generate. Set this to a reasonable value
                              (50 should suffice).
            - do_sample and temperature: For any temperature > 0, set do_sample=True.
                                         do_sample=False will use greedy decoding.
            - num_return_sequences: The number of sequences to return. Note that this will generate a flat
                                    list of len(prompts) * num_return_sequences entries.
            - eos_token_id: The end of sequence token id. This is used to stop generation. Set this
                            to self.tokenizer.eos_token_id.
        Pro Tip: Only batch_decode generated tokens by masking out the inputs with
                 outputs[:, len(inputs["input_ids"][0]) :]
        """
        from tqdm import tqdm  # Importing tqdm for progress bar

        # Preventing OOM
        # Depending on your GPU batched generation will use a lot of memory.
        # If you run out of memory, try to reduce the micro_batch_size.
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                )
                for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
            ]

        # Generation parameters
        do_sample = temperature > 0
        max_new_tokens = 100
        num_sequences = num_return_sequences if num_return_sequences is not None else 1
        
        # tokenize the input. All inputs after tokenization will have the same len which will be 
        # max of all the inputs. The smaller one would be left-padded.
        # truncation ensures the maximum length of a input doesn't exceede the model max accepted value

        # FROM DOCS AT https://huggingface.co/docs/transformers/v4.35.2/en/llm_tutorial#generate-text
        padding = len(prompts) > 1
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=padding, truncation=True).to(self.device)
        
        # Generate next predictions
        outputs = self.model.generate(**inputs, 
                                      max_new_tokens=max_new_tokens, 
                                      do_sample=do_sample, 
                                      temperature=temperature,
                                      num_return_sequences=num_sequences,
                                      eos_token_id = self.tokenizer.eos_token_id
                                      )
        
        # Only decode the token which were predicted(no point of decoding the original input)
        decoded_outputs = self.tokenizer.batch_decode(outputs[:, len(inputs["input_ids"][0]) :], skip_special_tokens=True)
        # If num_sequences is None or 1, return flat list
        if num_sequences == 1:
            return decoded_outputs
        
        # since the decoded_outputs is a flat list of len(inputs)*num_sequences
        # group all outputs of same input into a sublist
        answer = []
        for i in range(0, len(decoded_outputs), num_sequences):
            answer.append(decoded_outputs[i:i + num_sequences])
        return answer

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        print(f"{generations=}")
        parsed_answers = [self.parse_answer(g) for g in generations]
        print(f"{parsed_answers=}")
        return parsed_answers
            


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
