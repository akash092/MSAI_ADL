from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        #raise NotImplementedError()
        chat = [
            {"role": "system", "content": "Perform Unit Conversion."},
            {"role": "system", "content": "Place the final numerical value between an answer tag"},
            {"role": "user", "content": "How many gram are there per 6 kg?"},
            {"role": "assistant", "content": "1 kg = 1000 grams. 6 kg = 6 * 1000 = <answer>6000.0</answer>"},
            {"role": "user", "content": "Can you change 6 miles to its equivalent in m?"},
            {"role": "assistant", "content": "1 mile = 1609.34 m. 6 miles = 6 * 1609.34 = <answer>9656.04/answer>"},
            {"role": "user", "content": "What is the measurement of 3 kg when converted into pound?"},
            {"role": "assistant", "content": "1 kg = 2.2046226218 pounds. 3 kg = 3 * 2.2046226218 = <answer>6.6138678654</answer>"},
            {"role": "user", "content": question}
        ]

        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
