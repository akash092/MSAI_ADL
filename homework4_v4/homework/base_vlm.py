from pathlib import Path

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.image_utils import load_image

from .data import VQADataset, benchmark

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseVLM:
    def __init__(self, checkpoint="HuggingFaceTB/SmolVLM-256M-Instruct"):
        self.processor = AutoProcessor.from_pretrained(checkpoint)

        # important to set this to False, otherwise too many image tokens
        self.processor.image_processor.do_image_splitting = False

        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager",
        ).to(DEVICE)
        self.device = DEVICE

    def format_prompt(self, question: str) -> str:
        """
        Format the question into a prompt for the VLM.
        """
        return question

    def generate(self, image_path: str, question: str) -> str:
        """
        Generate a response to a question about an image.

        Args:
            image_path: Path to the image file
            question: Question about the image

        Returns:
            Generated text response
        """
        return self.batched_generate([image_path], [question])[0]

    def batched_generate(
        self,
        image_paths: list[str],
        questions: list[str],
        num_return_sequences: int | None = None,
        temperature: float = 0,
    ) -> list[str] | list[list[str]]:
        """
        Batched version of generate method.

        Args:
            image_paths: List of paths to image files
            questions: List of questions about the images
            num_return_sequences: Number of sequences to return per input
            temperature: Temperature for sampling

        Returns:
            List of generated text responses
        """
        # Load images
        images = [load_image(img_path) for img_path in image_paths]

        # Create input messages with proper image tokens
        messages = []
        for q in questions:
            # Create a message with an image token placeholder
            message = {
                "role": "user",
                "content": [
                    {"type": "image"},  # Correct type to insert image token
                    {"type": "text", "text": self.format_prompt(q)},
                ],
            }
            messages.append([message])

        # Prepare inputs
        prompts = [self.processor.apply_chat_template(message, add_generation_prompt=True) for message in messages]
        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt", padding=True, truncation=True, padding_side="left"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Set generation parameters
        generate_params = {
            "max_new_tokens": 32,
            "do_sample": temperature > 0,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }

        if temperature > 0:
            generate_params["temperature"] = temperature

        if num_return_sequences is not None:
            generate_params["num_return_sequences"] = num_return_sequences

        # Generate outputs
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_params)

        # Decode outputs
        generated_texts = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        # Extract only the assistant's answer
        cleaned_texts = []
        for text in generated_texts:
            # Find the last occurrence of "Assistant:" and take everything after
            if "Assistant:" in text:
                cleaned_texts.append(text.split("Assistant:")[-1].strip())
            else:
                cleaned_texts.append(text.strip())

        # Handle multiple return sequences
        if num_return_sequences is not None:
            return [
                cleaned_texts[i : i + num_return_sequences] for i in range(0, len(cleaned_texts), num_return_sequences)
            ]

        return cleaned_texts

    def answer(self, image_paths, questions) -> list[str]:
        """
        Answer multiple questions about an image.

        Args:
            *image_paths: Paths to the image files
            *questions: Questions about the image

        Returns:
            List of answers
        """
        return self.batched_generate(image_paths, questions)


def test_model():
    # Test the BaseVLM with a sample image and question
    model = BaseVLM()

    # Use a sample image from the internet
    current_dir = Path(__file__).parent
    image_path_1 = str((current_dir / "../data/train/00000_00_im.jpg").resolve())
    image_path_2 = str((current_dir / "../data/train/00000_01_im.jpg").resolve())

    # Test multiple questions
    questions = ["What is in the center of this image?", "What track is this?"]
    answers = model.answer([image_path_1, image_path_2], questions)
    print("\nTesting multiple questions:")
    for q, a in zip(questions, answers):
        print(f"Q: {q}")
        print(f"A: {a}")


def test_benchmark():
    model = BaseVLM()
    dataset = VQADataset("valid")
    result = benchmark(model, dataset, 8)
    print(result.accuracy)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "benchmark": test_benchmark})
