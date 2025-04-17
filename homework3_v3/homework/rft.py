from .base_llm import BaseLLM
from .sft import test_model, TokenizedDataset
from .data import Dataset
from transformers import Trainer, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model
from pathlib import Path

def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def format_example(prompt: str, answer: str, reasoning: str) -> dict[str, str]:
    return {"question": prompt, "answer": reasoning}


def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    #raise NotImplementedError()

    llm = BaseLLM()
    peft_config = LoraConfig(
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM", 
        inference_mode=False, 
        r=16, 
        lora_alpha=16*4, 
        lora_dropout=0.1
    )

    model = get_peft_model(llm.model, peft_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        gradient_checkpointing=True,
        learning_rate=1e-3,
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=32,
        num_train_epochs=5,
        # weight_decay=0.01,
        # eval_strategy="epoch",
        # save_strategy="epoch",
        # load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenizedDataset(llm.tokenizer, Dataset("rft"), format_example)
    )

    trainer.train()
    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name
    model.save_pretrained(model_path)



if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
