from homework import Dataset
from .base_llm import BaseLLM
from .sft import test_model, TokenizedDataset, format_example


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    from peft import get_peft_model, LoraConfig, PeftModel
    from transformers import Trainer, TrainingArguments

    llm = BaseLLM()
    llm.model = get_peft_model(llm.model, LoraConfig(**kwargs))
    llm.model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        gradient_checkpointing=True,
        learning_rate=5e-4,
        num_train_epochs=5,
        per_device_train_batch_size=32
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=TokenizedDataset(llm.tokenizer, Dataset("train"), format_example),
        eval_dataset=TokenizedDataset(llm.tokenizer, Dataset("valid"), format_example),
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
