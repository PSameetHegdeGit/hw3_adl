from .cot import *
from .base_llm import *
from .data import *
from .sft import *

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    cot = CoTModel()

    # load the dataset
    dataset = Dataset("train")
    output_data = []

    for question, correct_answer in dataset:
        completions = cot.batched_generate(
            [cot.format_prompt(question)] * oversample,
            num_return_sequences=oversample,
            temperature=temperature,
        )

        for completion in completions:
            answer = cot.parse_answer(completion)
            if is_answer_valid(answer, correct_answer):
                output_data.append(
                    [
                        question,
                        answer,
                        completion
                    ]
                )

    # store in output_json
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=4)



if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
