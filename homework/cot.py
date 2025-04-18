from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        # TODO:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "How many grams are in 2 kilograms?"},
            {"role": "assistant", "content": "1 kilogram = 1000 grams. 2 * 1000 = <answer>2000</answer>."},
            {"role": "user", "content": "How many grams are in 3 kilograms?"},
            {"role": "assistant", "content": "1 kilogram = 1000 grams. 3 * 1000 = <answer>3000</answer>."},
            {"role": "user", "content": "How many grams are in 4.5 kilograms?"},
            {"role": "assistant", "content": "1 kilogram = 1000 grams. 4.5 * 1000 = <answer>4500</answer>."},
            {"role": "user", "content": "How many grams are in 0.75 kilograms?"},
            {"role": "assistant", "content": "1 kilogram = 1000 grams. 0.75 * 1000 = <answer>750</answer>."},
            {"role": "user", "content": question},
        ]

        # TODO: Implement this method
        formatted_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # print(formatted_prompt)
        return formatted_prompt


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
