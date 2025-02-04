import os
from collections import Counter
from openai import OpenAI
import json
from dataclasses import dataclass, asdict
from typing import List
from textwrap import dedent
import string
import random
import time

from word_validator import WordValidator

local_models = [
    # "llama-3.1-supernova-lite-l4", # poor results
    # "deepseek-r1-distill-llama-8b-l4", # no results
    # "phi-4-fp8-l4", # broken
    # "phi-4-ollama-l4", # best slm
    # "phi-4-bnb-4bit-l4", poor responses
    # "phi-4-l4",
    "granite-3.1-dense-ollama-l4",
]


# model = "gpt-4o-mini"
together_ai_models = [
    # "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", too bad..
    # "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", # best model so far
    # "Qwen/Qwen2.5-7B-Instruct-Turbo",
    # "Qwen/Qwen2.5-72B-Instruct-Turbo",
    # "deepseek-ai/DeepSeek-V3",  # best oss model so far
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  # doesn't work because it thinks as response
    # "mistralai/Mistral-7B-Instruct-v0.2",  # decent small oss model so far
    # "google/gemma-2-9b-it", # error from togetherai
    # "Gryphe/MythoMax-L2-13b",
    # "mistralai/Mixtral-8x22B-Instruct-v0.1",
]


word_validator = WordValidator()


@dataclass
class Evaluation:
    evaluator_name: str
    model: str
    letters: List[str]
    model_reply: str
    valid_words: List[str]
    score: int

    def to_dict(self):
        return asdict(self)


class Evaluator:
    def __init__(self, model: str):
        self.word_validator = WordValidator()
        self.client = OpenAI()
        together_api_key = os.environ.get("TOGETHER_API_KEY")
        if together_api_key and model in together_ai_models:
            self.client = OpenAI(
                api_key=together_api_key,
                base_url="https://api.together.xyz/v1",
            )
        if model in local_models:
            self.client = OpenAI(
                base_url="http://localhost:8000/openai/v1",
                api_key="ignored",
            )
        self.model = model

    def get_prompt(self, letters: List[str], max_words: int = 20) -> str:
        raise NotImplementedError

    def model_response_to_words(self, response: str) -> List[str]:
        raise NotImplementedError

    def evaluate_model_on_letters(self, letters: List[str], max_words=20) -> Evaluation:
        """
        1. Prompts the model with the given letters.
        2. Parses the response.
        3. Validates the words.
        4. Computes a simple score.
        """
        try:
            prompt = self.get_prompt(letters, max_words=max_words)

            # Added retry logic for 429 errors
            max_retries = 10  # Number of retry attempts
            retry_delay = 10  # Initial delay in seconds

            for attempt in range(max_retries + 1):
                try:
                    start_time = time.time()  # Start timing before API call
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": prompt.strip()},
                        ],
                        temperature=0.0,
                        timeout=180.0,
                        max_tokens=5000,
                    )
                    end_time = time.time()  # Record time after API call
                    print(f"Model response time: {end_time - start_time:.2f} seconds")
                    break
                except Exception as e:
                    if attempt < max_retries and "429" in str(e):
                        print(f"Rate limit exceeded (429). Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise  # Re-raise the exception if not recoverable

            # Extract the model's reply
            model_reply = response.choices[0].message.content

            print("Raw model reply:")
            print(model_reply)
            if not model_reply:
                raise ValueError("No response from the model")

            # Attempt to parse the reply as JSON or fallback if format differs
            words = self.model_response_to_words(model_reply)

            # Keep only unique words (case-insensitive uniqueness)
            unique_words = []
            lower_seen = set()
            for w in words:
                lw = w.lower().strip()
                if lw not in lower_seen:
                    lower_seen.add(lw)
                    unique_words.append(w.strip())

            # Validate each word (only checking letter usage here)
            valid_words = []
            for w in unique_words:
                if can_form_word_from_letters(w, letters):
                    # also check if it's a real English word using an external dictionary:
                    if word_validator.is_real_english_word(w):
                        valid_words.append(w)
                    else:
                        print(f"Invalid English word (not in dictionary): {w}")
                else:
                    print(f"Invalid word (can't be formed from letters): {w}")

            # Compute scoring: +1 for each valid unique word
            # If you want more nuanced scoring, you can adjust this logic
            # In some cases the model predicts more than 20 valid words so need to do min.
            score = min(max_words, len(valid_words))

            # Print results
            print(f"\nEvaluated words: {unique_words}")
            print(f"Valid words: {valid_words}")
            print(
                f"The model found {len(words)} of which {len(unique_words)} were unique."
            )
            print(f"The model found {len(valid_words)} valid words.")
            print(f"Score: {score} / {max_words} possible\n")
            return Evaluation(
                evaluator_name=self.__class__.__name__,
                model=self.model,
                letters=letters,
                model_reply=model_reply,
                valid_words=valid_words,
                score=score,
            )
        except Exception as e:
            return Evaluation(
                evaluator_name=self.__class__.__name__,
                model=self.model,
                letters=letters,
                model_reply=f"Exception {e} occured during evaluation.",
                valid_words=[],
                score=0,
            )


class EvaluatorJsonResponse(Evaluator):
    def get_prompt(self, letters: List[str], max_words: int = 20):
        return dedent(f"""\
        You are a helpful assistant that can unscramble letters to form English words.
        Given these letters: {", ".join(letters)},
        Generate up to {max_words} English words that can be made from those letters.
        Please respond with a list of words in JSON format, e.g. ["cat", "bat"].

        The words need to be a minimum of 3 letters.

        Prefer to include commonly used words.

        Include words that have more letters as well.

        Do not include words that only have a sound.

        Do not include abbbreviations. Only include common words.

        Do not include the same word twice.

        Don't include multiples such as "cats" or "balls"

        Provide your final response using the format "```json" and end with ```
        """)

    def model_response_to_words(self, response: str) -> List[str]:
        # Get the content between ```json and ```
        json_string = extract_json_block(response)
        words = json.loads(json_string)
        return words


class EvaluatorNewLinePerWord(Evaluator):
    def get_prompt(self, letters: List[str], max_words: int = 20):
        return dedent(f"""\
        You are a helpful assistant that can unscramble letters to form English words.
        Given these letters: {", ".join(letters)}.
        Generate up to {max_words} English words that can be made from those letters.
        The words need to be a minimum of 3 letters.
        Prefer to include commonly used words.
        Include words that have more letters as well.
        Do not include words that only have a sound.
        Do not include abbbreviations.
        Do not include the same word twice.
        Don't include multiples such as "cats" or "balls".

        Your final response should use the following format:
        <response>
        word1
        word2
        </response>
        """)

    def model_response_to_words(self, response: str) -> List[str]:
        # Get the content between <response> and </response>
        words = (
            response.split("<response>")[1].split("</response>")[0].strip().split("\n")
        )
        return words


def generate_random_lowercase_letters(num_letters: int) -> List[str]:
    return [random.choice(string.ascii_lowercase) for _ in range(num_letters)]


def can_form_word_from_letters(word, letters):
    """
    Check if 'word' can be formed from the multiset of 'letters'.
    Example: word='cat', letters=['c','a','t','b'] -> True
    """

    word_letter_count = Counter(word.lower())
    letters_count = Counter(letter.lower() for letter in letters)

    for letter, count in word_letter_count.items():
        if letters_count[letter] < count:
            return False
    return True


def extract_json_block(text):
    """
    Extracts content between ```json and ``` markers from a string.

    Args:
        text (str): The input text containing JSON code blocks

    Returns:
        str: The extracted JSON content or empty string if no match found

    Example:
        >>> text = "Some text\\n```json\\n{\\"key\\": \\"value\\"}\\n```\\nMore text"
        >>> extract_json_block(text)
        '{\\"key\\": \\"value\\"}'
    """
    import re

    # Pattern to match content between ```json and ```
    pattern = r"```json\s*(.*?)\s*```"

    # Use re.DOTALL to make dot match newlines
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    return ""


def evaluate_all_models(number_of_test_cases: int = 10):
    test_cases = [
        generate_random_lowercase_letters(24) for _ in range(number_of_test_cases)
    ]
    evaluatorsClasses = [EvaluatorJsonResponse, EvaluatorNewLinePerWord]
    results = []
    for evaluatorClass in evaluatorsClasses:
        for model in together_ai_models + local_models:
            evaluator = evaluatorClass(model)
            model_score = 0
            for letters in test_cases:
                evaluation = evaluator.evaluate_model_on_letters(letters)
                print(f"Evaluation for {model}:")
                print(evaluation)
                model_score += evaluation.score
                results.append(evaluation.to_dict())
                print("\n")
            print(f"Average score for {model}: {model_score / len(test_cases)}\n")
        # Append results to results.json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        print("Results saved to results.json")


if __name__ == "__main__":
    evaluate_all_models()
