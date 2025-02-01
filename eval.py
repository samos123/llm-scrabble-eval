import os
from collections import Counter
from openai import OpenAI
import json

from word_validator import WordValidator

# Optional: Uncomment and replace with your own key or set via environment variable
# openai.api_key = "YOUR_OPENAI_API_KEY_HERE"

# If you want to perform dictionary checks using NLTK, you can do:
# client = OpenAI()
# model = "gpt-4o-mini"  # or "gpt-4" if you have access

client = OpenAI(
  api_key=os.environ.get("TOGETHER_API_KEY"),
  base_url="https://api.together.xyz/v1",
)
# model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
# model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
# model = "Qwen/Qwen2.5-7B-Instruct-Turbo"
# model = "Qwen/Qwen2.5-72B-Instruct-Turbo" # great model
# model = "deepseek-ai/DeepSeek-V3" # best oss model so far
# model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" # doesn't work because it thinks as response
# model = "mistralai/Mistral-7B-Instruct-v0.2" # decent small oss model so far
# model = "google/gemma-2-9b-it" # error from togetherai
# model = "Gryphe/MythoMax-L2-13b"
model = "mistralai/Mixtral-8x22B-Instruct-v0.1"


word_validator = WordValidator()


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


def evaluate_model_on_letters(letters, max_words=20):
    """
    1. Prompts the model with the given letters.
    2. Parses the response.
    3. Validates the words.
    4. Computes a simple score.
    """

    # Construct a prompt
    # Instruct the model to return up to 20 English words from the given letters
    prompt = f"""
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
    """

    response = client.chat.completions.create(
        model=model,  # or "gpt-4" if you have access
        messages=[
            {"role": "user", "content": prompt.strip()},
        ],
        temperature=0.0,
    )

    # Extract the model's reply
    model_reply = response.choices[0].message.content

    print("Raw model reply:")
    print(model_reply)

    # Attempt to parse the reply as JSON or fallback if format differs

    try:
        # Get the content between ```json and ```
        json_string = extract_json_block(model_reply)
        words = json.loads(json_string)
        # If the model responded with a string, convert it to a list
        if isinstance(words, str):
            words = [w.strip() for w in words.split(",")]
    except json.JSONDecodeError:
        # If it's not valid JSON, do a naive split or handle differently
        words = model_reply.split()

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
    score = len(valid_words)

    # Print results
    print(f"\nEvaluated words: {unique_words}")
    print(f"Valid words: {valid_words}")
    print(f"The model found {len(words)} of which {len(unique_words)} were unique.")
    print(f"The model found {len(valid_words)} valid words.")
    print(f"Score: {score} / {max_words} possible\n")

    return {
        "model_reply": model_reply,
        "unique_words": unique_words,
        "valid_words": valid_words,
        "score": score,
    }


if __name__ == "__main__":
    # Example usage
    letters = ["a", "c", "a", "t", "b", "s", "t", "e", "f", "x", "z", "d", "o", "p", "l", "u"]
    print(f"len(letters): {len(letters)}")
    print(f"input letters: {letters}")

    result = evaluate_model_on_letters(letters, max_words=20)
    print(result)

    # You can also handle the returned 'result' dict however you want
    # e.g., store it, compare multiple runs, or run multiple test cases.
