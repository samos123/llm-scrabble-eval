from typing import List
from textwrap import dedent
import json
from collections import Counter
from eval import generate_random_lowercase_letters

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="ignored",
)

def generate_words(letters: List[str], max_words: int = 20) -> List[str]:
    """
    Generates words from the given letters using a word validator.
    """
    prompt = get_prompt(letters, max_words)
    response = client.chat.completions.create(
        model="phi-4-ollama-l4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "```json"},
        ],
        max_tokens=1000,
        temperature=0.7,
    )

    model_reply = response.choices[0].message.content
    # Get the content between ```json and ```
    json_string = extract_json_block(model_reply)
    words = json.loads(json_string)
    valid_words = []
    for word in words:
        if can_form_word_from_letters(word, letters):
            valid_words.append(word)
        else:
            print(f"Ignoring word {word} because it's not valid.")
    return valid_words


def get_prompt(letters: List[str], max_words: int = 20):
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

if __name__ == "__main__":
    letters = generate_random_lowercase_letters(24)
    print(generate_words(letters, 15))
