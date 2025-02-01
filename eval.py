import os
from collections import Counter
from openai import OpenAI
import nltk
import json

nltk.download("words")
from nltk.corpus import words as nltk_words
# Optional: Uncomment and replace with your own key or set via environment variable
# openai.api_key = "YOUR_OPENAI_API_KEY_HERE"

# If you want to perform dictionary checks using NLTK, you can do:
# def is_real_english_word(w):
#     return w.lower() in nltk_words
client = OpenAI()


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

    Include words that have more letters as well.

    Do not include words that only have a sound. Only include common words.

    Do not include abbbreviations. Only include common words.

    Do not include ``` in your response and only respond with the json itself.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4" if you have access
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
        words = json.loads(model_reply)
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
            # if is_real_english_word(w):
            #     valid_words.append(w)
            # else:
            #     print(f"Invalid English word (not in dictionary): {w}")

            valid_words.append(w)
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
    letters = ["a", "c", "a", "t", "b", "s", "t", "e", "f", "x", "z", "d", "o"]
    print(f"len(letters): {len(letters)}")
    print(f"input letters: {letters}")

    result = evaluate_model_on_letters(letters, max_words=20)
    print(result)

    # You can also handle the returned 'result' dict however you want
    # e.g., store it, compare multiple runs, or run multiple test cases.
