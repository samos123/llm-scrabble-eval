from collections import Counter
import nltk
from nltk.corpus import words
from typing import List, Set


class WordValidator:
    """A class to validate words using NLTK corpus and letter constraints"""

    def __init__(self):
        """Initialize the validator by downloading and loading NLTK words"""
        try:
            nltk.download("words", quiet=True)
            self.english_words = set(words.words())
        except Exception as e:
            print(f"Error initializing NLTK words: {e}")
            self.english_words = set()

    def is_real_english_word(self, word: str) -> bool:
        """
        Check if a word exists in the NLTK English word corpus

        Args:
            word (str): The word to check

        Returns:
            bool: True if the word exists in the NLTK corpus
        """
        return word.lower() in self.english_words

    def can_form_word_from_letters(self, word: str, letters: List[str]) -> bool:
        """
        Check if a word can be formed from the given letters

        Args:
            word (str): The word to form
            letters (List[str]): Available letters

        Returns:
            bool: True if the word can be formed from the letters
        """
        word_letter_count = Counter(word.lower())
        letters_count = Counter(letter.lower() for letter in letters)

        for letter, count in word_letter_count.items():
            if letters_count[letter] < count:
                return False
        return True

    def get_valid_words(self, letters: List[str], max_words: int = None) -> List[str]:
        """
        Get all valid English words that can be formed from the given letters

        Args:
            letters (List[str]): Available letters
            max_words (int, optional): Maximum number of words to return

        Returns:
            List[str]: List of valid words that can be formed
        """
        valid_words = []
        for word in self.english_words:
            if self.can_form_word_from_letters(word, letters):
                valid_words.append(word)
                if max_words and len(valid_words) >= max_words:
                    break
        return valid_words


if __name__ == "__main__":
    test_word_validator()
