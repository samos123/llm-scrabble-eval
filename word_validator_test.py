from word_validator import WordValidator


def test_word_is_valid():
    assert WordValidator().is_real_english_word("hello") == True
    assert WordValidator().is_real_english_word("asdfas") == False
