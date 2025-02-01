from eval import EvaluatorJsonResponse, EvaluatorNewLinePerWord

from textwrap import dedent

model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"


def test_evaluator_json_response():
    evaluator = EvaluatorJsonResponse(model=model)
    letters = ["a", "b", "c", "d", "e"]
    max_words = 3
    prompt = evaluator.get_prompt(letters, max_words)
    assert "a, b, c, d, e" in prompt

    model_response = dedent(
        """\
        The model responded with something.
        And then a json block.

        ```json
        ["cat", "bat", "dog"]
        ```
        """
    )
    assert evaluator.model_response_to_words(model_response) == ["cat", "bat", "dog"]

    # evaluation = evaluator.evaluate_model_on_letters(letters, max_words=2)
    # assert evaluation.score != 0


def test_evaluator_new_line_per_word():
    evaluator = EvaluatorNewLinePerWord(model=model)
    letters = ["a", "b", "c", "d", "e"]
    max_words = 3
    prompt = evaluator.get_prompt(letters, max_words)
    assert "a, b, c, d, e" in prompt

    model_response = dedent(
        """\
        The model responded with something.
        And then a json block.

        <response>
        cat
        bat
        dog
        </response>
        ```
        """
    )
    assert evaluator.model_response_to_words(model_response) == ["cat", "bat", "dog"]
