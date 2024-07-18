from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    google_books_words,
    get_english_words,
    processed_five_letter_words,
    five_letter_words_anagrams,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=google_books_words,
                inputs=[
                    "params:volumes",
                    "params:response_code",
                    "params:most_common_letters",
                    "params:google_books",
                ],
                outputs="google_books",
                name="google_books_node"
            ),
            node(
                func=get_english_words,
                inputs=[
                    "params:response_code",
                    "params:most_common_letters",
                    "params:word_freq_fill_null",
                    "params:english_words",
                    "google_books",
                ],
                outputs="english_words",
                name="english_words_node"
            ),
            node(
                func=processed_five_letter_words,            
                inputs=[
                    "params:word_length",
                    "english_words",
                ],
                outputs="five_letter_words",
                name="five_letter_words_node"
            ),
            node(
                func=five_letter_words_anagrams,            
                inputs=[
                    "english_words",
                    "five_letter_words",
                ],
                outputs="five_letter_words_anagrams",
                name="five_letter_words_anagrams_node"
            ),
        ]
    )
