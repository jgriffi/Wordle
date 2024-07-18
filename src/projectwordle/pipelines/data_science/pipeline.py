from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    wordle_scoring_couples,
    best_couples,
    best_singles,
    top_words,
    simulating_openers,
    best_opening_word_from_best_couples,
    wordle_scoring_singles,
    simulating_top_words,
    simulating_alternative_openers,
    alternative_openers,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = wordle_scoring_couples,            
                inputs = [
                    "params:num_combo",
                    "params:word_length",
                    "params:word_col",
                    "params:batch",
                    "five_letter_words",
                    "five_letter_words_anagrams",
                ],
                outputs = "wordle_scoring_couples",
                name = "wordle_scoring_couples_node"
            ),
            node(
                func = best_couples,            
                inputs = ["wordle_scoring_couples"],
                outputs = "best_couples",
                name = "best_couples_node"
            ),
            node(
                func = wordle_scoring_singles,            
                inputs = [
                    "five_letter_words_anagrams",
                    "five_letter_words",
                    "params:word_length",
                    "params:word_col",
                    "params:batch",
                ],                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                outputs="wordle_scoring_singles",
                name = "wordle_scoring_singles_node"
            ),
            node(
                func = best_singles,            
                inputs = [
                    "wordle_scoring_singles",
                    "five_letter_words_anagrams"
                ],
                outputs = "best_singles",
                name = "best_singles_node"
            ),
            node(
                func = top_words,            
                inputs = [
                    "best_couples",
                    "five_letter_words_anagrams"
                ],
                outputs = "top_words",
                name = "top_words_node"
            ),
            node(
                func = simulating_openers,            
                inputs = [
                    "params:weighted",
                    "params:k",
                    "params:number_of_tries",
                    "params:mapping",
                    "five_letter_words",
                    "five_letter_words_anagrams",
                    "best_couples"
                ],
                outputs = "simulating_openers",
                name = "simulating_openers_node"
            ),
            node(
                func = best_opening_word_from_best_couples,            
                inputs = [
                    "wordle_scoring_couples",
                    "five_letter_words_anagrams"
                ],
                outputs = [
                    "best_singles_from_best_couples",
                    "best_opening_word_from_best_couples"
                ],
                name = "best_opening_word_from_best_couples_node"
            ),
            node(
                func = simulating_top_words,            
                inputs = [
                    "params:word_length",
                    "params:weighted",
                    "params:k",
                    "params:number_of_tries",
                    "params:mapping",
                    "params:num_words_to_simulate",
                    "five_letter_words",
                    "five_letter_words_anagrams",
                    "best_singles",
                    "params:raise_anagrams",
                    "params:salet_anagrams",
                    "params:soare_anagrams",
                    "params:trace_anagrams"
                ],
                outputs = "simulating_top_words",
                name = "simulating_top_words_node"
            ),
            node(
                func = simulating_alternative_openers,            
                inputs = [
                    "params:weighted",
                    "params:k",
                    "params:number_of_tries_alt_second_guess",
                    "params:mapping",
                    "simulating_top_words",
                    "five_letter_words",
                    "five_letter_words_anagrams",
                    "best_singles",
                ],
                outputs = "simulating_alternative_openers",
                name = "simulating_alternative_openers_node"
            ),
            node(
                func = alternative_openers,            
                inputs = [
                    "params:weighted",
                    "params:k",
                    "params:number_of_tries",
                    "params:mapping",
                    "five_letter_words",
                    "five_letter_words_anagrams",
                    "best_singles",
                    "simulating_alternative_openers"
                ],
                outputs = "alternative_openers",
                name = "alternative_openers_node"
            ),
        ]
    )
