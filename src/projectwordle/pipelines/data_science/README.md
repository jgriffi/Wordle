# Wordle Analysis Functions

This repository contains a collection of functions for analyzing and simulating Wordle-like games. Each function serves a specific purpose, from creating word combinations to simulating guesses and evaluating scoring patterns. Below are detailed descriptions of each function.

## `create_guess_combo_words`

Create a list of word combinations from a DataFrame column, filtering out words with repeat letters.

### Description
This function takes a Polars DataFrame, filters rows based on specific conditions, and generates all possible combinations of words from a specified column. It ensures that the resulting combinations do not have any repeated letters across the combined words.

### Parameters
- **dataf (pl.DataFrame):** The input DataFrame containing word data.
- **num_combo (int):** The number of words to combine.
- **word_col (str):** The name of the column containing the words to be combined.

### Returns
- **List[Tuple[str, str]]:** A list of tuples, where each tuple contains a combination of words that have no repeated letters.

## `add_group`

Adds a 'group' column to the input DataFrame, where each group consists of a specified number of rows.

### Description
This function creates a new column 'group' in the DataFrame by dividing the row indices by the `tries` parameter. This is useful for creating grouped data based on a fixed number of rows, such as in scenarios where you want to batch process rows (e.g., grouping every 6 rows together for a Wordle game attempt).

### Parameters
- **dataf (pl.DataFrame):** The input DataFrame to which the 'group' column will be added.
- **tries (int):** The number of rows to include in each group. For example, if `tries` is 6, every 6 rows will be grouped together.

### Returns
- **pl.DataFrame:** A new DataFrame with an additional 'group' column.

## `correct_guess_number`

Computes the number of tries until the correct guess is made in a grouped DataFrame.

### Description
This function iterates over groups in the provided DataFrame, checking for rows where the `challenge`, `guess`, and `next_guess` columns are equal, and where the `letter_differences` column matches the specified conditional value. It records the number of tries it took to achieve this condition within each group and returns a DataFrame with the number of tries appended as a new column.

### Parameters
- **dataf (pl.DataFrame):** The input DataFrame containing the data to be processed.
- **conditional (Union[str, int]):** The value that the `letter_differences` column must match.
- **challenge (str):** The name of the column containing the challenge data. Default is "challenge".
- **guess (str):** The name of the column containing the guess data. Default is "guess".
- **index (str):** The name of the index column. Default is "index".
- **next_guess (str):** The name of the column containing the next guess data. Default is "next_guess".
- **letter_differences (str):** The name of the column containing the letter differences data. Default is "letter_differences".

### Returns
- **pl.DataFrame:** The original DataFrame with an additional column named 'tries' that indicates the number of tries until the correct guess was made in each group. If no correct guess is found, the value is set to None for that group.

## `wordle_scoring_couples`

Computes the scoring for Wordle-like games based on challenge words and guessed word combinations.

### Description
This function processes the given data of five-letter words and their anagrams in batches to compute the scores for each challenge word against combinations of guessed words. The results include various scoring metrics and patterns that are returned as a dictionary of functions to generate Polars DataFrames.

### Parameters
- **num_combo (int):** Number of combinations of guessed words to be created.
- **word_col (str):** Column name in `five_letter_words_anagrams` for the words.
- **batch (int):** Batch size for processing the data.
- **five_letter_words (pl.DataFrame):** DataFrame containing the processed five-letter words to be used as challenge words.
- **five_letter_words_anagrams (pl.DataFrame):** DataFrame containing anagrams of five-letter words to be used as guessed word combinations.

### Returns
- **Dict[str, Callable[[], pl.DataFrame]]:** Dictionary where the keys are filenames and the values are functions that return Polars DataFrames containing the scoring data for each batch.

### Sub-function: `batched_processings`

Performs batched processing of challenge words and guessed word combinations to compute and score letter patterns.

#### Parameters
- **dataf (pl.DataFrame):** The input DataFrame containing the challenge words in the column specified by `word_col`.

#### Returns
- **pl.DataFrame:** A new DataFrame containing detailed score metrics and patterns for each challenge word and its respective guessed word combinations.

## `best_couples`

Computes the best word pairs from a collection of scored Wordle guesses.

### Parameters
- **wordle_scored (Dict[str, Callable[[], pl.DataFrame]]):** A dictionary where the keys are partition identifiers and the values are callables that return a DataFrame. Each DataFrame should contain at least two columns:
  - **guess_words (str):** The Wordle guess words.
  - **sum_correct_letter_pattern (int):** The score representing the number of correct letters in the guess word.

### Returns
- **pl.DataFrame:** A DataFrame containing the guess words, their total correct letter patterns summed across all partitions, and the absolute difference from the top score. The DataFrame is sorted in descending order by the total correct letter pattern.

## `wordle_scoring_singles`

Computes Wordle scoring patterns for single batches of words.

### Description
This function takes in two dataframes: one containing potential challenge words and another containing guess words without repeated letters. It computes positional scoring patterns for each pair of challenge and guess words in batches, returning the results in a dictionary of dataframes.

### Parameters
- **five_letter_words_anagrams (pl.DataFrame):** DataFrame containing words with their lengths and a flag for repeated letters.
- **five_letter_words (pl.DataFrame):** DataFrame containing five-letter words.
- **word_length (int):** The length of the words to be considered.
- **word_col (str):** The name of the column containing the words.
- **batch (int):** The number of words to process in each batch.

### Returns
- **Dict[str, pl.DataFrame]:** A dictionary where keys are filenames and values are DataFrames containing scoring results.

### Sub-function: `batched_processings`

Processes a batch of words to compute Wordle scoring patterns.

#### Parameters
- **dataf (pl.DataFrame):** DataFrame containing a batch of challenge words.

#### Returns
- **pl.DataFrame:** DataFrame containing the challenge words, guess words, their positional patterns, and total scores.

## `best_singles`

Aggregates and analyzes Wordle scores, returning a DataFrame of best single guesses along with their scores and anagram information.

### Parameters
- **wordle_scored (Dict[str, Callable[[], pl.DataFrame]]):** A dictionary where keys are partition names and values are functions that return DataFrames containing Wordle guesses and their scores.
- **five_letter_words_anagrams (pl.DataFrame):** A DataFrame containing five-letter words and their corresponding anagrams.

### Returns
- **pl.DataFrame:** A DataFrame with the best guesses, their summed scores, difference from the top score, and associated anagrams, sorted by the difference from the top score.

### Sub-function: `get_grouped_sum`

Computes the sum of guess scores for each partitioned dataset and concatenates the results.

#### Parameters
- **partitioned_datasets (Dict[str, Callable[[], pl.DataFrame]]):** A dictionary of functions returning DataFrames to be processed.

#### Returns
- **pl.DataFrame:** A DataFrame with summed scores for each guess across all partitions.

## `top_words`

Finds the most frequent five-letter words that do not repeat letters and are common in the given best couple words list.

### Parameters
- **best_couples (pl.DataFrame):** DataFrame containing a column 'guess_words' with words.
- **five_letter_words_anagrams (pl.DataFrame):** DataFrame containing columns 'words', 'has_repeat_letters', and 'common_letters'.

### Returns
- **pl.DataFrame:** DataFrame containing the top frequent words sorted by count in descending order.

## `simulating_openers`

Simulate a word guessing game with specified parameters and generate results.

### Description
This function simulates a word guessing game by trying to guess challenge words using opening guesses from a list of best couple guesses. The simulation is based on certain criteria such as word frequency, letter matching, and pattern recognition, and returns a DataFrame with the simulation results.

### Parameters
- **word_length (int):** The length of the words to be used as challenge words.
- **weighted (bool):** If True, the selection of possible guesses is weighted by word frequency.
- **k (int):** The maximum number of possible guesses to consider at each step.
- **num_tries (int):** The number of attempts allowed to guess the challenge word.
- **mapping (Dict[str, Any]):** A dictionary mapping match pattern characters to their respective values.
- **five_letter_words (pl.DataFrame):** A DataFrame containing five-letter words with their frequencies and lengths.
- **best_couples (pl.DataFrame):** A DataFrame containing the best opening guess word pairs.



### Returns
- **pl.DataFrame:** A DataFrame containing the simulation results, including the number of tries, matched patterns, and guessed words.

### Sub-function: `simulate`

Simulates the guessing of a single word based on the criteria and parameters provided.

#### Parameters
- **challenge_word (str):** The word to be guessed.
- **guess_words (List[str]):** A list of potential guess words.
- **k (int):** The maximum number of guesses to consider at each step.
- **num_tries (int):** The maximum number of tries allowed to guess the word.
- **mapping (Dict[str, Any]):** A dictionary mapping match pattern characters to their respective values.
- **weighted (bool):** If True, guesses are weighted by frequency.

#### Returns
- **pl.DataFrame:** A DataFrame containing the simulation results for the given challenge word.

---

Each function in this collection provides a unique capability to analyze, score, and simulate word-guessing games like Wordle. By leveraging these functions, users can gain insights into patterns, optimize their guessing strategies, and evaluate word combinations effectively.