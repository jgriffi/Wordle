import polars as pl
import numpy as np
import random
from tqdm import tqdm
from itertools import zip_longest
from typing import Dict, Any, Callable, Union, Tuple, List
from itertools import combinations


def create_guess_combo_words(dataf: pl.DataFrame, num_combo: int, word_length: int, word_col: str) -> List[Tuple[str, str]]:
    """
    Create a list of word combinations from a DataFrame column, filtering out words with repeat letters.

    This function takes a Polars DataFrame, filters rows based on specific conditions, and generates
    all possible combinations of words from a specified column. It ensures that the resulting
    combinations do not have any repeated letters across the combined words.

    Parameters:
    -----------
        - dataf (pl.DataFrame): The input DataFrame containing word data.
        - num_combo (int): The number of words to combine.
        - word_col (str): The name of the column containing the words to be combined.

    Returns:
    --------
    List[Tuple[str, str]]: A list of tuples, where each tuple contains a combination of words
                           that have no repeated letters.
    """
    # Convert the DataFrame to a Polars DataFrame and filter rows based on conditions
    guess_words = (
        dataf
        .filter(
            ~(pl.col("has_repeat_letters")) # same as (pl.col("has_repeat_letters") == False)
            & (pl.col("common_letters")) # (pl.col("common_letters") == True) 
        )
        [word_col]  
    )

    # Generate combinations of words and filter out combinations with repeated letters
    return list(
        combo for combo in combinations(guess_words, num_combo)
        if len(set("".join(combo))) == (word_length * num_combo)
    )


def add_group(dataf: pl.DataFrame, tries: int) -> pl.DataFrame:
    """
    Adds a 'group' column to the input DataFrame, where each group consists of a specified number of rows.

    This function creates a new column 'group' in the DataFrame by dividing the row indices by the `tries` parameter.
    This is useful for creating grouped data based on a fixed number of rows, such as in scenarios where
    you want to batch process rows (e.g., grouping every 6 rows together for a Wordle game attempt).

    Parameters:
    -----------
        - dataf (pl.DataFrame): The input DataFrame to which the 'group' column will be added.
        - tries (int): The number of rows to include in each group. For example, if `tries` is 6, every 6 rows
                       will be grouped together.

    Returns:
    --------
    pl.DataFrame: A new DataFrame with an additional 'group' column.
    
    """
    return (
        dataf
        .with_row_index("index")
        .with_columns(group = (pl.col("index") // tries).cast(pl.UInt32))
    )


def wordle_scoring_couples(
    num_combo: int,
    word_length: int,
    word_col: str,
    batch: int,
    five_letter_words: pl.DataFrame,
    five_letter_words_anagrams: pl.DataFrame,
) -> Dict[str, Callable[[], pl.DataFrame]]:
    """
    Computes the scoring for Wordle-like games based on challenge words and guessed word combinations.

    This function processes the given data of five-letter words and their anagrams in batches to 
    compute the scores for each challenge word against combinations of guessed words. The results 
    include various scoring metrics and patterns that are returned as a dictionary of functions 
    to generate Polars DataFrames.

    Parameters:
    -----------
        - processed_five_letter_words : pl.DataFrame
            DataFrame containing the processed five-letter words to be used as challenge words.
        - five_letter_words_anagrams : pl.DataFrame
            DataFrame containing anagrams of five-letter words to be used as guessed word combinations.
        - "num_combo": int, number of combinations of guessed words to be created.
        - "words": str, column name in `five_letter_words_anagrams` for the words.
        - "batch": int, batch size for processing the data.

    Returns:
    --------
    Dict[str, Callable[[], pl.DataFrame]]
        Dictionary where the keys are filenames and the values are functions that return Polars DataFrames 
        containing the scoring data for each batch.
    """

    # Define guess words
    guess_words = create_guess_combo_words(five_letter_words_anagrams, num_combo, word_length, word_col)
    print(f"Number of guess word pairs: {len(guess_words)}")

    # Lets process the data in batches to avoid memory issues
    groups = (
        five_letter_words
        .with_row_index()
        .group_by(pl.col("index") % batch)
    )

    # Define a function for batched processing
    def batched_processings(dataf: pl.DataFrame) -> pl.DataFrame:
        # List to store dictionaries of scores
        data = []

        # Define challenge words
        challenge_words = dataf[word_col]

        # Iterate over each challenge word
        for word in tqdm(challenge_words):
            # Iterate over each combination of guessed words
            for combo in guess_words:
                num_of_words, length_of_words = len(combo), len(combo[0])
                # Initialize scores array
                scores = np.zeros(num_of_words * length_of_words, dtype = np.int8).reshape(num_of_words, length_of_words)
                
                # Calculate scores for each guessed word
                for idx, selection in enumerate(combo):
                    for idx2, (x, y) in enumerate(zip(word, selection)):
                        if x == y:
                            scores[idx][idx2] = 1
                        elif x != y and y in word:
                            scores[idx][idx2] = 0
                        else:
                            scores[idx][idx2] = -1

                # Convert scores array to string representation for DataFrame
                positional_scores = [" ".join(map(str, subarr)) for subarr in scores]

                # Create dictionary containing score-related metrics
                temp: Dict[str, Any] = {
                    "challenge_word": word,
                    "guess_words": ", ".join(combo),
                    "first_guess": combo[0],
                    "second_guess": combo[1],
                    "first_word_pattern": positional_scores[0],
                    "second_word_pattern": positional_scores[1],
                    "correct_letter_pattern": " ".join([str(elem) for elem in np.max(scores, axis = 0)]),
                    "first_word_score": scores[0].sum(),
                    "second_word_score": scores[1].sum(),
                    "sum_scores": scores.sum(),
                    "sum_correct_letter_pattern": np.sum(np.max(scores, axis = 0)),
                }

                # Append dictionary to data list
                data.append(temp)  

        # Convert data list to Polars DataFrame
        return (
            pl.DataFrame(data)
            .with_columns(
                pl.col("challenge_word").cast(pl.String),
                pl.col("guess_words").cast(pl.String),
                pl.col("first_guess").cast(pl.String),
                pl.col("second_guess").cast(pl.String),
                pl.col("first_word_pattern").cast(pl.String),
                pl.col("second_word_pattern").cast(pl.String),
                pl.col("correct_letter_pattern").cast(pl.String),
                pl.col("first_word_score").cast(pl.Int8),
                pl.col("second_word_score").cast(pl.Int8),
                pl.col("sum_scores").cast(pl.Int8),
                pl.col("sum_correct_letter_pattern").cast(pl.Int8)
            )
        )

    return {
        f"wordle_scoring_data_{str(i[0])}.parquet": batched_processings(group)
        for i, group in groups
    }


def best_couples(wordle_scored: Dict[str, Callable[[], pl.DataFrame]]) -> pl.DataFrame:
    """
    Computes the best word pairs from a collection of scored Wordle guesses.

    Parameters:
    -----------
    wordle_scored (Dict[str, Callable[[], pl.DataFrame]]): A dictionary where the keys are partition identifiers 
    and the values are callables that return a DataFrame. Each DataFrame should contain at least two columns:
        - "guess_words" (str): The Wordle guess words.
        - "sum_correct_letter_pattern" (int): The score representing the number of correct letters in the guess word.

    Returns:
    --------
    pl.DataFrame: A DataFrame containing the guess words, their total correct letter patterns summed across all partitions,
    and the absolute difference from the top score. The DataFrame is sorted in descending order by the total correct letter pattern.
        
    The resulting DataFrame has the following columns:
        - "guess_words" (str): The Wordle guess words.
        - "total_correct_letter_pattern" (int): The summed score of correct letter patterns for each guess word across all partitions.
        - "abs_diff_from_top_score" (int): The absolute difference of each guess word's total score from the highest score.
    """

    def get_grouped_sum(partitioned_datasets: Dict[str, Callable[[], pl.DataFrame]]) -> pl.DataFrame:
        """
        Aggregates the total correct letter patterns for each guess word across multiple partitions.

        Parameters:
        -----------
        partitioned_datasets (Dict[str, Callable[[], pl.DataFrame]]): A dictionary of partitioned datasets.

        Returns:
        --------
        pl.DataFrame: A DataFrame containing the guess words and their summed correct letter patterns.
        """

        # List to store aggregated DataFrames from each partition
        summed_data = [] 
        
        # Iterate through each partition's dataset
        for _, partition_val in tqdm(partitioned_datasets.items()):
            dataset = (
                partition_val() # Call the function to get the DataFrame
                .select(["guess_words", "sum_correct_letter_pattern"]) # Select only relevant columns
                .group_by("guess_words")
                .agg(total_correct_letter_pattern = (pl.col("sum_correct_letter_pattern").sum()).cast(pl.Int16))
            )
            summed_data.append(dataset) 
        return pl.concat(summed_data)

    # The main process
    return (
        get_grouped_sum(wordle_scored) # Call the function to get aggregated results
        .group_by("guess_words") # Group by 'guess_words' again for final aggregation
        .agg(total_correct_letter_pattern = (pl.col("total_correct_letter_pattern").sum()).cast(pl.Int16))
        .sort("total_correct_letter_pattern", descending=True)
        .with_columns(
            # Compute the absolute difference from the top score
            abs_diff_from_top_score = (
                (pl.col("total_correct_letter_pattern") - pl.col("total_correct_letter_pattern").max())
                .abs()
                .cast(pl.UInt16)
            )
        )
    )


def wordle_scoring_singles(
    five_letter_words_anagrams: pl.DataFrame,
    five_letter_words: pl.DataFrame,
    word_length: int,
    word_col: str,
    batch: int
) -> Dict[str, Callable[[], pl.DataFrame]]:
    """
    Computes Wordle scoring patterns for single batches of words.

    This function takes in two dataframes: one containing potential challenge words 
    and another containing guess words without repeated letters. It computes 
    positional scoring patterns for each pair of challenge and guess words in 
    batches, returning the results in a dictionary of dataframes.

    Parameters:
    -----------
        - five_letter_words_anagrams (pl.DataFrame): DataFrame containing words with their lengths and a flag for repeated letters.
        - five_letter_words (pl.DataFrame): DataFrame containing five-letter words.
        - word_length (int): The length of the words to be considered.
        - word_col (str): The name of the column containing the words.
        - batch (int): The number of words to process in each batch.

    Returns:
    --------
    Dict[str, pl.DataFrame]: A dictionary where keys are filenames and values are DataFrames containing scoring results.
    """

    # Group words by their index modulo batch size
    groups = (
        five_letter_words
        .with_row_index()
        .group_by(pl.col("index") % batch)
    )

    # Filter guess words to those with the specified word length and without repeated letters
    guess_words = (
        five_letter_words_anagrams
        .filter(
            (pl.col("word_length") == word_length) &
            ~(pl.col("has_repeat_letters"))
        )
        [word_col]
        .to_list()
    )

    def batched_processings(dataf: pl.DataFrame) -> pl.DataFrame:
        """
        Processes a batch of words to compute Wordle scoring patterns.

        Parameters:
        -----------
            - dataf (pl.DataFrame): DataFrame containing a batch of challenge words.

        Returns:
        --------
        pl.DataFrame: DataFrame containing the challenge words, guess words, 
                      their positional patterns, and total scores.
        """

        data = []

        # Extract challenge words with the specified word length
        challenge_words = (
            dataf
            .filter(pl.col("word_length") == word_length)[word_col]
            .to_list()
        )

        # Compute positional patterns for each pair of challenge and guess words
        for word in tqdm(challenge_words):
            for guess_word in guess_words:
                score = np.zeros(len(word), dtype=np.int8)

                # Determine positional score for each character
                for idx, (x, y) in enumerate(zip(word, guess_word)):
                    if x == y:
                        score[idx] = 1  # Correct character and position
                    elif y in word:
                        score[idx] = 0  # Correct character, wrong position
                    else:
                        score[idx] = -1  # Incorrect character

                positional_pattern = " ".join(map(str, score))

                temp: Dict[str, Union[str, int]] = {
                    "challenge": word,
                    "guess": guess_word,
                    "guess_pattern": positional_pattern,
                    "guess_score": score.sum(),  # Total score for the guess
                }

                data.append(temp)

        return (
            pl.DataFrame(data)
            .with_columns(
                pl.col("challenge").cast(pl.String),
                pl.col("guess").cast(pl.String),
                pl.col("guess_pattern").cast(pl.String),
                pl.col("guess_score").cast(pl.Int8),
            )
        )

    # Apply batched processing to each group and store results in a dictionary
    return {
        f"wordle_scoring_data_{str(i[0])}.parquet": batched_processings(group)
        for i, group in groups
    }


def best_singles(
    wordle_scored: Dict[str, Callable[[], pl.DataFrame]],
    five_letter_words_anagrams: pl.DataFrame
) -> pl.DataFrame:
    """
    Aggregates and analyzes Wordle scores, returning a DataFrame of best single guesses along with their scores and anagram information.

    Parameters:
    -----------
        - wordle_scored (Dict[str, Callable[[], pl.DataFrame]]): A dictionary where keys are partition names and values are functions that return DataFrames containing Wordle guesses and their scores.
        - five_letter_words_anagrams (pl.DataFrame): A DataFrame containing five-letter words and their corresponding anagrams.

    Returns:
    --------
        - pl.DataFrame: A DataFrame with the best guesses, their summed scores, difference from the top score, and associated anagrams, sorted by the difference from the top score.
    """

    def get_grouped_sum(partitioned_datasets: Dict[str, Callable[[], pl.DataFrame]]) -> pl.DataFrame:
        """
        Computes the sum of guess scores for each partitioned dataset and concatenates the results.

        Parameters:
        -----------
            - partitioned_datasets (Dict[str, Callable[[], pl.DataFrame]]): A dictionary of functions returning DataFrames to be processed.

        Returns:
        --------
            - pl.DataFrame: A DataFrame with summed scores for each guess across all partitions.
        """
        summed_data = []

        # Iterate over partitioned datasets and compute the sum of guess scores
        for _, partition_val in tqdm(partitioned_datasets.items()):
            dataset = (
                partition_val() 
                .group_by("guess", "guess_score")
                .agg(summed_scores=pl.col("guess_score").sum().cast(pl.Int32))
            )
            summed_data.append(dataset)
        
        # Concatenate all the summed DataFrames
        return pl.concat(summed_data)

    # Calculate the absolute difference from the top score and join with anagram data
    return (
        get_grouped_sum(wordle_scored)
        .group_by("guess")
        .agg(summed_scores=pl.col("summed_scores").sum().cast(pl.Int32))
        .sort("summed_scores", descending=True)
        .with_columns(
            abs_diff_from_top_score=(
                (pl.col("summed_scores") - pl.col("summed_scores").max())
                .abs()
                .cast(pl.UInt32)
            )
        )
        .join(
            five_letter_words_anagrams.select(["words", "anagrams"]),
            left_on="guess",
            right_on="words",
            how="left",
            coalesce=True
        )
        .rename({"guess": "word"})
        .sort("abs_diff_from_top_score", descending=False, maintain_order=True)
        .with_row_index(offset=1)
    )


def top_words(best_couples: pl.DataFrame, five_letter_words_anagrams: pl.DataFrame) -> pl.DataFrame:
    """
    Finds the most frequent five-letter words that do not repeat letters and are 
    common in the given best couple words list.

    Parameters:
    -----------
        - best_couples (pl.DataFrame): DataFrame containing a column 'guess_words' with words.
          five_letter_words_anagrams (pl.DataFrame): DataFrame containing columns 'words', 
                                                   'has_repeat_letters', and 'common_letters'.

    Returns:
    --------
    pl.DataFrame: DataFrame containing the top frequent words sorted by count in descending order.
    """

    # Extract the 'guess_words' column from the best_couples DataFrame
    best_couple_words = best_couples["guess_words"]

    # Filter five-letter anagrams to include only those without repeating letters and marked as common
    no_repeat_char_five_letter_anagrams = (
        five_letter_words_anagrams
        .filter(
            ~(pl.col("has_repeat_letters")) &
            (pl.col("common_letters"))
        )
        ["words"]
    )

    # Generate pairs of words where each word in no_repeat_char_five_letter_anagrams 
    # has all its letters present in a word from best_couple_words
    top_letter_word_matches = pl.DataFrame(
        (word, guess_word)
        for word in no_repeat_char_five_letter_anagrams
        for guess_word in best_couple_words
        if all(letter in guess_word for letter in word)
    ).rename({"column_0": "words", "column_1": "guess_words"})

    # Count the occurrences of each word, join with original matches, sort, and remove duplicates
    return (
        top_letter_word_matches
        ["words"]
        .value_counts()
        .join(
            top_letter_word_matches,
            on="words",
            how="inner",
            coalesce=True
        )
        .sort("count", descending=True)
        .drop("guess_words")
        .unique(maintain_order=True)
    )


def simulating_openers(
    weighted: bool,
    k: int,
    num_tries: int,
    mapping: Dict[str, str],
    five_letter_words: pl.DataFrame,
    five_letter_words_anagrams: pl.DataFrame,
    best_couples: pl.DataFrame,
) -> pl.DataFrame:
    """
    Simulate guesses for a word challenge.

    Parameters:
    -----------
        - weighted (bool): Whether to sample guesses weighted by word frequency.
        - k (int): Number of words to consider in each simulation step.
        - num_tries (int): Number of guesses allowed per challenge word.
        - mapping (Dict[str, str]): Mapping for match patterns.
        - five_letter_words (pl.DataFrame): DataFrame of five-letter words with frequencies.
        - five_letter_words_anagrams (pl.DataFrame): A DataFrame containing five-letter words and their anagrams.
        - best_couples (pl.DataFrame): DataFrame containing the best starting word pairs.

    Returns:
    --------
        pl.DataFrame: Simulation results including guess details and patterns.
    """
    data = []

    # Get the best opening words
    openers = best_couples["guess_words"].head(1).item().split(", ")

    # Prepare the list of five-letter words sorted by frequency
    sorted_five_letter_words = (
        five_letter_words
        .select(["words", "word_freq", "word_length"])
        .sort("word_freq", descending=True)
    )

    challenge_words = five_letter_words["words"].shuffle(seed=42)

    for challenge in tqdm(challenge_words):
        # Use the first opener from the list
        current_guess = openers[0]
        encountered_guesses = set()
        updated_letters_to_omit = set()
        updated_letters_to_keep = dict()
        matching_indexes = dict()
        miss_matched_indexes = dict()

        for _ in range(num_tries):
            guess = current_guess
            encountered_guesses.add(guess)

            # Generate match pattern for the current guess
            match_pattern = "".join(
                "G" if x == y else "Y" if x != y and y in challenge else "B"
                for x, y in zip(challenge, guess)
            )
            
            current_guess_match_pattern = [mapping[char] for char in match_pattern]

            # Map letters to their match patterns and positions
            mapped = {
                letter: (pattern, index) for index, (letter, pattern)
                in enumerate(zip(guess, current_guess_match_pattern))
            }

            # Update letters to omit, keep, and index information
            updated_letters_to_omit.update(k for k, v in mapped.items() if v[0] == -1)
            updated_letters_to_keep.update({k: v for k, v in mapped.items() if v[0] != -1})
            matching_indexes.update({k: v[1] for k, v in mapped.items() if v[0] == 1})
            miss_matched_indexes.update({k: v[1] for k, v in mapped.items() if v[0] == 0})

            # Filter out words that don't match the updated constraints
            words_to_filter_out = [
                word for word in challenge_words
                if any(letter in word for letter in updated_letters_to_omit)
                or any(word[position] == letter for letter, (pattern, position) in updated_letters_to_keep.items() if pattern == 0)
                or not all(word[position] == letter for letter, (pattern, position) in updated_letters_to_keep.items() if pattern == 1)
                or not all(letter in word for letter in updated_letters_to_keep.keys())
                or word in encountered_guesses
            ]

            # Update encountered guesses with words not to be considered
            encountered_guesses.update(words_to_filter_out)

            # Filter remaining possible words, excluding encountered guesses
            remaining_words = sorted_five_letter_words.filter(~(pl.col("words").is_in(encountered_guesses))).to_pandas()
            # remaining_words = sorted_five_letter_words.filter(~(pl.col("words").is_in(words_to_filter_out))).to_pandas()

            if weighted:
                try:
                    # Sample words weighted by frequency
                    filtered_words = remaining_words.sample(n=min(k, len(remaining_words)), weights="word_freq")["words"]
                except ValueError:
                    filtered_words = remaining_words["words"].head(k)
            else:
                filtered_words = remaining_words.head(k)["words"]

            num_choices_after_guess = len(remaining_words)

            encountered_guesses.add(current_guess)

            possible_guesses = ", ".join(filtered_words)

            challenge_in_possible_guesses = challenge in possible_guesses

            # Determine the next guess
            if guess == challenge:
                next_guess = guess
                letter_differences = ""
                num_diff_letters = 0
            else:
                # Use the next opener if available, otherwise sample from filtered words
                if guess in openers:
                    next_guess_index = openers.index(guess) + 1
                    next_guess = openers[next_guess_index] if next_guess_index < len(openers) else next(iter(filtered_words), random.choice(challenge_words))
                else:
                    next_guess = next(iter(filtered_words), random.choice(challenge_words))

            if guess != challenge:
                letter_differences = "".join(updated_letters_to_omit)
                num_diff_letters = len(updated_letters_to_omit)
            
            current_guess = next_guess
            
            # Store simulation data
            temp: Dict[str, Any] = {
                "challenge": challenge,
                "guess": guess,
                "match_pattern": match_pattern,
                "letter_differences": letter_differences,
                "common_letters": "".join(updated_letters_to_keep.keys()),
                "num_diff_letters": num_diff_letters,
                "num_common_letters": len(updated_letters_to_keep),
                "num_matching_index": len(matching_indexes),
                "num_choices_after_guess": num_choices_after_guess,
                "possible_guesses": possible_guesses,
                "challenge_in_possible_guesses": challenge_in_possible_guesses,
                "next_guess": next_guess,
            }
            data.append(temp)

    # Create the final DataFrame with appropriate column types
    return (
        pl.DataFrame(data)
        .with_columns(
            pl.col("num_diff_letters").cast(pl.UInt8),
            pl.col("num_common_letters").cast(pl.UInt8),
            pl.col("num_matching_index").cast(pl.UInt8),
            pl.col("num_choices_after_guess").cast(pl.UInt16),
            ((pl.col("challenge") == pl.col("guess")) & (pl.col("challenge") == pl.col("next_guess"))).alias("is_match")
        )
        .pipe(add_group, num_tries)
        .pipe(compute_tries)
        .pipe(difficulty)
        .join(
            five_letter_words_anagrams.select("words", "anagrams", "anagram_num"),
            how="left",
            left_on="guess",
            right_on="words",
            coalesce=True
        )
        .rename({"anagrams": "guess_word_anagrams"})
    )


def compute_tries(data: pl.DataFrame) -> pl.DataFrame:
    """
    Computes the number of tries (rows) before encountering the first True value 
    in the 'is_match' column within each group in the given DataFrame.

    Parameters:
    -----------
        - data (pl.DataFrame): Input DataFrame containing the columns 'group' and 'is_match'.

    Returns:
    --------
    pl.DataFrame: DataFrame with the same structure as the input, but with an additional 
                  'tries' column indicating the number of rows before the first True 
                  'is_match' value within each group. If no True value is found, 'tries' 
                  will be missing for that group.
    """
    # Add a temporary row index within each group
    grouped = (
        data
        .with_columns(pl.arange(0, pl.len()).over("group").alias("temp_index"))
    )

    # Compute the first index of True values within each group
    first_true_idx = (
        grouped
        .filter(pl.col("is_match"))  # Filter rows where 'is_match' is True
        .group_by("group")  # Group by 'group' column again
        .agg([(pl.col("temp_index").min() + 1).alias("tries").cast(pl.UInt8)])  # Calculate the minimum 'temp_index' and cast to UInt8
    )

    # Merge the tries information back to the original DataFrame
    result = (
        grouped
        .join(first_true_idx, on="group", how="left", coalesce=True)  # Left join on 'group' to add 'tries' column
        .drop("temp_index", "is_match")  # Drop temporary columns no longer needed
    )

    return result


def difficulty(dataf: pl.DataFrame) -> pl.DataFrame:
    """
    Assigns a difficulty level to each row in the input DataFrame based on the number of tries.

    Parameters:
    -----------
        - dataf (pl.DataFrame): A Polars DataFrame containing a column 'tries' with the number of attempts.

    Returns:
    --------
    pl.DataFrame: A new DataFrame with an additional 'difficulty' column indicating the difficulty level.
                  The levels are defined as:
                  - 'easy' for tries <= 2
                  - 'moderate' for 3 <= tries < 5
                  - 'hard' for tries >= 5
                  - 'very hard' for null values in 'tries'
    """
    return (
        dataf
        # Add a 'difficulty' column based on the number of tries
        .with_columns(
            difficulty=(
                pl.when(pl.col("tries") <= 2).then(pl.lit("easy"))  # Easy if tries <= 2
                .when((pl.col("tries") > 2) & (pl.col("tries") < 5)).then(pl.lit("moderate"))  # Moderate if 3 <= tries < 5
                .when(pl.col("tries") >= 5).then(pl.lit("hard"))  # Hard if tries >= 5
                .when(pl.col("tries").is_null()).then(pl.lit("very hard"))  # Very hard if tries is null
            )
        )
    )


def best_opening_word_from_best_couples(
    wordle_scored: Dict[str, Callable[[], pl.DataFrame]],
    five_letter_words_anagrams: pl.DataFrame,
) -> pl.DataFrame:
    """
    Determines the best opening word for a Wordle-like game based on scored guesses.

    Parameters:
    -----------
        - wordle_scored (Dict[str, Callable[[], pl.DataFrame]]): A dictionary where keys are dataset names 
          and values are callables that return corresponding Polars DataFrames with scored guesses.
        - five_letter_words_anagrams (pl.DataFrame): A DataFrame containing five-letter words and their anagrams.

    Returns:
    --------
    pl.DataFrame: The processed DataFrame with unique challenge words and best opening word suggestions.
    """
    
    def get_partition(partitioned_datasets: Dict[str, Callable[[], pl.DataFrame]]) -> pl.DataFrame:
        """
        Processes partitioned datasets to extract and concatenate guesses and their scores.

        Parameters:
        -----------
        partitioned_datasets (Dict[str, Callable[[], pl.DataFrame]]): A dictionary where keys are dataset 
        names and values are callables that return corresponding Polars DataFrames with scored guesses.

        Returns:
        --------
        pl.DataFrame: A concatenated DataFrame of first and second guesses with their patterns and scores.
        """
        data = []  # List to hold DataFrames from each partition
        
        for _, partition_val in tqdm(partitioned_datasets.items()):
            # Load the partition data
            df = partition_val()
            
            # Process the first guesses
            first_guess = (
                df
                .select(["challenge_word", "first_guess", "first_word_pattern", "first_word_score"])
                .rename({
                    "first_guess": "guess",
                    "first_word_pattern": "pattern",
                    "first_word_score": "score"
                })
            )
            data.append(first_guess)

            # Process the second guesses
            second_guess = (
                df
                .select(["challenge_word", "second_guess", "second_word_pattern", "second_word_score"])
                .rename({
                    "second_guess": "guess",
                    "second_word_pattern": "pattern",
                    "second_word_score": "score"
                })
            )
            data.append(second_guess)
        
        # Concatenate all DataFrames into a single DataFrame
        return pl.concat(data)

    # Get the concatenated and processed DataFrame from the partitioned datasets
    best_singles_from_best_couples = (
        get_partition(wordle_scored)
        .unique(subset=["challenge_word", "guess"])
        .rename({"challenge_word": "challenge"})
    )
    
    # Group by 'word', sum their scores, and sort by the summed scores in descending order
    best_opening_word_df = (
        best_singles_from_best_couples
        .group_by("guess")
        .agg(summed_scores=pl.col("score").sum().cast(pl.Int32))
        .sort("summed_scores", descending=True)
        .with_columns(
            abs_diff_from_top_score=(
                (pl.col("summed_scores") - pl.col("summed_scores").max())
                .abs()
                .cast(pl.UInt32)
            )
        )
        .join(
            five_letter_words_anagrams.select(["words", "anagrams"]),
            left_on="guess",
            right_on="words",
            coalesce=True
        )
        .rename({"guess": "word"})
        .sort("abs_diff_from_top_score", descending=False, maintain_order=True)
        .with_row_index(offset=1)
    )
    
    return best_singles_from_best_couples, best_opening_word_df


def simulating_top_words(
    word_length: int,
    weighted: bool,
    k: int,
    num_tries: int,
    mapping: Dict[str, Any],
    num_words_to_simulate: int,
    five_letter_words: pl.DataFrame,
    five_letter_words_anagrams: pl.DataFrame,
    best_singles: pl.DataFrame,
    raise_anagrams: List[str],
    salet_anagrams: List[str],
    soare_anagrams: List[str],
    trace_anagrams: List[str]
) -> pl.DataFrame:
    """
    Simulates top word guesses against a list of challenge words and collects data on the simulation.

    Parameters:
    -----------
        - word_length (int): The length of the words to consider for the challenge.
        - weighted (bool): Whether to sample guesses based on word frequency.
        - k (int): Number of possible words to consider for the next guess.
        - num_tries (int): Number of tries allowed for each word challenge.
        - mapping (Dict[str, Any]): Mapping of match pattern characters to their values.
        - num_words_to_simulate (int): Number of top words to simulate.
        - five_letter_words (pl.DataFrame): DataFrame containing words with their frequencies and lengths.
        - five_letter_words_anagrams (pl.DataFrame): A DataFrame containing five-letter words and their anagrams.
        - best_singles (pl.DataFrame): DataFrame containing the best single words.
        - raise_anagrams, salet_anagrams, soare_anagrams, trace_anagrams (List[str]): Lists of anagrams for specific words.

    Returns:
    --------
        - pl.DataFrame: DataFrame containing the simulation results.
    """
    
    # Combine all words to simulate including anagrams
    words_to_simulate = set(best_singles["word"].head(num_words_to_simulate))
    words_to_simulate.update(salet_anagrams)
    words_to_simulate.update(raise_anagrams)
    words_to_simulate.update(soare_anagrams)
    words_to_simulate.update(trace_anagrams)
    
    
    # Normalize word frequencies
    five_letter_words = (
        five_letter_words
        .select(["words", "word_freq", "word_length"])
        .filter(pl.col("word_length") == word_length)
        .sort("word_freq", descending=True)
    )

    challenge_words = five_letter_words["words"].shuffle(seed=42)
    
    data = []

    for challenge in tqdm(challenge_words):
        current_guess = None
        for g in words_to_simulate:
            encountered_guesses = set()
            updated_letters_to_omit = set()
            updated_letters_to_keep = {}
            matching_indexes = {}
            miss_matched_indexes = {}

            current_guess = g
            for i in range(num_tries):
                guess = g if i == 0 else current_guess

                # Add guess to current guesses
                encountered_guesses.add(guess)
                
                match_pattern = "".join(
                    "G" if x == y else "Y" if x != y and y in challenge else "B"
                    for x, y in zip(challenge, guess)
                )

                current_guess_match_pattern = [mapping[char] for char in match_pattern]
                
                mapped = {
                    letter: (pattern, index) for index, (letter, pattern)
                    in enumerate(zip(guess, current_guess_match_pattern))
                }

                # Update dictionaries
                updated_letters_to_omit.update(k for k, v in mapped.items() if v[0] == -1)
                updated_letters_to_keep.update({k: v for k, v in mapped.items() if v[0] != -1})
                matching_indexes.update({k: v[1] for k, v in mapped.items() if v[0] == 1})
                miss_matched_indexes.update({k: v[1] for k, v in mapped.items() if v[0] == 0})

                words_to_filter_out = [
                    word for word in challenge_words
                    if any(letter in word for letter in updated_letters_to_omit)
                    or any(word[position] == letter for letter, (pattern, position) in updated_letters_to_keep.items() if pattern == 0)
                    or not all(word[position] == letter for letter, (pattern, position) in updated_letters_to_keep.items() if pattern == 1)
                    or not all(letter in word for letter in updated_letters_to_keep.keys())
                    or word in encountered_guesses
                ]


                # Update encountered guesses with words not to be considered
                encountered_guesses.update(words_to_filter_out)
                
                # Filter remaining possible words, excluding encountered guesses
                remaining_words = five_letter_words.filter(~(pl.col("words").is_in(encountered_guesses))).to_pandas()
                
                if weighted:
                    try:
                        filtered_words = (
                            remaining_words
                            .sample(n=min(k, len(remaining_words)), weights="word_freq", random_state=42)
                            ["words"]
                        )
                    except ValueError:
                        filtered_words = (
                            remaining_words["words"]
                            .head(k)
                        )
                else:
                    filtered_words = remaining_words.head(k)["words"]

                num_choices_after_guess = len(remaining_words)

                encountered_guesses.add(current_guess)

                possible_guesses = ", ".join(filtered_words)

                challenge_in_possible_guesses = challenge in possible_guesses

                # Determine next guess and letter differences
                if guess == challenge:
                    next_guess = guess
                    letter_differences = ""
                    num_diff_letters = 0
                else:
                    next_guess = next(iter(filtered_words), random.choice(challenge_words))
                    letter_differences = "".join(updated_letters_to_omit)
                    num_diff_letters = len(updated_letters_to_omit)

                current_guess = next_guess

                # Collect simulation data
                temp: Dict[str, Any] = {
                    "challenge": challenge,
                    "guess": guess,
                    "match_pattern": match_pattern,
                    "letter_differences": letter_differences,
                    "common_letters": "".join(updated_letters_to_keep.keys()),
                    "num_diff_letters": num_diff_letters,
                    "num_common_letters": len(updated_letters_to_keep),
                    "num_matching_index": len(matching_indexes),
                    "num_choices_after_guess": num_choices_after_guess,
                    "possible_guesses": possible_guesses,
                    "challenge_in_possible_guesses": challenge_in_possible_guesses,
                    "next_guess": current_guess,
                }
                data.append(temp)
    
    return (
        pl.DataFrame(data)
        .with_columns(
            pl.col("num_diff_letters").cast(pl.UInt8),
            pl.col("num_common_letters").cast(pl.UInt8),
            pl.col("num_matching_index").cast(pl.UInt8),
            pl.col("num_choices_after_guess").cast(pl.UInt16),
            ((pl.col("challenge") == pl.col("guess")) & (pl.col("challenge") == pl.col("next_guess"))).alias("is_match")
        )
        .pipe(add_group, num_tries)
        .pipe(compute_tries)
        .pipe(difficulty)
        .join(
            five_letter_words_anagrams.select("words", "anagrams", "anagram_num"),
            how="left",
            left_on="guess",
            right_on="words",
            coalesce=True
        )
        .rename({"anagrams": "guess_word_anagrams"})
    )


def get_guess_stats(simulating_top_words: pl.DataFrame, five_letter_word_anagrams: pl.DataFrame) -> pl.DataFrame:
    first_guess = (
        simulating_top_words
        .select(
            "challenge", "guess", "letter_differences", "common_letters",
            "num_diff_letters", "num_common_letters", "num_matching_index",
            "match_pattern", "num_choices_after_guess", "possible_guesses",
            "challenge_in_possible_guesses", "next_guess", "group", "tries", "difficulty",
            "guess_word_anagrams", "anagram_num"
        )
        .join(
            five_letter_word_anagrams.select("words", "anagrams", "anagram_num"),
            left_on="next_guess",
            right_on="words",
            how="left",
            coalesce=True
        )
        .group_by("group", maintain_order=True)
        .first()
    )

    return (
        first_guess
        .group_by("guess")
        .agg(
            tries_mode=pl.col("tries").mode().cast(pl.UInt8),
            tries_mean=(pl.col("tries").mean().round(3).cast(pl.Float32)),
            tries_null_pct=(pl.col("tries").is_null().mean() * 100).round(3).cast(pl.Float32),
            avg_letters_identified=(pl.col("num_common_letters").mean().round(3).cast(pl.Float32)),
            avg_letters_matched=(pl.col("num_matching_index").mean().round(3).cast(pl.Float32)),
        )
        .explode("tries_mode")
        .join(
            (
                first_guess
                .select("guess_word_anagrams", "guess")
                .unique(subset="guess", keep="first")
            ),
            left_on="guess",
            right_on="guess"
        )
    )


def simulating_alternative_openers(
    weighted: bool,
    k: int,
    number_of_tries_alt_second_guess: int,
    mapping: Dict[str, str],
    simulating_top_words: pl.DataFrame,
    five_letter_words: pl.DataFrame,
    five_letter_words_anagrams: pl.DataFrame,
    best_singles: pl.DataFrame,
) -> pl.DataFrame:
    """
    Simulates the process of trying alternative opening guesses for a word challenge game.

    Parameters:
    - weighted (bool): Flag indicating whether to sample words weighted by frequency.
    - k (int): The number of top possible guesses to consider after each guess.
    - num_tries (int): The number of guesses allowed per challenge.
    - mapping (Dict[str, str]): A dictionary mapping match patterns to certain values.
    - simulating_top_words (pl.DataFrame): DataFrame of top words for simulation.
    - five_letter_words (pl.DataFrame): DataFrame containing five-letter words and their frequencies.
    - five_letter_words_anagrams (pl.DataFrame): DataFrame containing anagrams of five-letter words.
    - best_singles (pl.DataFrame): DataFrame of best single words for initial guesses.

    Returns:
    - pl.DataFrame: DataFrame containing the results of the simulation.
    """

    def best_guess_word(guess_stats: pl.DataFrame) -> str:
        return (
            guess_stats
            .select(["guess", "avg_letters_matched", "tries_mean"])
            .sort(["avg_letters_matched", "tries_mean"], descending=[True, False])
            .with_row_index(offset=1)
            ["guess"]
            .head(1)
            .item()
        )

    # Get guess statistics from the top simulating words and their anagrams
    guess_stats = get_guess_stats(simulating_top_words, five_letter_words_anagrams)

    # Determine the best guess word based on guess statistics
    best_guess = best_guess_word(guess_stats=guess_stats)

    best_guess_regex = f"[{best_guess}]"

    # Generate alternative second guesses by filtering out words containing any character from the best guess
    alternative_second_guesses = best_singles.filter(~pl.col("word").str.contains(best_guess_regex))["word"]

    # Pair the best guess with each alternative second guess
    alternative_openers = list(zip_longest([best_guess], alternative_second_guesses, fillvalue=best_guess))

    # Prepare the list of five-letter words sorted by frequency
    sorted_five_letter_words = (
        five_letter_words
        .select(["words", "word_freq", "word_length"])
        .sort("word_freq", descending=True)
    )

    # Shuffle the challenge words with a fixed seed for reproducibility
    challenge_words = five_letter_words["words"].shuffle(seed=42)

    data = []

    # Iterate through the challenge words
    for challenge in tqdm(challenge_words):
        # Use the first opener from the list
        for openers in alternative_openers:
            current_guess = openers[0]
            encountered_guesses = set()
            updated_letters_to_omit = set()
            updated_letters_to_keep = dict()
            matching_indexes = dict()
            miss_matched_indexes = dict()

            for _ in range(number_of_tries_alt_second_guess):
                guess = current_guess
                encountered_guesses.add(guess)

                # Generate match pattern for the current guess
                match_pattern = "".join(
                    "G" if x == y else "Y" if x != y and y in challenge else "B"
                    for x, y in zip(challenge, guess)
                )
                
                current_guess_match_pattern = [mapping[char] for char in match_pattern]

                # Map letters to their match patterns and positions
                mapped = {
                    letter: (pattern, index) for index, (letter, pattern)
                    in enumerate(zip(guess, current_guess_match_pattern))
                }

                # Update letters to omit, keep, and index information
                updated_letters_to_omit.update(k for k, v in mapped.items() if v[0] == -1)
                updated_letters_to_keep.update({k: v for k, v in mapped.items() if v[0] != -1})
                matching_indexes.update({k: v[1] for k, v in mapped.items() if v[0] == 1})
                miss_matched_indexes.update({k: v[1] for k, v in mapped.items() if v[0] == 0})

                # Filter out words that don't match the updated constraints
                words_to_filter_out = [
                    word for word in challenge_words
                    if any(letter in word for letter in updated_letters_to_omit)
                    or any(word[position] == letter for letter, (pattern, position) in updated_letters_to_keep.items() if pattern == 0)
                    or not all(word[position] == letter for letter, (pattern, position) in updated_letters_to_keep.items() if pattern == 1)
                    or not all(letter in word for letter in updated_letters_to_keep.keys())
                    or word in encountered_guesses
                ]

                # Update encountered guesses with words not to be considered
                encountered_guesses.update(words_to_filter_out)

                # Filter remaining possible words, excluding encountered guesses
                remaining_words = sorted_five_letter_words.filter(~(pl.col("words").is_in(encountered_guesses))).to_pandas()

                if weighted:
                    try:
                        # Sample words weighted by frequency
                        filtered_words = remaining_words.sample(n=min(k, len(remaining_words)), weights="word_freq")["words"]
                    except ValueError:
                        filtered_words = remaining_words["words"].head(k)
                else:
                    filtered_words = remaining_words.head(k)["words"]

                num_choices_after_guess = len(remaining_words)

                encountered_guesses.add(current_guess)

                possible_guesses = ", ".join(filtered_words)

                challenge_in_possible_guesses = challenge in possible_guesses

                # Determine the next guess
                if guess == challenge:
                    next_guess = guess
                    letter_differences = ""
                    num_diff_letters = 0
                else:
                    # Use the next opener if available, otherwise sample from filtered words
                    if guess in openers:
                        next_guess_index = openers.index(guess) + 1
                        next_guess = openers[next_guess_index] if next_guess_index < len(openers) else next(iter(filtered_words), random.choice(challenge_words))
                    else:
                        next_guess = next(iter(filtered_words), random.choice(challenge_words))

                if guess != challenge:
                    letter_differences = "".join(updated_letters_to_omit)
                    num_diff_letters = len(updated_letters_to_omit)
                
                current_guess = next_guess
                
                # Store simulation data
                temp: Dict[str, Any] = {
                    "challenge": challenge,
                    "guess": guess,
                    "match_pattern": match_pattern,
                    "letter_differences": letter_differences,
                    "common_letters": "".join(updated_letters_to_keep.keys()),
                    "num_diff_letters": num_diff_letters,
                    "num_common_letters": len(updated_letters_to_keep),
                    "num_matching_index": len(matching_indexes),
                    "num_choices_after_guess": num_choices_after_guess,
                    "possible_guesses": possible_guesses,
                    "challenge_in_possible_guesses": challenge_in_possible_guesses,
                    "next_guess": next_guess,
                }
                data.append(temp)

    # Create the final DataFrame with appropriate column types
    return (
        pl.DataFrame(data)
        .with_columns(
            pl.col("num_diff_letters").cast(pl.UInt8),
            pl.col("num_common_letters").cast(pl.UInt8),
            pl.col("num_matching_index").cast(pl.UInt8),
            pl.col("num_choices_after_guess").cast(pl.UInt16),
            ((pl.col("challenge") == pl.col("guess")) & (pl.col("challenge") == pl.col("next_guess"))).alias("is_match")
        )
        .pipe(add_group, number_of_tries_alt_second_guess)
        .pipe(compute_tries)
        .pipe(difficulty)
        .join(
            five_letter_words_anagrams.select("words", "anagrams", "anagram_num"),
            how="left",
            left_on="guess",
            right_on="words",
            coalesce=True
        )
        .rename({"anagrams": "guess_word_anagrams"})
    )


def alternative_openers(
    weighted: bool,
    k: int,
    num_tries: int,
    mapping: Dict[str, str],
    *dataframes: pl.DataFrame,
) -> pl.DataFrame:
    """
    Simulates alternative word openers for a word challenge game.

    Parameters:
    -----------
        weighted (bool): If True, words are sampled weighted by frequency.
        k (int): Number of words to consider after each guess.
        num_tries (int): Number of tries allowed for each challenge word.
        mapping (Dict[str, str]): Mapping of match pattern characters to numerical values.
        *dataframes: A variable number of Polars DataFrames in the following order:
            - five_letter_words: DataFrame containing words, frequencies, and lengths.
            - five_letter_words_anagrams: DataFrame containing words and their anagrams.
            - best_singles: DataFrame with the best initial guess word.
            - simulating_alternative_openers: DataFrame used to find the best second guess.

    Returns:
    --------
        pl.DataFrame: A Polars DataFrame containing the results of the simulation.

    Raises:
    -------
        ValueError: If the number of provided DataFrames is not exactly 4.
    """

    data = []

    # Unpack dataframes if the number of provided dataframes is known and fixed
    if len(dataframes) == 4:  # expecting exactly 3 dataframes
        (
            five_letter_words,
            five_letter_words_anagrams,
            best_singles,
            simulating_alternative_openers
        ) = dataframes
    else:
        raise ValueError("Expected exactly 4 dataframes")

    # Get the best word
    best_word = (
        best_singles
        ["word"]
        .head(1)
        .item()
    )

    # Get the best second word
    best_second_word = (
        simulating_alternative_openers
        .filter(pl.col("guess") != "soare")
        .group_by("guess")
        .agg(avg_min_choice_after_2nd_guess = pl.col("num_choices_after_guess").mean().cast(pl.Float32))
        .sort("avg_min_choice_after_2nd_guess", descending=False)
        ["guess"]
        .head(1)
        .item()
    )

    # Best alternative opening words
    openers = best_word, best_second_word

    # Prepare the list of five-letter words sorted by frequency
    sorted_five_letter_words = (
        five_letter_words
        .select(["words", "word_freq", "word_length"])
        .sort("word_freq", descending=True)
    )

    challenge_words = five_letter_words["words"].shuffle(seed=42)

    for challenge in tqdm(challenge_words):
        # Use the first opener from the list
        current_guess = openers[0]
        encountered_guesses = set()
        updated_letters_to_omit = set()
        updated_letters_to_keep = dict()
        matching_indexes = dict()
        miss_matched_indexes = dict()

        for _ in range(num_tries):
            guess = current_guess
            encountered_guesses.add(guess)

            # Generate match pattern for the current guess
            match_pattern = "".join(
                "G" if x == y else "Y" if x != y and y in challenge else "B"
                for x, y in zip(challenge, guess)
            )
            
            current_guess_match_pattern = [mapping[char] for char in match_pattern]

            # Map letters to their match patterns and positions
            mapped = {
                letter: (pattern, index) for index, (letter, pattern)
                in enumerate(zip(guess, current_guess_match_pattern))
            }

            # Update letters to omit, keep, and index information
            updated_letters_to_omit.update(k for k, v in mapped.items() if v[0] == -1)
            updated_letters_to_keep.update({k: v for k, v in mapped.items() if v[0] != -1})
            matching_indexes.update({k: v[1] for k, v in mapped.items() if v[0] == 1})
            miss_matched_indexes.update({k: v[1] for k, v in mapped.items() if v[0] == 0})

            # Filter out words that don't match the updated constraints
            words_to_filter_out = [
                word for word in challenge_words
                if any(letter in word for letter in updated_letters_to_omit)
                or any(word[position] == letter for letter, (pattern, position) in updated_letters_to_keep.items() if pattern == 0)
                or not all(word[position] == letter for letter, (pattern, position) in updated_letters_to_keep.items() if pattern == 1)
                or not all(letter in word for letter in updated_letters_to_keep.keys())
                or word in encountered_guesses
            ]

            # Update encountered guesses with words not to be considered
            encountered_guesses.update(words_to_filter_out)

            # Filter remaining possible words, excluding encountered guesses
            remaining_words = sorted_five_letter_words.filter(~(pl.col("words").is_in(encountered_guesses))).to_pandas()
            # remaining_words = sorted_five_letter_words.filter(~(pl.col("words").is_in(words_to_filter_out))).to_pandas()

            if weighted:
                try:
                    # Sample words weighted by frequency
                    filtered_words = remaining_words.sample(n=min(k, len(remaining_words)), weights="word_freq")["words"]
                except ValueError:
                    filtered_words = remaining_words["words"].head(k)
            else:
                filtered_words = remaining_words.head(k)["words"]

            num_choices_after_guess = len(remaining_words)

            encountered_guesses.add(current_guess)

            possible_guesses = ", ".join(filtered_words)

            challenge_in_possible_guesses = challenge in possible_guesses

            # Determine the next guess
            if guess == challenge:
                next_guess = guess
                letter_differences = ""
                num_diff_letters = 0
            else:
                # Use the next opener if available, otherwise sample from filtered words
                if guess in openers:
                    next_guess_index = openers.index(guess) + 1
                    next_guess = openers[next_guess_index] if next_guess_index < len(openers) else next(iter(filtered_words), random.choice(challenge_words))
                else:
                    next_guess = next(iter(filtered_words), random.choice(challenge_words))

            if guess != challenge:
                letter_differences = "".join(updated_letters_to_omit)
                num_diff_letters = len(updated_letters_to_omit)
            
            current_guess = next_guess
            
            # Store simulation data
            temp: Dict[str, Any] = {
                "challenge": challenge,
                "guess": guess,
                "match_pattern": match_pattern,
                "letter_differences": letter_differences,
                "common_letters": "".join(updated_letters_to_keep.keys()),
                "num_diff_letters": num_diff_letters,
                "num_common_letters": len(updated_letters_to_keep),
                "num_matching_index": len(matching_indexes),
                "num_choices_after_guess": num_choices_after_guess,
                "possible_guesses": possible_guesses,
                "challenge_in_possible_guesses": challenge_in_possible_guesses,
                "next_guess": next_guess,
            }
            data.append(temp)

    # Create the final DataFrame with appropriate column types
    return (
        pl.DataFrame(data)
        .with_columns(
            pl.col("num_diff_letters").cast(pl.UInt8),
            pl.col("num_common_letters").cast(pl.UInt8),
            pl.col("num_matching_index").cast(pl.UInt8),
            pl.col("num_choices_after_guess").cast(pl.UInt16),
            ((pl.col("challenge") == pl.col("guess")) & (pl.col("challenge") == pl.col("next_guess"))).alias("is_match")
        )
        .pipe(add_group, num_tries)
        .pipe(compute_tries)
        .pipe(difficulty)
        .join(
            five_letter_words_anagrams.select("words", "anagrams", "anagram_num"),
            how="left",
            left_on="guess",
            right_on="words",
            coalesce=True
        )
        .rename({"anagrams": "guess_word_anagrams"})
    )