import polars as pl
import requests
import spacy
import inflect
from io import StringIO
from collections import defaultdict
from typing import List, Dict
from tqdm import tqdm
from spacy.language import Language
from spacy.tokens import Token
from nltk.corpus import stopwords as nltk_stopwords

# Load spaCy
cls = spacy.util.get_lang_class("en")
cls.Defaults.stop_words.remove("least")
nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])

# Load NLTK stopwords
nltk_stop_words = set(nltk_stopwords.words("english"))

# Update spaCy's stopwords list with NLTK stopwords
for stopword in nltk_stop_words:
    nlp.Defaults.stop_words.add(stopword)
    nlp.vocab[stopword].is_stop = True


def check_anagrams(dataf: pl.DataFrame) -> Dict[str, List[str]]:
    """
    Identify anagrams from a DataFrame and sort them by frequency. If words have the same frequency,
    they are sorted alphabetically. The function returns a dictionary where the key is the highest 
    frequency anagram (or alphabetically first if frequencies are the same) and the value is a list 
    of all anagrams as a comma-separated string.

    Parameters:
    -----------
    dataf (pl.DataFrame): A DataFrame with columns 'words' (List[str]) and 'word_freq' (List[int]) 
                          representing the words and their respective frequencies.

    Returns:
    --------
    Dict[str, List[str]]: A dictionary with the highest frequency anagram as the key and a 
                          comma-separated string of all anagrams as the value.
    """
    # Create a dictionary to store anagrams
    anagrams = defaultdict(list)
    
    # Iterate over each word and its frequency in the DataFrame
    for word, word_freq in zip(dataf["words"], dataf["word_freq"]):
        # Define a signature for the word (sorted characters)
        signature = "".join(sorted(word))
        
        # Append the word and its frequency to the list of anagrams associated with its signature
        anagrams[signature].append((word, word_freq))
    
    # Create a dictionary to store the final anagrams
    anagrams_dict = {}
    
    for signature, word_freq_list in anagrams.items():
        # Sort the list first by frequency (descending) and then alphabetically if frequencies are the same
        word_freq_list.sort(key=lambda x: (-x[1], x[0]))
        
        # Get the anagram with the highest frequency (or alphabetically first if frequencies are the same)
        anagram_word_freq, _ = word_freq_list[0]
        
        # Store the anagram with all words that match the signature
        anagrams_dict[anagram_word_freq] = [", ".join(word for word, _ in word_freq_list)]
    
    return anagrams_dict


def google_books_words(
    num_volumes: int,
    response_code: int,
    most_common_letters: str,
    google_books_url: str
) -> pl.DataFrame:
    """
    Fetch and process word frequency data from Google Books.

    This function retrieves words curated from Google Books by Peter Norvig, processes it, 
    and returns a DataFrame containing word frequency statistics.

    Parameters:
    -----------
        - "num_volumes" (int): The number of volumes to use for normalization.
        - "response_code" (int): The expected HTTP response code for a successful request.
        - "most_common_letters" (List[str]): A list of letters to check for commonality in words.
        - "google_books_url" (str): The URL to fetch the Google Books data from.

    Returns:
    --------
    pl.DataFrame
        A DataFrame containing the following columns:
        - "words" (str): The words from the Google Books dataset.
        - "count" (int): The frequency count of each word.
        - "word_freq" (float): The normalized frequency of each word.
        - "common_letters" (bool): Whether each word contains only the most common letters.
        - "word_length" (int): The length of each word.
        - "has_repeat_letters" (bool): Whether each word contains repeated letters.
    
    Exceptions:
    -----------
    If an error occurs during the request or processing, an exception is printed and an empty DataFrame is returned.
    """

    response = requests.get(google_books_url)
    try:
        if response.status_code == response_code:
            # Use StringIO to create a file-like object from the text content
            file_content = StringIO(response.text)
            # Load the data into a DataFrame
            google_books = (
                pl.read_csv(
                    file_content,
                    separator = "\t",
                    has_header = False,
                )
                .rename({"column_1": "words", "column_2": "count"})
                .with_columns(
                    pl.col("words").str.strip_chars().str.to_lowercase(),
                )
            )

            # Calculate the total count
            total_count = google_books.select(pl.col("count").sum()).item()
            
            # google_books = (
            #     google_books
            #     .with_columns(
            #         pl.col("words").str.to_lowercase().str.strip_chars(),
            #         word_freq = ((pl.col("count") / num_volumes) / total_count).cast(pl.Float32), # https://aclanthology.org/P12-3029.pdf,
            #         common_letters = (
            #             pl.col("words")
            #             .map_elements(lambda word: all(letter in most_common_letters for letter in word), return_dtype=pl.Boolean)
            #         ),
            #         word_length = (pl.col("words").str.len_chars()).cast(pl.UInt8),
            #         has_repeat_letters = (
            #             pl.col("words")
            #             .map_elements(lambda word: len(set(word)) < len(word), return_dtype=pl.Boolean)
            #         )
            #     )  
            # )

            google_books = (
                google_books
                .with_columns(
                    word_freq = ((pl.col("count") / num_volumes) / total_count).cast(pl.Float32), # https://aclanthology.org/P12-3029.pdf,
                    word_length = (pl.col("words").str.len_chars()).cast(pl.UInt8),
                    num_common_letters=(
                        pl.col("words")
                        .str.split("")
                        .list.eval(pl.element().is_in(list(most_common_letters)))
                        .list.sum()
                    ).cast(pl.UInt8),
                    num_unique_letters=(
                        pl.col("words")
                        .str.split("")
                        .list.unique()
                        .list.len()
                    ).cast(pl.UInt8),
                )
                .with_columns(
                    common_letters=(pl.col("word_length") == pl.col("num_common_letters")),
                    has_repeat_letters=(pl.col("word_length") != pl.col("num_unique_letters"))
                ) 
            )
    except Exception as e:
        # Handle exceptions (e.g., network errors, invalid URLs, etc.)
        print(f"An error occurred: {e}")
        google_books = pl.DataFrame()  # Return an empty DataFrame in case of an error
    return google_books


def get_english_words(
    response_code: int,
    most_common_letters: str,
    word_freq_fill_null: int,
    english_words_url: str,
    google_books: pl.DataFrame
) -> pl.DataFrame:
    """
    Fetches English words from a specified URL and enriches them with additional information.

    Parameters:
    -----------
        - response_code (int): The expected HTTP response code from the URL.
        - most_common_letters (str): String containing the most common letters to check against.
        - word_freq_fill_null (float): Fill value for word frequency if null.
        - english_words_url (str): URL to fetch the English words from.
        - google_books (pl.DataFrame): DataFrame containing words and their frequency.

    Returns:
    --------
    pl.DataFrame: DataFrame containing English words with additional columns:
        - words: The English words fetched from the URL.
        - word_freq: Frequency of words obtained from 'google_books' DataFrame.
        - common_letters: Boolean indicating if the word contains only the most common letters.
        - word_length: Length of each word.
        - has_repeat_letters: Boolean indicating if the word has repeat letters.

    Raises:
    -------
        Exception: Any exceptions that occur during the process, such as network errors or invalid URLs.
    """

    # Fetch data from URL
    response = requests.get(english_words_url)
    
    try:
        # Check if response is successful
        if response.status_code == response_code:
            # Use StringIO to create a file-like object from the text content
            file_content = StringIO(response.text)
            # Load the data into a DataFrame
            english_words = (
                pl.read_csv(
                    file_content,
                    separator=",",
                    has_header=False,
                )
                .rename({"column_1": "words"})
                .with_columns(pl.col("words").str.to_lowercase().str.strip_chars())
                .join(
                    google_books.select(["words", "word_freq"]),
                    how="left",
                    left_on="words",
                    right_on="words",
                    coalesce=True
                )
                # .with_columns(
                #     pl.col("word_freq").fill_null(word_freq_fill_null),
                #     common_letters=(
                #         pl.col("words")
                #         .map_elements(lambda word: all(letter in most_common_letters for letter in word), return_dtype=pl.Boolean)
                #     ),
                #     word_length=(pl.col("words").str.len_chars()).cast(pl.UInt8),
                #     has_repeat_letters=(
                #         pl.col("words")
                #         .map_elements(lambda word: len(set(word)) < len(word), return_dtype=pl.Boolean)
                #     )
                # )
                .with_columns(
                    pl.col("word_freq").fill_null(word_freq_fill_null),
                    word_length = (pl.col("words").str.len_chars()).cast(pl.UInt8),
                    num_common_letters=(
                        pl.col("words")
                        .str.split("")
                        .list.eval(pl.element().is_in(list(most_common_letters)))
                        .list.sum()
                    ).cast(pl.UInt8),
                    num_unique_letters=(
                        pl.col("words")
                        .str.split("")
                        .list.unique()
                        .list.len()
                    ).cast(pl.UInt8),
                )
                .with_columns(
                    common_letters=(pl.col("word_length") == pl.col("num_common_letters")),
                    has_repeat_letters=(pl.col("word_length") != pl.col("num_unique_letters"))
                ) 
            )
    except Exception as e:
        # Handle exceptions (e.g., network errors, invalid URLs, etc.)
        print(f"An error occurred: {e}")
    
    return english_words


def processed_five_letter_words(
    word_length: int,
    english_words: pl.DataFrame,
) -> pl.DataFrame:
    """
    Processes and filters a DataFrame of English words, lemmatizing each word and returning
    a DataFrame of unique five-letter lemmatized words.

    Parameters:
    -----------
    word_length : int
        The length of words to filter from the DataFrame.
    english_words : pl.DataFrame
        A DataFrame containing English words with a column named 'words' and 'word_length'.

    Returns:
    --------
    pl.DataFrame
        A DataFrame of unique lemmatized words of the specified length.
    """
    
    def lemmatize_word(word: str) -> str:
        """
        Lemmatizes a word using spaCy, ensuring it is lowercase and meets certain criteria.

        Parameters:
        -----------
        word : str
            The word to be lemmatized.

        Returns:
        --------
        str or None
            The lemmatized word if it meets the criteria, else None.
        """
        # Lowercase the word
        word = word.lower() if isinstance(word, str) else None
        if word:
            doc = nlp(word)
            for token in doc:
                if (
                    not token.is_stop  # Remove stop words
                    and token.is_alpha  # Keep only alphabetic tokens
                    and not any(char.isdigit() for char in token.text)  # Remove words with digits
                ):
                    return token.lemma_
        else:
            return None
        
    # lemmatize words
    words_len_five = (english_words.filter(pl.col("word_length") == word_length))
    lemmatized_words = [
        lemma_word for word in tqdm(words_len_five["words"])
        if (lemma_word := lemmatize_word(word)) and len(lemma_word) == word_length
    ]

    # convert plural words to their singular form
    # Initialize the inflect engine
    p = inflect.engine()
    singular_lemmatized_words = [
        singular_word if (singular_word := p.singular_noun(word)) and len(singular_word) == word_length else word
        for word in lemmatized_words if len(word) == word_length
    ]

    # Return a DataFrame of unique processed words
    return (
        english_words
        .filter(pl.col("words").is_in(singular_lemmatized_words))
        .drop_nulls(subset="words")
        .unique(subset="words", keep="first")
    )


def five_letter_words_anagrams(
    english_words: pl.DataFrame,
    five_letter_words: pl.DataFrame,
) -> pl.DataFrame:
    """
    Generate a DataFrame of five-letter words and their anagrams, including word frequency.

    This function takes in two DataFrames, one containing a list of English words with their frequencies
    and another containing a list of five-letter words. It then generates a DataFrame with each five-letter word,
    its corresponding anagrams, the number of anagrams, and the word frequency.

    Parameters:
    -----------
        - english_words (pl.DataFrame): A DataFrame containing English words with their frequencies.
        - five_letter_words (pl.DataFrame): A DataFrame containing five-letter words.

    Returns:
    --------
        pl.DataFrame: A DataFrame with columns 'words', 'anagrams', 'anagram_num', and 'word_freq', 
                      where 'words' are the five-letter words, 'anagrams' are their corresponding anagrams, 
                      'anagram_num' is the count of anagrams for each word, and 'word_freq' is the frequency 
                      of the words in the English language.
    """
    
    # Create five letter dict
    five_letter_words_anagrams_dict = check_anagrams(five_letter_words)
    
    # Create an anagram dataframe
    five_letter_words_anagrams_df = (
        (
            pl.DataFrame(list({k: ", ".join(v) for k, v in five_letter_words_anagrams_dict.items()}.items()))
            .transpose()
            .rename({"column_0": "words", "column_1": "anagrams"})
            .with_columns(
                anagram_num = pl.col("anagrams").str.split(", ").list.len().cast(pl.UInt8)
            ) 
        )
        .join(
            english_words,
            on="words",
            how="left",
            coalesce=True
        )
        .unique(subset=["words"], keep="first")
        .sort(pl.col("word_freq"), descending=True)
    )
    return five_letter_words_anagrams_df
