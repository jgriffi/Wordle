# Project Functions Overview

## check_anagrams

Identify and return the required frequency anagram for each signature from a DataFrame.

This function processes a DataFrame containing words and their frequencies, identifies anagrams by sorting the characters of each word to form a signature, and then determines the anagram with the highest frequency for each signature. The result is a dictionary where the keys are the anagrams with the highest frequency and the values are lists of all anagrams associated with the same signature.

**Parameters:**

- `dataf` (pl.DataFrame): A Polars DataFrame with two columns:
  - `words`: Column containing words (strings).
  - `word_freq`: Column containing the corresponding word frequencies (integers).

**Returns:**

- `Dict[str, List[str]]`: A dictionary where each key is the anagram with the required frequency for a given signature, and the value is a list of all words that share the same signature.

## google_books_words

Fetch and process word frequency data from Google Books.

This function retrieves words curated from Google Books by Peter Norvig, processes it, and returns a DataFrame containing word frequency statistics.

**Parameters:**

- `num_volumes` (int): The number of volumes to use for normalization.
- `response_code` (int): The expected HTTP response code for a successful request.
- `most_common_letters` (List[str]): A list of letters to check for commonality in words.
- `google_books_url` (str): The URL to fetch the Google Books data from.

**Returns:**

- `pl.DataFrame`: A DataFrame containing the following columns:
  - `words` (str): The words from the Google Books dataset.
  - `count` (int): The frequency count of each word.
  - `word_freq` (float): The normalized frequency of each word.
  - `common_letters` (bool): Whether each word contains only the most common letters.
  - `word_length` (int): The length of each word.
  - `has_repeat_letters` (bool): Whether each word contains repeated letters.

**Exceptions:**

If an error occurs during the request or processing, an exception is printed and an empty DataFrame is returned.

## get_english_words

Fetches English words from a specified URL and enriches them with additional information.

**Parameters:**

- `response_code` (int): The expected HTTP response code from the URL.
- `most_common_letters` (str): String containing the most common letters to check against.
- `word_freq_fill_null` (float): Fill value for word frequency if null.
- `english_words_url` (str): URL to fetch the English words from.
- `google_books` (pl.DataFrame): DataFrame containing words and their frequency.

**Returns:**

- `pl.DataFrame`: DataFrame containing English words with additional columns:
  - `words`: The English words fetched from the URL.
  - `word_freq`: Frequency of words obtained from 'google_books' DataFrame.
  - `common_letters`: Boolean indicating if the word contains only the most common letters.
  - `word_length`: Length of each word.
  - `has_repeat_letters`: Boolean indicating if the word has repeat letters.

**Raises:**

- `Exception`: Any exceptions that occur during the process, such as network errors or invalid URLs.

## processed_five_letter_words

Processes a DataFrame of English words to filter and return unique five-letter words that are not stopwords, not in the specified POS tags, and are valid lemmas.

**Parameters:**

- `word_length` (int): The target word length to filter words (typically 5 for five-letter words).
- `nltk_pos_tags` (List[str]): List of POS tags to exclude from the final list of words.
- `nltk_custom_lemmas` (Dict[str, Any]): A dictionary mapping custom lemmatization rules where the keys are words and the values are their respective lemmas.
- `english_words` (pl.DataFrame): A DataFrame containing English words and their attributes.

**Returns:**

- `pl.DataFrame`: A DataFrame containing unique processed words that are five letters long, lemmatized, and filtered based on the specified criteria.

## five_letter_words_anagrams

Generates a DataFrame of five-letter words and their anagrams, including the frequency of each word.

**Parameters:**

- `english_words` (pl.DataFrame): DataFrame containing English words with their frequencies.
- `five_letter_words` (pl.DataFrame): DataFrame of processed five-letter words.

**Returns:**

- `pl.DataFrame`: DataFrame with columns 'words', 'anagrams', 'anagram_num', and 'word_freq', sorted by word frequency in descending order.

The function performs the following steps:

1. Generates a dictionary of five-letter words and their anagrams using the `check_anagrams` function.
2. Converts the dictionary to a DataFrame, transposes it, and renames the columns.
3. Adds a column for the number of anagrams for each word.
4. Joins the resulting DataFrame with the input DataFrame of English words to include word frequency.
5. Ensures each word is unique and sorts the DataFrame by word frequency in descending order.
```