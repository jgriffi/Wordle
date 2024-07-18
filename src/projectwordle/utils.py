import polars as pl
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Callable, Union, Tuple, List


# def color_pattern_matching(
#     dataf: pl.DataFrame,
#     challenge_col: str,
#     guess_col: str
# ) -> List[str]:
#     """
#     Calculate colored and bold patterns for pairs of words in a DataFrame.

#     The function calculates a pattern for each pair of words based on certain conditions:
#     - "<span style='color: green; font-weight: bold;'>G</span>" 
#       if the characters in the same positions are equal.
#     - "<span style='color: yellow; font-weight: bold;'>Y</span>"
#       if the characters in the same positions are not equal but present in the other word.
#     - "<span style='color: black; font-weight: bold;'>B</span>"
#       if the characters in the same positions are not equal and not present in the other word.

#     Parameters:
#     -----------
#     dataf (pl.DataFrame): The DataFrame containing the words.
#     challenge_col (str, optional): The column name for challenge words. Defaults to "challenge".
#     guess_col (str, optional): The column name for guess words. Defaults to "guess".

#     Returns:
#     --------
#     List[str]: A list of colored and bold patterns for each pair of words.
        
        
#     # Yellow alternatives
#     # ochre = #CC7722
#     # banana = #FFE135
#     # tuscany = #FCD12A
#     # cyber yellow = #FFD300
#     # golden yellow = #FFDF00
#     # mango = #FDBE02
#     """
#     all_patterns = [
#         "".join([
#             f"<span style='color: green; font-weight: bold;'>G</span>" if c2 == c1
#             else f"<span style='color: #FDBE02; font-weight: bold;'>Y</span>" if c2 != c1 and c2 in word1
#             else f"<span style='color: black; font-weight: bold;'>B</span>"
#             for c1, c2 in zip(word1, word2)
#         ])
#         for word1, word2 in zip(dataf[challenge_col], dataf[guess_col])
#     ]
#     return all_patterns


def color_pattern_matching(
    dataf: pl.DataFrame,
    challenge_col: str,
    guess_col: str
) -> List[str]:
    """
    Calculate colored patterns for pairs of words in a DataFrame.

    The function calculates a pattern for each pair of words based on certain conditions:
    - A green tile if the characters in the same positions are equal.
    - A yellow tile if the characters in the same positions are not equal but present in the other word.
    - A black tile if the characters in the same positions are not equal and not present in the other word.

    Parameters:
    -----------
    dataf (pl.DataFrame): The DataFrame containing the words.
    challenge_col (str, optional): The column name for challenge words. Defaults to "challenge".
    guess_col (str, optional): The column name for guess words. Defaults to "guess".

    Returns:
    --------
    List[str]: A list of colored patterns for each pair of words.
    """
    tile_style = "display:inline-block; width:20px; height:20px; margin:1px; text-align:center; font-weight:bold;"

    all_patterns = [
        "".join([
            f"<div style='{tile_style} background-color: green;'>{c2}</div>" if c2 == c1
            else f"<div style='{tile_style} background-color: #FDBE02;'>{c2}</div>" if c2 != c1 and c2 in word1
            else f"<div style='{tile_style} background-color: black;'>{c2}</div>"
            for c1, c2 in zip(word1, word2)
        ])
        for word1, word2 in zip(dataf[challenge_col], dataf[guess_col])
    ]
    return all_patterns



def plot_guess_stats(
    dataf: pl.DataFrame,
    x_axis_data: str,
    y_axis_data: str,
    words: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str
) -> None:
    """
    Create and display a scatter plot using Plotly.

    Parameters:
    -----------
    - dataf (pl.DataFrame): DataFrame to plot.
    - x_axis_data (str): X-axis data column name.
    - y_axis_data (str): Y-axis data column name.
    - words (str): Color data based on a specified column.
    - title (str): Title for the scatter plot.
    - xaxis_title (str): Label for the X-axis.
    - yaxis_title (str): Label for the Y-axis.

    Returns:
    None

    This function creates a scatter plot using Plotly with the provided data and parameters.
    It customizes the appearance, labels, and legend for the plot and displays it.
    """

    # dataf = dataf.to_pandas()

    # Create a scatter plot using Plotly
    fig = px.scatter(
        dataf,
        x=x_axis_data,
        y=y_axis_data,
        color=words,
        title=title
    )

    # Update the appearance of the markers in the scatter plot
    fig.update_traces(marker=dict(size=12, opacity=0.8))

    # Update the labels and titles for the X and Y axes
    fig.update_xaxes(title_text=xaxis_title)
    fig.update_yaxes(title_text=yaxis_title)

    # Center the plot title horizontally and set the figure size
    fig.update_layout(
        title_x=0.5,
        showlegend=False,  # Hide the legend
        width=800,         # Set the width of the figure
        height=600         # Set the height of the figure
    )

    # Show the plot
    fig.show()


def plot_challenge_stats(
    dataf: pl.DataFrame,
    x_axis_data: str,
    y_axis_data: str,
    words: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str
) -> None:
    """
    Create an interactive scatter plot to visualize challenge statistics.

    This function generates an interactive scatter plot using Plotly to visualize
    challenge statistics. It displays data points with custom colors based on a
    provided 'words' column, providing visual insights into the data.

    Parameters:
    -----------
    - dataf (pl.DataFrame): A Pandas DataFrame containing the data.
    - x_axis_data (str): The column name for the x-axis data.
    - y_axis_data (str): The column name for the y-axis data.
    - words (str): The column name for challenge words used for custom coloring.
    - title (str): The title of the plot.
    - xaxis_title (str): The title of the x-axis.
    - yaxis_title (str): The title of the y-axis.

    Returns:
    - None: This function displays the interactive plot but doesn't return a value.
    """
    # Plotting data
    x = dataf[x_axis_data]
    y = dataf[y_axis_data]
    word_col = dataf[words]
    
    # Create a custom color mapping for challenge words
    word_colors = {
        word: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
        for i, word in enumerate(word_col)
    }

    # Get the colors for the challenge words by mapping the dictionary
    word_colors_list = [word_colors[word] for word in word_col]

    # Create a scatter plot with Plotly using the custom colors
    fig = go.Figure(data=go.Scattergl(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            size=12,
            opacity=0.8,
            color=word_colors_list,  # Use the mapped colors
            showscale=False,         # Disable color scale legend
        ),
        text=word_col,               # Text to display when hovering
    ))

    # Customize the layout of the plot
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title_x=0.5,  # Center the plot title horizontally
        width=800,    # Set the width of the figure
        height=600    # Set the height of the figure
    )

    # Add interactivity with hover information
    fig.update_traces(marker=dict(size=12, opacity=0.8), selector=dict(mode="markers+text"))

    # Show the plot
    fig.show()
    

def difficulty_distribution(dataf: pl.DataFrame) -> None:
    """
    Generates a bar chart showing the distribution of difficulty levels in the provided DataFrame.

    Parameters:
    -----------
    dataf (pl.DataFrame): A Polars DataFrame containing a 'group' column and a 'difficulty' column.

    Returns:
    --------
    None: The function displays a Plotly bar chart of the difficulty distribution.
        
    The chart will:
        - Use a custom order for the difficulty categories: ["easy", "moderate", "hard", "very hard"].
        - Apply specific hex colors for each difficulty level:
            - easy: #1f77b4
            - moderate: #ff7f0e
            - hard: #2ca02c
            - very hard: #d62728
        - Center the chart title horizontally.
    """

    # Define the custom category order
    category_orders = {"difficulty": ["easy", "moderate", "hard", "very hard"]}

    # Define custom hex colors for each category
    custom_colors = {
        "easy": "#1f77b4",
        "moderate": "#ff7f0e",
        "hard": "#2ca02c",
        "very hard": "#d62728"
    }

    # Create difficulty dataframe
    difficulty_counts = (
        dataf
        # .unique(subset=["group"], keep="first")
        .group_by("group")
        .first()
        .select("difficulty")
        .to_pandas()
        .value_counts()
        .reset_index(name='count')
        .rename(columns={"index": "difficulty"})
    )

    # Create a bar chart using Plotly
    fig = px.bar(
        difficulty_counts,
        x="difficulty",
        y="count",
        title="Difficulty Distribution",
        color="difficulty",
        color_discrete_map=custom_colors
    )

    # Specify the custom category order for the x-axis
    fig.update_xaxes(
        title_text="Difficulty",
        categoryorder="array",
        categoryarray=category_orders["difficulty"]
    )
    fig.update_yaxes(title_text="Count")

    # Center the title horizontally and set figure size
    fig.update_layout(
        title_x=0.5,
        width=800,  # Set the width of the figure
        height=600  # Set the height of the figure
    )

    # Show the plot
    fig.show()


def plot_most_difficult_words(dataf: pl.DataFrame) -> None:
    """
    Generate and display a bar chart of the top 100 most difficult challenge words.

    This function creates a bar chart using Plotly to visualize the percentage of incomplete games 
    for the top 100 most difficult challenge words. It utilizes the default Seaborn color palette 
    for custom coloring of the bars.

    Parameters:
    -----------
    dataf (pl.DataFrame): A Polars DataFrame containing the challenge words and their corresponding 
                          incomplete game percentages.

    Returns:
    --------
    None: This function displays the interactive plot but does not return a value.
    """

    # Get the default Seaborn color palette and convert it to hex codes for custom coloring
    default_palette = sns.color_palette()
    hex_colors = default_palette.as_hex()

    dataf = dataf.to_pandas()

    # Extract the top 100 difficult challenge words
    incomplete_game_words = dataf["challenge"].head(100)

    # Extract the top 100 incomplete game percentages
    incomplete_game_pct = dataf["incomplete_games_pct"].head(100)

    # Create a bar chart using Plotly with the extracted data and custom colors
    fig = px.bar(
        dataf,
        x=incomplete_game_words,
        y=incomplete_game_pct,
        title="Top 100 most difficult challenge words",
        labels={"x": "Challenge", "y": "Incomplete Percent"},
        color=incomplete_game_words,
        color_discrete_sequence=hex_colors,  # Apply the custom colors
    )

    # Update the layout of the plot, setting axis titles and other properties
    fig.update_layout(
        xaxis_title_text="Challenge",
        yaxis_title_text="Incomplete Percent",
        title_x=0.5,         # Center the title horizontally
        showlegend=False,    # Hide the legend
        width=1000,          # Set the width of the figure
        height=600           # Set the height of the figure
    )

    # Show the plot
    fig.show()


def download_list(url: str, file_path: str) -> None:
    """
    Downloads content from the given URL and saves it to a specified file.

    Parameters:
    -----------
        - url (str): The URL to download the content from.
        - file_path (str): The file path where the content will be saved.

    Returns:
    --------
        None

    Raises:
    -------
        requests.exceptions.RequestException: If an error occurs during the HTTP request.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors
        
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)
        
        print(f"Content successfully saved to {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


def read_txt_to_polars(file_path: str) -> pl.DataFrame:
    """
    Reads a text file containing a list of words (one per line) into a Polars DataFrame.

    Parameters:
    -----------
        - file_path (str): The path to the text file.

    Returns:
    --------
        pl.DataFrame: A Polars DataFrame containing the words.
    """
    try:
        # Read the file content
        with open(file_path, "r", encoding="utf-8") as file:
            words = file.readlines()

        # Strip newline characters from each line
        words = [word.strip() for word in words]

        # Create a Polars DataFrame
        df = pl.DataFrame({
            "words": words
        })

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def plot_guess_stats_highlighted_words(
    dataf: pl.DataFrame,
    x_axis_data: str,
    y_axis_data: str,
    words: list[str],
    title: str,
    xaxis_title: str,
    yaxis_title: str
) -> None:
    # Convert Polars DataFrame to Pandas DataFrame for operability with Plotly
    dataf = dataf.to_pandas()

    # Create a new column to indicate label based on specific words
    def get_highlight_label(word):
        return word if word in words else "other"
    
    dataf["highlight_label"] = dataf["guess"].apply(get_highlight_label)

    # Generate perceptually distinct colors for the words using a colormap
    num_colors = len(words)
    cmap = plt.get_cmap("tab20", num_colors)  # Use a perceptually uniform colormap
    colors = [cmap(i) for i in range(num_colors)]
    color_discrete_map = {word: f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})" for word, (r, g, b, _) in zip(words, colors)}
    color_discrete_map["other"] = "rgb(169, 169, 169)"  # Add color for "other"

    # Create a scatter plot using Plotly
    fig = px.scatter(
        dataf,
        x=x_axis_data,
        y=y_axis_data,
        color="highlight_label",  # Use the new column for coloring
        title=title,
        color_discrete_map=color_discrete_map  # Use the generated color map
    )

    # Update the appearance of the markers in the scatter plot
    fig.update_traces(marker=dict(size=12, opacity=0.8))

    # Update the labels and titles for the X and Y axes
    fig.update_xaxes(title_text=xaxis_title)
    fig.update_yaxes(title_text=yaxis_title)

    # Center the plot title horizontally and set the figure size
    fig.update_layout(
        title_x=0.5,
        width=800,         # Set the width of the figure
        height=600,        # Set the height of the figure
        showlegend=True    # Show the legend
    )

    return fig
