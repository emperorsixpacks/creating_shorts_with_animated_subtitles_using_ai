import requests
from huggingface_hub import InferenceClient

def get_entities(text):
    """
    Retrieves named entities from the given text using a pre-trained RoBERTa model for English named entity recognition.

    Parameters:
        text (str): The input text from which named entities will be extracted.

    Returns:
        List[str]: A list of named entities extracted from the input text. Each entity is a string without leading or trailing whitespace.
    """
    ner_model = InferenceClient(token="<your huggingface token>")
    result = ner_model.token_classification(
        text=text, model="jean-baptiste/roberta-large-ner-english"
    )
    return [i["word"].strip() for i in result]

def wiki_search(query: str):
    """
    Searches for a given query on the Wikipedia API and returns a list of page keys.

    Parameters:
        query (str): The search query to be used for the Wikipedia API.

    Returns:
        list: A list of page keys extracted from the response of the Wikipedia API.
    """

    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "titles": query,
    }

    # Send the API request
    response = requests.get(
        f"https://en.wikipedia.org/w/rest.php/v1/search/page?q={query}&limit=4", params=params, timeout=60
    )

    if response.status_code == 200:
        # Parse the JSON data
        data = response.json()

        # Extract the page content from the response
        pages = data["pages"]
        return [page["key"].lower() for page in pages]

    else:
        print("Failed to retrieve page content.")


print(wiki_search("Lightning McQueen"))

