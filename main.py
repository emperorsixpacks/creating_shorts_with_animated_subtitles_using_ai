
from huggingface_hub import InferenceClient

def get_entities(text):
    ner_model = InferenceClient(token="<your huggingface token>")
    result = ner_model.token_classification(
        text=text, model="jean-baptiste/roberta-large-ner-english"
    )
    return [i["word"].strip() for i in result]
