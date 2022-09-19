from transformers import AutoTokenizer, AutoConfig, AutoModel

MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def initialise_model():
    model = AutoModel.from_pretrained(MODEL)
    return model


def initialise_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    return tokenizer
