```python
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ==============================
# Configuration
# ==============================

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"


# ==============================
# Load Model
# ==============================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# ==============================
# Translation Function
# ==============================

def translate(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=4,
            early_stopping=True
        )

    translation = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return translation


# ==============================
# Streamlit Interface
# ==============================

st.set_page_config(
    page_title="English → French Translator",
    page_icon="🇫🇷",
    layout="centered"
)

st.title("🇬🇧 → 🇫🇷 English to French Translator")

st.write(
    "Translate English text into French using "
    "a pretrained Transformer model."
)


text = st.text_area(
    "Enter English text",
    placeholder="Type your English sentence here...",
    height=150
)


if st.button("Translate", type="primary"):

    if text.strip():

        with st.spinner("Translating..."):

            result = translate(text)

        st.subheader("French Translation")

        st.success(result)

    else:

        st.warning("Please enter some English text.")
```
