import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


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
            num_beams=4
        )

    translation = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return translation


st.set_page_config(
    page_title="English → French Translator",
    page_icon="🇫🇷"
)

st.title("🇬🇧 → 🇫🇷 English to French Translator")

st.write(
    "Translate English text into French using a pretrained Transformer model."
)


text = st.text_area(
    "Enter English text",
    placeholder="Type your English text here...",
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
