import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



MODEL_NAME = "facebook/nllb-200-distilled-600M"


@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    return tokenizer, model


tokenizer, model = load_model()




def translate(text):

    tokenizer.src_lang = "eng_Latn"

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                "fra_Latn"
            ),
            max_new_tokens=200
        )

    translation = tokenizer.batch_decode(
        output,
        skip_special_tokens=True
    )[0]

    return translation




st.set_page_config(
    page_title="English → French Translator",
    page_icon="🌍"
)

st.title("🌍 English → French Translator")

st.write(
    "Translate English text into French using "
    "Meta's NLLB-200 pretrained model."
)

text = st.text_area(
    "English Text",
    placeholder="Enter your English text here...",
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