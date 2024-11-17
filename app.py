import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer, BertForSequenceClassification
import torch

st.title('News Summary Creater')
st.write('This is news summarization app. Enter the news text in the below box.')

user = st.text_area("Enter the news text to summarize:")

cat_tokenizer = BertTokenizer.from_pretrained("./categorization_model")
cat_model = BertForSequenceClassification.from_pretrained("./categorization_model")
sum_tokenizer = T5Tokenizer.from_pretrained("./summary_model")
sum_model = T5ForConditionalGeneration.from_pretrained("./summary_model")

if st.button("Summarize"):
    inputs = sum_tokenizer.encode("summarize: " + user, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = sum_model.generate(inputs, max_length=256, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    inputs = cat_tokenizer(user, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    cat_model.eval()
    with torch.no_grad():
        outputs = cat_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
    label_map = {0: "business", 1: "entertainment", 2: "politics", 3: "sport", 4: "tech"}
    predicted_class = label_map[predictions.item()]
    st.write("Class:" + " " + predicted_class)
    st.write("Summary:")
    st.write(summary)    