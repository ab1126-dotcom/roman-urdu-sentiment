import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "abd87/bert-roman-urdu-sentiment"

st.set_page_config(page_title="Roman Urdu Sentiment", page_icon="💬")

@st.cache_resource(show_spinner="Model load ho raha hai...")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    mdl.eval()
    return tok, mdl

tok, mdl = load_model()

def predict(text):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probs = torch.softmax(mdl(**enc).logits, dim=-1)[0].numpy()
    return {mdl.config.id2label[i]: float(p) for i, p in enumerate(probs)}

st.title("💬 Roman Urdu Sentiment Analysis")
st.caption("Fine-tuned BERT (bert-base-multilingual-cased) | Accuracy 64.4%, Macro-F1 0.632")

text = st.text_area("Roman Urdu sentence likho:", placeholder="ye drama bohat acha hai")

if st.button("Predict", type="primary"):
    if text.strip():
        scores = predict(text)
        label = max(scores, key=scores.get)
        emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}.get(label, "")
        st.subheader(f"{emoji} {label.upper()}")
        for k, v in scores.items():
            st.write(f"{k}: {v:.1%}")
            st.progress(v)
    else:
        st.warning("Pehle kuch likho.")

st.markdown("---")
st.caption("Note: model Roman Urdu spelling variations aur slang pe kabhi kabhi galat ho sakta hai.")
