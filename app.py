import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS = {
    "BERT (Encoder-only)": {
        "id": "abd87/bert-roman-urdu-sentiment",
        "info": "bert-base-multilingual-cased | Test Acc 64.4%, Macro-F1 0.632",
    },
    "GPT-2 (Decoder-only)": {
        "id": "abd87/gpt2-roman-urdu-sentiment",
        "info": "gpt2 | Test Acc 62.3%, Macro-F1 0.610",
    },
}
EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}

st.set_page_config(page_title="Roman Urdu Sentiment", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner="Model load ho raha hai...")
def load_model(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl.config.pad_token_id = tok.pad_token_id
    mdl.eval()
    return tok, mdl


def predict(name, text):
    tok, mdl = load_model(MODELS[name]["id"])
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probs = torch.softmax(mdl(**enc).logits, dim=-1)[0].numpy()
    return {mdl.config.id2label[i]: float(p) for i, p in enumerate(probs)}


def show(name, scores):
    label = max(scores, key=scores.get)
    st.subheader(f"{EMOJI.get(label, '')} {label.upper()}")
    for k, v in scores.items():
        st.write(f"{k}: {v:.1%}")
        st.progress(v)


st.title("💬 Roman Urdu Sentiment Analysis")
st.caption("Encoder-only (BERT) vs Decoder-only (GPT-2), both fine-tuned on Roman Urdu")

choice = st.sidebar.radio("Model chuno", list(MODELS.keys()) + ["Compare Both"])
st.sidebar.markdown("---")
for n, m in MODELS.items():
    st.sidebar.caption(f"**{n}**\n\n{m['info']}")

text = st.text_area("Roman Urdu sentence likho:", placeholder="ye drama bohat acha hai")

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Pehle kuch likho.")
    elif choice == "Compare Both":
        results = {}
        cols = st.columns(2)
        for col, name in zip(cols, MODELS):
            with col:
                st.markdown(f"### {name}")
                results[name] = predict(name, text)
                show(name, results[name])
        labels = [max(r, key=r.get) for r in results.values()]
        if labels[0] == labels[1]:
            st.success(f"Dono models agree: {labels[0].upper()}")
        else:
            st.info("Models disagree: " + " vs ".join(
                f"{n.split()[0]} = {l}" for n, l in zip(results, labels)))
    else:
        show(choice, predict(choice, text))

st.markdown("---")
st.caption("Note: Roman Urdu spelling variations aur slang pe model kabhi kabhi galat ho sakta hai.")
st.caption("Note: model Roman Urdu spelling variations aur slang pe kabhi kabhi galat ho sakta hai.")
