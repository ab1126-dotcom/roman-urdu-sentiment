import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Roman Urdu Sentiment AI", page_icon="💬", layout="wide")

MODELS = {
    "BERT (Encoder-only)": {
        "id": "abd87/bert-roman-urdu-sentiment",
        "base": "bert-base-multilingual-cased",
        "params": "178M",
        "acc": 64.4,
        "f1": 63.2,
        "desc": "Bidirectional encoder. Pura sentence dono taraf se samajhta hai.",
    },
    "GPT-2 (Decoder-only)": {
        "id": "abd87/gpt2-roman-urdu-sentiment",
        "base": "gpt2",
        "params": "124M",
        "acc": 62.3,
        "f1": 61.0,
        "desc": "Causal decoder. Left-to-right padhta hai, last token se classify karta hai.",
    },
}
COLORS = {"positive": "#22C55E", "neutral": "#F59E0B", "negative": "#EF4444"}
EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😠"}
ORDER = ["positive", "neutral", "negative"]
EXAMPLES = [
    "ye drama bohat acha hai",
    "bilkul bakwas hai, paisa barbaad",
    "theek hai, kuch khaas nahi",
    "mujhe ye phone bohat pasand aaya",
    "service bohat buri thi",
]

st.markdown(
    """
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 1150px;}
.hero {
  background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #06B6D4 100%);
  border-radius: 20px; padding: 32px 36px; margin-bottom: 22px;
  box-shadow: 0 10px 40px rgba(79,70,229,0.35);
}
.hero h1 {color: #fff; margin: 0; font-size: 2.2rem; font-weight: 800;}
.hero p {color: rgba(255,255,255,0.88); margin: 8px 0 0 0; font-size: 1.02rem;}
.badge {
  display: inline-block; background: rgba(255,255,255,0.18); color: #fff;
  padding: 4px 12px; border-radius: 999px; font-size: 0.78rem; margin: 14px 8px 0 0;
}
.card {
  background: #121A33; border: 1px solid #1F2A4D; border-radius: 16px;
  padding: 22px; height: 100%;
}
.card h3 {margin: 0 0 4px 0; color: #E5E7EB; font-size: 1.05rem;}
.sub {color: #94A3B8; font-size: 0.8rem; margin-bottom: 14px;}
.verdict {font-size: 1.8rem; font-weight: 800; margin: 6px 0 2px 0;}
.conf {color: #94A3B8; font-size: 0.85rem; margin-bottom: 16px;}
.row {display: flex; align-items: center; margin: 9px 0; font-size: 0.88rem; color: #CBD5E1;}
.row .lbl {width: 78px;}
.row .track {flex: 1; background: #0B1020; height: 10px; border-radius: 999px; overflow: hidden;}
.row .fill {height: 100%; border-radius: 999px;}
.row .pct {width: 58px; text-align: right; font-variant-numeric: tabular-nums;}
.agree {
  border-radius: 14px; padding: 14px 18px; margin-top: 18px; font-weight: 600;
}
.stButton > button {border-radius: 12px; font-weight: 600;}
.stTextArea textarea {border-radius: 14px; font-size: 1.05rem;}
.small {color: #64748B; font-size: 0.78rem; text-align: center; margin-top: 30px;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Model load ho raha hai... (pehli baar 1-2 minute)")
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


def result_card(name, scores):
    label = max(scores, key=scores.get)
    color = COLORS[label]
    bars = ""
    for k in ORDER:
        v = scores.get(k, 0.0)
        bars += (
            f'<div class="row"><div class="lbl">{EMOJI[k]} {k.capitalize()}</div>'
            f'<div class="track"><div class="fill" style="width:{v*100:.1f}%;background:{COLORS[k]};"></div></div>'
            f'<div class="pct">{v:.1%}</div></div>'
        )
    html = (
        f'<div class="card"><h3>{name}</h3>'
        f'<div class="sub">{MODELS[name]["base"]}</div>'
        f'<div class="verdict" style="color:{color};">{EMOJI[label]} {label.upper()}</div>'
        f'<div class="conf">Confidence: {scores[label]:.1%}</div>{bars}</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def set_text(t):
    st.session_state.user_text = t


if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# ---------- HERO ----------
st.markdown(
    """
<div class="hero">
  <h1>💬 Roman Urdu Sentiment AI</h1>
  <p>Encoder-only (BERT) vs Decoder-only (GPT-2): dono fine-tuned Roman Urdu sentiment ke liye.</p>
  <span class="badge">🤗 Hugging Face</span>
  <span class="badge">3 classes: positive / neutral / negative</span>
  <span class="badge">20k sentences dataset</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    choice = st.radio("Model chuno", list(MODELS.keys()) + ["Compare Both"])
    st.markdown("---")
    if choice != "Compare Both":
        m = MODELS[choice]
        st.markdown(f"**{choice}**")
        st.caption(m["desc"])
        st.caption(f"Base: {m['base']} · Params: {m['params']}")
        st.caption(f"Test Accuracy: {m['acc']}% · Macro-F1: {m['f1']/100:.3f}")
    else:
        st.caption("Dono models ek saath chalenge aur side by side result milega.")

tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📊 Model Performance", "ℹ️ About"])

# ---------- TAB 1 ----------
with tab1:
    st.markdown("##### Roman Urdu sentence likho")
    st.text_area(
        "sentence",
        key="user_text",
        height=110,
        placeholder="ye drama bohat acha hai",
        label_visibility="collapsed",
    )
    st.caption("Examples (click karke try karo):")
    cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        cols[i].button(ex, key=f"ex{i}", on_click=set_text, args=(ex,), use_container_width=True)

    go = st.button("🚀 Analyze", type="primary", use_container_width=True)

    if go:
        text = st.session_state.user_text
        if not text.strip():
            st.warning("Pehle kuch likho.")
        elif choice == "Compare Both":
            with st.spinner("Dono models chal rahe hain..."):
                results = {n: predict(n, text) for n in MODELS}
            c1, c2 = st.columns(2)
            names = list(MODELS.keys())
            with c1:
                result_card(names[0], results[names[0]])
            with c2:
                result_card(names[1], results[names[1]])
            l1 = max(results[names[0]], key=results[names[0]].get)
            l2 = max(results[names[1]], key=results[names[1]].get)
            if l1 == l2:
                st.markdown(
                    f'<div class="agree" style="background:rgba(34,197,94,0.12);'
                    f'border:1px solid #22C55E;color:#86EFAC;">'
                    f'✅ Dono models agree: {l1.upper()}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="agree" style="background:rgba(245,158,11,0.12);'
                    f'border:1px solid #F59E0B;color:#FCD34D;">'
                    f'⚠️ Models disagree: BERT = {l1.upper()} vs GPT-2 = {l2.upper()}</div>',
                    unsafe_allow_html=True,
                )
        else:
            with st.spinner("Predict ho raha hai..."):
                scores = predict(choice, text)
            result_card(choice, scores)

# ---------- TAB 2 ----------
with tab2:
    st.markdown("##### Test set results (1967 sentences)")
    n1, n2 = list(MODELS.keys())
    c1, c2 = st.columns(2)
    with c1:
        st.metric(f"{n1} Accuracy", f"{MODELS[n1]['acc']}%",
                  f"{MODELS[n1]['acc'] - MODELS[n2]['acc']:+.1f} vs GPT-2")
        st.metric(f"{n1} Macro-F1", f"{MODELS[n1]['f1']/100:.3f}")
    with c2:
        st.metric(f"{n2} Accuracy", f"{MODELS[n2]['acc']}%",
                  f"{MODELS[n2]['acc'] - MODELS[n1]['acc']:+.1f} vs BERT")
        st.metric(f"{n2} Macro-F1", f"{MODELS[n2]['f1']/100:.3f}")
    st.markdown("---")
    st.markdown(
        "- **Winner:** BERT, kyunki multilingual pretraining Roman Urdu ke liye zyada helpful hai.\n"
        "- **GPT-2** mostly English pe pretrain hua, phir bhi sirf ~2 points peeche hai.\n"
        "- Dono models neutral class mein behtar aur negative class mein kamzor hain."
    )

# ---------- TAB 3 ----------
with tab3:
    st.markdown("##### Project details")
    st.markdown(
        "- **Task:** Roman Urdu sentiment classification (3 classes)\n"
        "- **Dataset:** community-datasets/roman_urdu (~20k sentences), 80/10/10 split\n"
        "- **Training:** 3 epochs, AdamW, mixed precision, best checkpoint by Macro-F1\n"
        "- **Models:** [BERT](https://huggingface.co/abd87/bert-roman-urdu-sentiment) · "
        "[GPT-2](https://huggingface.co/abd87/gpt2-roman-urdu-sentiment)\n"
        "- **Code:** [GitHub](https://github.com/ab1126-dotcom/roman-urdu-sentiment)"
    )
    st.info(
        "Limitations: Roman Urdu ki standard spelling nahi hoti, sarcasm aur slang pe model "
        "galat ho sakta hai, aur dataset ke labels subjective hain."
    )

st.markdown('<div class="small">Built with Streamlit · Hugging Face Transformers</div>', unsafe_allow_html=True)lang pe kabhi kabhi galat ho sakta hai.")
