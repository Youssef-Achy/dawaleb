import pandas as pd
import streamlit as st

from assistant import DrugAssistant, detect_language

st.set_page_config(page_title="DawaLeb", page_icon="💊", layout="centered")

st.markdown(
    """
<style>
.rtl { direction: rtl; text-align: right; }
.disclaimer { font-size: 0.85em; color: #8a6d3b; background: #fcf8e3;
              border: 1px solid #faebcc; border-radius: 6px; padding: 8px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assistant() -> DrugAssistant:
    return DrugAssistant()


assistant = load_assistant()

st.title("💊 DawaLeb · دوا لبنان")
st.caption(
    "Find officially registered alternatives, prices and info for medications "
    "in Lebanon — data from the Ministry of Public Health. · "
    "ابحث عن البدائل المسجّلة رسمياً وأسعار الأدوية في لبنان"
)
st.markdown(
    '<div class="disclaimer">This is an information tool only. not medical advice. '
    "Always confirm with your pharmacist or doctor. · "
    "أداة معلومات فقط وليست نصيحة طبية . استشر الصيدلي أو الطبيب دائماً.</div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        css = "rtl" if msg.get("lang") == "ar" else ""
        st.markdown(
            f'<div class="{css}">{msg["content"]}</div>', unsafe_allow_html=True
        )
        if msg.get("table") is not None:
            st.dataframe(msg["table"], use_container_width=True, hide_index=True)

prompt = st.chat_input("Type a medication name… · اكتب اسم الدواء…")

if prompt:
    lang = detect_language(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "lang": lang})
    with st.chat_message("user"):
        css = "rtl" if lang == "ar" else ""
        st.markdown(f'<div class="{css}">{prompt}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("Searching the MoPH database…"):
            res = assistant.answer(prompt)

        css = "rtl" if res["lang"] == "ar" else ""
        st.markdown(f'<div class="{css}">{res["text"]}</div>', unsafe_allow_html=True)

        table = None
        if res["found"] and res["result"]["alternatives"]:
            rows = []
            qp = res["result"]["query_product"]
            if qp:
                rows.append(qp)
            rows.extend(res["result"]["alternatives"])
            table = pd.DataFrame(
                [
                    {
                        "Name": r["brand_name"],
                        "Type": r["b_g"],
                        "Ingredients": r["ingredients_raw"],
                        "Form": r["form"],
                        "Price (LBP)": f"{r['price_lbp']:,}" if r["price_lbp"] else "—",
                        "MoPH page": r["detail_url"],
                    }
                    for r in rows
                ]
            )
            st.dataframe(table, use_container_width=True, hide_index=True)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": res["text"],
                "lang": res["lang"],
                "table": table,
            }
        )

with st.sidebar:
    st.header("About · حول")
    st.write(
        "DawaLeb helps people in Lebanon find registered generic "
        "alternatives when a medication is unavailable or unaffordable. "
        "Data: Lebanon National Drugs Database (MoPH)."
    )
    st.write("Try · جرّب:")
    st.code("abilify\nاسبرين\naripiprazole\nabiprx 10  (typo demo)")
    n = len(assistant.products)
    st.metric("Products indexed", f"{n:,}")
    st.caption(
        "Built for the LebNet Tech Fellows program. "
        "Source data © Lebanese Ministry of Public Health."
    )
