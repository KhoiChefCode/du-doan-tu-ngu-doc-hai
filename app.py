import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

st.set_page_config(page_title="Phân loại bình luận độc hại", layout="centered")
st.title("🔍 Phân loại bình luận độc hại (Đa ngôn ngữ)")

with st.spinner("🚀 Đang tải mô hình..."):
    # Tải tokenizer và model
    model_name = "textdetox/xlmr-base-toxicity-classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # 💡 Đảm bảo mô hình chạy trên CPU
    model.to(torch.device("cpu"))

    # Tạo pipeline thủ công
    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False, device=-1)

# Nhập văn bản
text = st.text_area("✍️ Nhập bình luận:", height=150)

if st.button("🧠 Dự đoán"):
    if not text.strip():
        st.warning("⚠️ Vui lòng nhập nội dung trước khi dự đoán.")
    else:
        try:
            result = classifier(text)[0]
            label = result["label"]
            score = result["score"]

            if label.lower().startswith("toxic"):
                st.error(f"🚫 Bình luận độc hại ({score:.2%} tin cậy)")
            else:
                st.success(f"✅ Bình luận không độc hại ({score:.2%} tin cậy)")
        except Exception as e:
            st.exception(f"❌ Lỗi khi xử lý: {e}")
