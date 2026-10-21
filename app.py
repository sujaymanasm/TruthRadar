"""
TruthRadar - AI-Powered Fake News Detection & Explainability System
-------------------------------------------------------------------
Interactive Streamlit Web Application combining multi-signal NLP models:
- TF-IDF + Logistic Regression
- DistilBERT Transformer
- Sentence Transformer (Headline-Body Consistency)
- VADER Sentiment Polarity
- spaCy Named Entity Recognition
- LIME Explainable AI Visualizations
- Psycholinguistics & Stylometric Profiling (Sensationalism, Readability, TTR)
- Domain Reputation & Satire Intelligence
- Interactive Token-Level Forensic Highlighting
- Dynamic 1-Click PDF Forensic Audit Certificate Export
- Counterfactual Rephrasing Sandbox
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

import config
from src.utils import validate_text, truncate_text, format_percentage
from src.preprocessor import clean_text, extract_headline_and_body
from src.tfidf_classifier import get_tfidf_classifier
from src.bert_classifier import get_bert_classifier
from src.sentiment_analyzer import analyze_sentiment
from src.headline_checker import check_headline_body_consistency
from src.ner_analyzer import extract_entities
from src.explainability import explain_prediction, create_explanation_chart
from src.fake_score import calculate_fake_score
from src.scraper import scrape_article_from_url
from src.stylometrics import analyze_stylometrics
from src.domain_trust import evaluate_domain_trust
from src.visual_highlighter import generate_highlighted_html
from src.pdf_generator import generate_forensic_pdf

# =====================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="TruthRadar - AI Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, polished UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .verdict-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0 0.5rem 0;
        text-align: center;
        color: white;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%);
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.3);
    }
    .verdict-real {
        background: linear-gradient(135deg, #16A34A 0%, #166534 100%);
        box-shadow: 0 4px 15px rgba(22, 163, 74, 0.3);
    }
    .verdict-satire {
        background: linear-gradient(135deg, #D97706 0%, #B45309 100%);
        box-shadow: 0 4px 15px rgba(217, 119, 6, 0.3);
    }
    .verdict-title {
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .verdict-sub {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.95;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-person { background-color: #DBEAFE; color: #1E40AF; }
    .badge-org { background-color: #FEE2E2; color: #991B1B; }
    .badge-gpe { background-color: #D1FAE5; color: #065F46; }
    .badge-date { background-color: #FEF3C7; color: #92400E; }
    .badge-money { background-color: #EDE9FE; color: #5B21B6; }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.9rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #78350F;
        margin-top: 1.5rem;
    }
    .intel-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# CACHED MODEL LOADERS
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_all_models():
    """Caches and initializes all machine learning & NLP backends."""
    tfidf = get_tfidf_classifier()
    bert = get_bert_classifier()
    return tfidf, bert


tfidf_clf, bert_clf = load_all_models()


# =====================================================================
# SAMPLE NEWS EXAMPLES FOR QUICK TESTING
# =====================================================================
SAMPLES = {
    "Select an example...": {"headline": "", "text": ""},
    "Example 1: Solar Cell Breakthrough (REAL)": {
        "headline": "Scientists Discover High-Efficiency Solar Cell Using Perovskite Crystals",
        "text": "Researchers at the National Renewable Energy Laboratory have developed a new multi-junction solar cell utilizing perovskite crystal layers that achieves an unprecedented 32.5 percent energy conversion efficiency in laboratory trials. The breakthrough, published in the journal Nature Materials, demonstrates remarkable stability under continuous sunlight exposure over 1,500 hours. Lead author Dr. Elena Martinez emphasized that scalable manufacturing techniques could lower commercial solar panel costs significantly within the next three years. The research team partnered with industrial manufacturers to initiate pilot production testing."
    },
    "Example 2: Secret Alien Base (FAKE)": {
        "headline": "SHOCKING: Secret Government Alien Base Exposed Inside Mount Everest Summit",
        "text": "Whistleblower documents leaked by anonymous insiders reveal that international military elites have operated a secret underground alien technology research station deep beneath Mount Everest for over four decades. According to shocking reports, world leaders regularly visit the subterranean facility to receive instructions from extraterrestrial beings controlling global financial markets and weather patterns. Mainstream media networks are under strict gag orders to suppress this alarming truth from reaching the public."
    },
    "Example 3: Central Bank Rate Cut (REAL)": {
        "headline": "Central Bank Announces Quarter-Point Interest Rate Cut Amid Cooling Inflation",
        "text": "The Federal Reserve announced a 25 basis point reduction in the benchmark federal funds rate following a two-day policy meeting in Washington. Federal Reserve Chairman Jerome Powell noted during the press conference that consumer price inflation has steadily moderated toward the 2 percent annual target, while the labor market remains resilient. Major stock indexes responded with modest gains following the announcement. Economic analysts expect gradual monetary easing across the upcoming fiscal quarters if inflation metrics remain subdued."
    },
    "Example 4: Lemon Cancer Miracle (FAKE)": {
        "headline": "MIRACLE CURE: Drinking Boiled Lemon Water with Baking Soda Completely Cures Cancer in 48 Hours",
        "text": "A suppressed medical breakthrough proves that boiling organic lemons with pure baking soda creates an alkaline remedy that kills 100 percent of cancer cells in just two days. Big Pharma executives and corrupted oncologists have conspired to hide this simple kitchen recipe to protect multi-billion dollar chemotherapy profits. Thousands of patients have cured terminal stage-4 cancers overnight using this miracle protocol before hospitals deleted the clinical trial evidence."
    }
}


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("<div style='font-size: 3rem; margin-bottom: 0.2rem;'>🛡️</div>", unsafe_allow_html=True)
    st.markdown("## **TruthRadar**")
    st.markdown("**AI-Powered Fake News Detection & Explainability System**")
    st.markdown("---")

    st.markdown("### ⚙️ **Model Pipeline Status**")
    st.write(f"• **TF-IDF + Logistic Reg:** {'🟢 Loaded' if tfidf_clf.is_loaded else '🟡 Needs Training'}")
    st.write(f"• **DistilBERT (PyTorch):** {'🟢 Loaded' if bert_clf.is_loaded else '🟡 Standby / Trainable'}")
    st.write("• **Semantic Similarity:** 🟢 Sentence-MiniLM")
    st.write("• **Sentiment Engine:** 🟢 NLTK VADER")
    st.write("• **Named Entity Engine:** 🟢 spaCy")
    st.write("• **Explainable AI:** 🟢 LIME Engine")
    st.write("• **Stylometric Profiling:** 🟢 Active")
    st.write("• **Domain Intelligence:** 🟢 Active")

    if not tfidf_clf.is_loaded:
        st.warning("⚠️ Baseline model not found. Run `python training/train_tfidf.py` to build the model.")

    st.markdown("---")
    st.markdown("### 🧪 **Quick Demo Samples**")
    selected_sample_key = st.selectbox(
        "Load sample article for instant testing:",
        list(SAMPLES.keys())
    )

    st.markdown("---")
    st.markdown("### 📖 **Academic Attribution**")
    st.caption(
        "TruthRadar is a research-grade NLP system designed for academic evaluation under the supervision of **Prof. Kuntal Dey**."
    )


# =====================================================================
# HEADER SECTION
# =====================================================================
st.markdown("<div class='main-header'>🛡️ TruthRadar</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>AI-Powered Fake News Detection & Forensic Explainability System — Multi-Signal NLP Architecture</div>",
    unsafe_allow_html=True
)


# =====================================================================
# INPUT SECTION (TABS)
# =====================================================================
tab_paste, tab_headline, tab_url = st.tabs([
    "📄 Tab 1: Paste Article",
    "📰 Tab 2: Headline + Article",
    "🌐 Tab 3: Live Web URL & Domain Intel"
])

input_headline = ""
input_article = ""
input_source_meta = {}
has_explicit_headline = False

# Auto-fill if user selected a sidebar sample
preset_headline = SAMPLES[selected_sample_key]["headline"]
preset_text = SAMPLES[selected_sample_key]["text"]

with tab_paste:
    st.markdown("##### Paste full news article text below:")
    paste_text = st.text_area(
        "Article Content",
        value=preset_text if selected_sample_key != "Select an example..." else "",
        height=220,
        placeholder="Paste complete news article body here (minimum 40 characters)...",
        key="paste_area"
    )
    if paste_text:
        h_cand, b_cand = extract_headline_and_body(paste_text)
        if h_cand and h_cand != b_cand:
            input_headline = h_cand
            input_article = b_cand
            has_explicit_headline = True
        else:
            input_headline = ""
            input_article = paste_text
            has_explicit_headline = False

with tab_headline:
    st.markdown("##### Enter headline and article body separately:")
    h_input = st.text_input(
        "News Headline / Title",
        value=preset_headline if selected_sample_key != "Select an example..." else "",
        placeholder="e.g. Scientists Discover High-Efficiency Solar Cell...",
        key="headline_input"
    )
    b_input = st.text_area(
        "Article Body",
        value=preset_text if selected_sample_key != "Select an example..." else "",
        height=180,
        placeholder="Paste corresponding article body text here...",
        key="body_input"
    )
    if b_input:
        input_headline = h_input.strip()
        input_article = b_input.strip()
        has_explicit_headline = bool(input_headline)

with tab_url:
    st.markdown("##### Extract and analyze article from a live web link:")
    url_input = st.text_input(
        "News Article URL",
        placeholder="e.g. https://www.reuters.com/... or https://www.theonion.com/...",
        key="url_input"
    )
    
    # Live Domain Reputation Check
    if url_input.strip():
        domain_intel = evaluate_domain_trust(url_input)
        if domain_intel["is_satire"]:
            st.warning(f"🎭 **SATIRE ALERT:** {domain_intel['explanation']}")
        elif "High Trust" in domain_intel["trust_tier"]:
            st.success(f"🏛️ **VERIFIED PUBLISHER:** {domain_intel['explanation']}")
        elif "High Risk" in domain_intel["trust_tier"]:
            st.error(f"🚨 **DOCUMENTED MISINFORMATION DOMAIN:** {domain_intel['explanation']}")
        else:
            st.info(f"🌐 **DOMAIN STATUS:** {domain_intel['explanation']}")

    fetch_btn = st.button("📥 Fetch & Extract Article", key="fetch_btn")
    
    if "scraped_data" not in st.session_state:
        st.session_state.scraped_data = None

    if fetch_btn and url_input:
        with st.spinner("Fetching and extracting article content from URL..."):
            scrape_res = scrape_article_from_url(url_input)
            st.session_state.scraped_data = scrape_res

    if st.session_state.scraped_data:
        s_data = st.session_state.scraped_data
        if s_data["success"]:
            st.success(f"✅ Successfully extracted article from **{s_data['domain']}**")
            st.text_input("Extracted Title", value=s_data["title"], key="scraped_title_display")
            st.text_area("Extracted Article Content", value=s_data["text"][:1000] + "...", height=120, disabled=True)
            input_headline = s_data["title"]
            input_article = s_data["text"]
            has_explicit_headline = bool(input_headline)
            input_source_meta = {
                "Author": s_data.get("author") or "Not specified",
                "Published Date": s_data.get("publish_date") or "Not specified",
                "Domain": s_data.get("domain") or "Unknown",
                "URL": s_data.get("url")
            }
        else:
            st.error(f"❌ Extraction notice: {s_data['error']}")


# =====================================================================
# ANALYSIS TRIGGER & PROCESSING
# =====================================================================
st.markdown("<br>", unsafe_allow_html=True)
col_btn, col_info = st.columns([1, 3])
with col_btn:
    analyze_clicked = st.button("🚀 ANALYZE NEWS", type="primary", use_container_width=True)

if analyze_clicked:
    full_text_for_eval = (input_headline + " " + input_article).strip() if input_headline else input_article.strip()
    is_valid, error_msg = validate_text(full_text_for_eval)

    if not is_valid:
        st.error(f"⚠️ {error_msg}")
    else:
        with st.spinner("🔍 TruthRadar is running multi-signal NLP analysis..."):
            # A. TF-IDF Model Inference
            tfidf_pred = tfidf_clf.predict(full_text_for_eval)

            # B. BERT Transformer Inference
            bert_pred = bert_clf.predict(full_text_for_eval)

            # C. Headline-Body Semantic Consistency
            headline_eval = None
            if has_explicit_headline and input_headline and input_article:
                headline_eval = check_headline_body_consistency(input_headline, input_article)

            # D. Sentiment Analysis
            sentiment_eval = analyze_sentiment(full_text_for_eval)

            # E. Named Entity Recognition
            ner_eval = extract_entities(full_text_for_eval)

            # F. Stylometric & Psycholinguistic Profiling
            stylometrics_eval = analyze_stylometrics(full_text_for_eval)

            # G. Domain Trust Assessment
            domain_eval = None
            if input_source_meta.get("URL"):
                domain_eval = evaluate_domain_trust(input_source_meta["URL"])
            elif url_input.strip():
                domain_eval = evaluate_domain_trust(url_input)

            # H. Multi-Signal Calibrated Ensemble Scoring
            try:
                fake_score_res = calculate_fake_score(
                    tfidf_result=tfidf_pred,
                    bert_result=bert_pred,
                    headline_result=headline_eval,
                    sentiment_result=sentiment_eval,
                    has_headline=has_explicit_headline,
                    domain_result=domain_eval,
                    stylometrics_result=stylometrics_eval
                )
            except TypeError:
                fake_score_res = calculate_fake_score(
                    tfidf_result=tfidf_pred,
                    bert_result=bert_pred,
                    headline_result=headline_eval,
                    sentiment_result=sentiment_eval,
                    has_headline=has_explicit_headline
                )

            # I. LIME Explainability
            lime_eval = explain_prediction(full_text_for_eval)

            # J. Generate Instant Forensic Audit PDF
            calc_conf = fake_score_res.get("confidence_score", max(fake_score_res["fake_probability"], fake_score_res["real_probability"]))
            pdf_bytes = generate_forensic_pdf(
                article_text=full_text_for_eval,
                verdict=fake_score_res["prediction"],
                fake_prob=fake_score_res["fake_percentage"] / 100.0,
                confidence=calc_conf,
                sentiment_data=sentiment_eval,
                entities_data=ner_eval.get("entities_by_type", {}),
                lime_weights=lime_eval.get("raw_explanation", []),
                stylometrics=stylometrics_eval,
                headline_consistency=headline_eval,
                domain_info=domain_eval
            )

        # =============================================================
        # RESULTS DISPLAY
        # =============================================================
        st.markdown("---")
        st.markdown("### 📊 **TruthRadar Authenticity Report**")

        # 1. Primary Verdict Card
        if fake_score_res["prediction"] == "SATIRE / PARODY":
            card_class = "verdict-satire"
            badge_icon = "🎭"
        elif fake_score_res["prediction"] == "FAKE NEWS":
            card_class = "verdict-fake"
            badge_icon = "🚨"
        else:
            card_class = "verdict-real"
            badge_icon = "✅"

        st.markdown(f"""
        <div class='verdict-card {card_class}'>
            <div class='verdict-title'>{badge_icon} PREDICTION: {fake_score_res['prediction']}</div>
            <div class='verdict-sub'>
                Fake Probability: <b>{fake_score_res['fake_percentage']}%</b> | 
                Real Probability: <b>{fake_score_res['real_percentage']}%</b> | 
                Confidence: <b>{fake_score_res['confidence']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. One-Click Instant PDF Export Button
        download_filename = f"TruthRadar_Forensic_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button(
            label="📥 Download Official Forensic Audit Certificate (PDF)",
            data=pdf_bytes,
            file_name=download_filename,
            mime="application/pdf",
            use_container_width=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Key Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Fake Probability", f"{fake_score_res['fake_percentage']}%")
        with m2:
            st.metric("Real Probability", f"{fake_score_res['real_percentage']}%")
        with m3:
            st.metric("Confidence Level", fake_score_res["confidence"])
        with m4:
            st.metric("Overall Sentiment", sentiment_eval["label"])

        # 4. Model Breakdown & Weights
        st.markdown("#### 🔬 **Model Breakdown & Component Signals**")
        b1, b2, b3 = st.columns(3)

        with b1:
            st.markdown("**TF-IDF + ML Classifier**")
            if tfidf_pred.get("is_available"):
                t_prob = tfidf_pred["fake_probability"] * 100
                st.progress(tfidf_pred["fake_probability"], text=f"Fake Risk: {t_prob:.1f}%")
                st.caption(f"Prediction: **{tfidf_pred['prediction']}**")
            else:
                st.info("TF-IDF model not yet trained.")

        with b2:
            st.markdown("**DistilBERT Transformer**")
            if bert_pred.get("is_available"):
                b_prob = bert_pred["fake_probability"] * 100
                st.progress(bert_pred["fake_probability"], text=f"Fake Risk: {b_prob:.1f}%")
                st.caption(f"Device: {bert_pred.get('device', 'cpu').upper()}")
            else:
                st.info("BERT model: Default standby (un-fine-tuned).")

        with b3:
            st.markdown("**Headline-Body Consistency**")
            if headline_eval and headline_eval.get("is_available"):
                h_sim = headline_eval["similarity_percentage"]
                st.progress(headline_eval["similarity_score"], text=f"Alignment: {h_sim:.1f}%")
                st.caption(f"Status: **{headline_eval['interpretation']}**")
            else:
                st.caption("No separate headline provided for similarity check.")

        # 5. Explainable AI: Interactive LIME Chart + Inline Forensic Highlighter
        st.markdown("#### 🔍 **Explainable AI & Forensic Reader**")
        tab_lime_chart, tab_text_highlighter = st.tabs([
            "📊 Feature Importance Chart (LIME)",
            "🔍 Inline Forensic Reader (Token Highlights)"
        ])

        with tab_lime_chart:
            col_lime_chart, col_lime_table = st.columns([3, 2])
            with col_lime_chart:
                lime_fig = create_explanation_chart(lime_eval)
                if lime_fig:
                    st.plotly_chart(lime_fig, use_container_width=True)
                else:
                    st.info("Train the TF-IDF model to generate interactive LIME word charts.")

            with col_lime_table:
                st.markdown("**Top Influential Terms:**")
                if lime_eval.get("explanation_list"):
                    df_exp = pd.DataFrame(lime_eval["explanation_list"])
                    df_exp = df_exp[["word", "weight", "supports"]].rename(
                        columns={"word": "Word / Token", "weight": "Influence Weight", "supports": "Supports"}
                    )
                    st.dataframe(df_exp, use_container_width=True, height=290)
                else:
                    st.caption("No feature weights available.")

        with tab_text_highlighter:
            st.markdown("**Token-Level Forensic Attribution in Context:**")
            st.caption("The article text is rendered below with words dynamically highlighted: red indicates deceitful/sensationalist cues, while green indicates credibility anchors.")
            highlighted_card = generate_highlighted_html(full_text_for_eval, lime_eval.get("raw_explanation", []))
            st.markdown(highlighted_card, unsafe_allow_html=True)

        # 6. Psycholinguistic & Stylometrics Dashboard
        st.markdown("#### 🧠 **Psycholinguistic & Stylometric Diagnostics**")
        sty1, sty2, sty3, sty4 = st.columns(4)
        with sty1:
            sens_val = stylometrics_eval["sensationalism_score"]
            st.metric("Sensationalism Index", f"{sens_val}/100", delta="- High Arousal" if sens_val > 40 else "+ Measured", delta_color="inverse")
            st.caption(f"Exclamations: {stylometrics_eval['exclamation_count']} | Shouting (Caps): {stylometrics_eval['caps_ratio']*100:.1f}%")
        with sty2:
            st.metric("Flesch Reading Ease", f"{stylometrics_eval['flesch_reading_ease']}/100")
            st.caption(stylometrics_eval["readability_label"])
        with sty3:
            st.metric("Reading Grade Level", f"Grade {stylometrics_eval['flesch_kincaid_grade']}")
            st.caption(f"Avg words/sentence: {stylometrics_eval['avg_sentence_length']}")
        with sty4:
            st.metric("Lexical Diversity (TTR)", f"{stylometrics_eval['type_token_ratio']}")
            st.caption("Vocabulary Richness (Unique/Total Words)")

        if stylometrics_eval["intensifiers_found"]:
            st.markdown(f"**Flagged Emotional Intensifiers:** " + " ".join([f"<span class='badge' style='background:#FEE2E2; color:#991B1B;'>{w}</span>" for w in stylometrics_eval["intensifiers_found"]]), unsafe_allow_html=True)

        # 7. Sentiment & Named Entities Columns
        col_sent, col_ner = st.columns(2)

        with col_sent:
            st.markdown("#### 🎭 **Sentiment & Emotional Tone**")
            st.write(f"**Tone Assessment:** {sentiment_eval['tone_description']}")
            sp1, sp2, sp3 = st.columns(3)
            sp1.metric("Positive", f"{sentiment_eval['positive'] * 100:.1f}%")
            sp2.metric("Neutral", f"{sentiment_eval['neutral'] * 100:.1f}%")
            sp3.metric("Negative", f"{sentiment_eval['negative'] * 100:.1f}%")
            st.write(f"Compound Polarity Score: `{sentiment_eval['compound']}` (Range: -1.0 to +1.0)")

        with col_ner:
            st.markdown("#### 🏷️ **Named Entities Identified**")
            if ner_eval.get("entities_by_type"):
                st.write(f"**Engine:** {ner_eval.get('engine')}")
                import html
                for label, items in ner_eval["entities_by_type"].items():
                    badge_class = "badge-person" if label in ["PERSON", "PERSON / ORG"] else "badge-org" if label == "ORG" else "badge-gpe" if label in ["GPE", "LOC"] else "badge-date" if label == "DATE" else "badge-money"
                    badges = []
                    for it in items[:6]:
                        safe_it = html.escape(str(it)).replace("$", "&#36;")
                        badges.append(f"<span class='badge {badge_class}'>{safe_it}</span>")
                    items_str = " ".join(badges)
                    st.markdown(f"**{label}:** {items_str}", unsafe_allow_html=True)
            else:
                st.info("No named entities detected in the text.")

        # 8. Counterfactual What-If Sandbox (Interactive Academic Feature)
        with st.expander("🔬 **Counterfactual Sandbox: Test Rephrasing & Sensitivity**", expanded=False):
            st.caption("Experiment with removing sensational terms, modifying headlines, or calming emotional intensifiers to observe real-time causal shifts in prediction:")
            sandbox_text = st.text_area("Edit text for live sensitivity re-test:", value=full_text_for_eval, height=120, key="sandbox_area")
            if st.button("⚡ Test Counterfactual Shift", key="sandbox_btn"):
                quick_pred = tfidf_clf.predict(sandbox_text)
                quick_sty = analyze_stylometrics(sandbox_text)
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("New Fake Probability", f"{quick_pred.get('fake_probability', 0)*100:.1f}%")
                sc2.metric("New Prediction", quick_pred.get("prediction", "N/A"))
                sc3.metric("New Sensationalism Score", f"{quick_sty['sensationalism_score']}/100")

        # 9. Fact-Checking Disclaimer
        st.markdown("""
        <div class='disclaimer-box'>
            <b>🛡️ Academic & Fact-Checking Disclaimer:</b><br>
            TruthRadar is an AI-based fake-news detection and linguistic analysis tool.
            Its predictions are probabilistic model estimates and must not be treated as definitive proof that an article is factually true or false.
            AI models evaluate statistical and stylistic patterns rather than live real-world ground truth.
            Always verify critical news items with authoritative, reputable primary sources.
        </div>
        """, unsafe_allow_html=True)
