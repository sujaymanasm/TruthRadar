"""
src/pdf_generator.py - On-Demand Forensic Audit Report PDF Generator
-------------------------------------------------------------------
Compiles an interactive evaluation session into a downloadable,
publication-quality forensic verification certificate PDF.
"""

import io
from datetime import datetime
from typing import Dict, Any, List, Tuple

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

def generate_forensic_pdf(
    article_text: str,
    verdict: str,
    fake_prob: float,
    confidence: float,
    sentiment_data: Dict[str, Any],
    entities_data: Dict[str, List[str]],
    lime_weights: List[Tuple[str, float]],
    stylometrics: Dict[str, Any] = None,
    headline_consistency: Dict[str, Any] = None,
    domain_info: Dict[str, Any] = None
) -> bytes:
    """
    Compiles an instant 1-page Forensic Verification Audit PDF and returns raw bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Styles
    header_title = ParagraphStyle(
        'HeaderTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=19,
        textColor=colors.HexColor("#0D3B66")
    )
    
    header_sub = ParagraphStyle(
        'HeaderSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569")
    )
    
    sec_title = ParagraphStyle(
        'SecTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=5,
        spaceAfter=3
    )
    
    body_txt = ParagraphStyle(
        'BodyTxt',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#334155")
    )

    story = []

    # 1. Top Header Banner
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    audit_id = f"TR-{abs(hash(article_text[:60])) % 10000000:07d}"
    
    top_table = Table([
        [
            Paragraph("<b>TruthRadar Forensic Audit Certificate</b><br/><font size=7 color='#64748B'>Automated Multi-Signal Natural Language Processing & Explainability Engine</font>", header_title),
            Paragraph(f"<b>Audit Case ID:</b> {audit_id}<br/><b>Generated:</b> {now_str}<br/><b>Evaluator:</b> Academic Evaluation", header_sub)
        ]
    ], colWidths=[360, 180])
    top_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(top_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0D3B66"), spaceBefore=2, spaceAfter=6))

    # 2. Executive Verdict Card
    is_fake = verdict.upper() == "FAKE"
    verdict_bg = colors.HexColor("#FEE2E2") if is_fake else colors.HexColor("#DCFCE7")
    verdict_text_color = "#991B1B" if is_fake else "#166534"
    verdict_label = "HIGH RISK: LIKELY DECEPTIVE / FABRICATED" if is_fake else "VERIFIED: LIKELY FACTUAL / AUTHENTIC"
    
    verdict_html = f"""
    <div style="text-align: center;">
        <font size=14 color="{verdict_text_color}"><b>{verdict_label}</b></font><br/>
        <font size=8.5 color="#334155">Calculated Deception Probability: <b>{fake_prob:.1%}</b> | Calibration Confidence: <b>{confidence:.1%}</b></font>
    </div>
    """
    
    v_table = Table([[Paragraph(verdict_html, body_txt)]], colWidths=[540])
    v_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), verdict_bg),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#EF4444" if is_fake else "#22C55E")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (0,0), (-1,-1), 'CENTER')
    ]))
    story.append(v_table)
    story.append(Spacer(1, 6))

    # 3. Multi-Signal Diagnostic Metrics
    story.append(Paragraph("1. Multi-Signal Diagnostic Assessment", sec_title))
    
    # Stylometrics extract
    fre = stylometrics.get("flesch_reading_ease", "N/A") if stylometrics else "N/A"
    fk_grade = stylometrics.get("flesch_kincaid_grade", "N/A") if stylometrics else "N/A"
    sens_score = f"{stylometrics.get('sensationalism_score', 'N/A')}/100" if stylometrics else "N/A"
    ttr = f"{stylometrics.get('type_token_ratio', 'N/A')}" if stylometrics else "N/A"
    
    sent_label = sentiment_data.get("sentiment_label", "Neutral")
    sent_compound = sentiment_data.get("compound", 0.0)
    
    c_score = f"{headline_consistency.get('similarity_score', 0):.2f}" if headline_consistency else "N/A"
    c_label = headline_consistency.get("verdict", "N/A") if headline_consistency else "Direct Article Mode"
    
    dom_label = domain_info.get("trust_tier", "N/A") if domain_info else "Pasted Text Analysis"

    metrics_matrix = [
        ["Analytical Signal", "Observed Value", "Forensic Interpretation"],
        ["Lexical Deception Probability (TF-IDF)", f"{fake_prob:.2%}", "Statistical n-gram distribution matches fake training patterns" if fake_prob > 0.5 else "Term frequency matches standard journalistic reporting"],
        ["Sentiment Polarity & Affect (VADER)", f"{sent_label} ({sent_compound:+.2f})", "Extreme affective valence often correlates with sensational manipulation"],
        ["Headline-Body Semantic Coherence", f"{c_score} ({c_label})", "Detects clickbait disparity between sensational title and text"],
        ["Psycholinguistic Sensationalism Index", f"{sens_score}", "Syntactic density of intensifiers, exclamations, and shouting"],
        ["Readability & Lexical Diversity", f"Grade {fk_grade} | TTR: {ttr}", f"Reading Ease: {fre}/100 (Lower grade levels promote viral spread)"],
        ["Source / Domain Reputation", f"{dom_label}", "Evaluates publisher domain accountability & satire status"]
    ]
    
    m_table = Table(metrics_matrix, colWidths=[175, 125, 240])
    m_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0D3B66")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(m_table)
    story.append(Spacer(1, 6))

    # 4. Token-Level Attribution (LIME) & Named Entities
    story.append(Paragraph("2. Explainable AI (LIME) Feature Attributions & Named Entities", sec_title))
    
    fake_triggers = [f"<b>{w}</b> (+{sc:.3f})" for w, sc in lime_weights if sc > 0][:5]
    real_anchors = [f"<b>{w}</b> ({sc:.3f})" for w, sc in lime_weights if sc <= 0][:5]
    
    fake_triggers_str = ", ".join(fake_triggers) if fake_triggers else "No significant positive triggers found"
    real_anchors_str = ", ".join(real_anchors) if real_anchors else "No significant negative anchors found"

    persons = ", ".join(entities_data.get("PERSON", [])[:4]) or "None detected"
    orgs = ", ".join(entities_data.get("ORG", [])[:4]) or "None detected"
    places = ", ".join(entities_data.get("GPE", [])[:4]) or "None detected"

    exp_table_data = [
        [Paragraph("<b>Deceptive / Sensationalist Attribution Terms (LIME):</b><br/>" + fake_triggers_str, body_txt)],
        [Paragraph("<b>Credibility / Journalistic Anchoring Terms (LIME):</b><br/>" + real_anchors_str, body_txt)],
        [Paragraph(f"<b>Extracted Named Entities:</b> [Persons: {persons}] | [Organizations: {orgs}] | [Geopolitical Locations: {places}]", body_txt)]
    ]
    
    exp_table = Table(exp_table_data, colWidths=[540])
    exp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#FEF2F2")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#F0FDF4")),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(exp_table)
    story.append(Spacer(1, 6))

    # 5. Article Excerpt Audited
    story.append(Paragraph("3. Audited Content Excerpt", sec_title))
    excerpt = article_text.strip()[:350].replace('\n', ' ') + ("..." if len(article_text) > 350 else "")
    excerpt_p = Paragraph(f"<i>\"{excerpt}\"</i>", body_txt)
    exc_table = Table([[excerpt_p]], colWidths=[540])
    exc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(exc_table)
    story.append(Spacer(1, 8))

    # 6. Certification Footer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceBefore=2, spaceAfter=4))
    footer_text = (
        "<b>TruthRadar System Architecture:</b> Trained on IEEE WELFake Corpus (72,134 Articles, 95.73% Accuracy) | "
        "Ensemble: TF-IDF + DistilBERT + MiniLM + VADER + spaCy + LIME | Live: https://truthradar-w9h24zcwzxu7hdo6fx9ckq.streamlit.app/<br/>"
        "<i>Academic Disclaimer: This forensic certificate is automatically generated for NLP research and evaluation purposes under the supervision of Prof. Kuntal Dey.</i>"
    )
    story.append(Paragraph(footer_text, header_sub))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
