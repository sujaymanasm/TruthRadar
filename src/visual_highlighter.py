"""
src/visual_highlighter.py - Token-Level Visual Explainability & Forensic Reader
-------------------------------------------------------------------------------
Renders the full article text with inline token highlights indicating
linguistic attribution towards Fake (Red) vs Real (Green) predictions.
"""

import html
import re
from typing import List, Tuple, Dict

def generate_highlighted_html(text: str, lime_weights: List[Tuple[str, float]], max_words: int = 250) -> str:
    """
    Wraps words in the text with colored inline spans based on attribution weights.
    
    Args:
        text: Original article text.
        lime_weights: List of (word, weight) tuples.
        max_words: Limit words to prevent overly huge DOM rendering.
        
    Returns:
        HTML formatted string with CSS tooltips and color-coded spans.
    """
    if not text or not text.strip():
        return "<p style='color: #6B7280; font-style: italic;'>No text to highlight.</p>"

    # Build lookup dictionary for lowercase words
    weight_map: Dict[str, float] = {}
    for word, weight in lime_weights:
        clean_w = str(word).lower().strip()
        weight_map[clean_w] = float(weight)

    # Tokenize preserving spaces and punctuation
    tokens = re.findall(r'\b[A-Za-z0-9\'-]+\b|[^\w\s]|\s+', text)
    
    output_parts = []
    word_count = 0

    for token in tokens:
        # Check if token is a word
        if re.match(r'^[A-Za-z0-9\'-]+$', token):
            word_count += 1
            if word_count > max_words:
                output_parts.append(f"<span style='color: #9CA3AF; font-style: italic;'>... [truncated for display, {len(text.split()) - max_words} more words]</span>")
                break
                
            lower_token = token.lower()
            escaped_token = html.escape(token)

            if lower_token in weight_map:
                w = weight_map[lower_token]
                # Fake-leaning (positive weight): Red
                if w > 0:
                    intensity = min(1.0, abs(w) * 3) # scale for visual visibility
                    bg_color = f"rgba(239, 68, 68, {0.2 + 0.45 * intensity})"
                    border_color = "#DC2626"
                    text_color = "#7F1D1D"
                    tooltip = f"Flagged Term | Fake Contribution: +{w:.3f}"
                # Real-leaning (negative weight): Green
                else:
                    intensity = min(1.0, abs(w) * 3)
                    bg_color = f"rgba(34, 197, 94, {0.2 + 0.45 * intensity})"
                    border_color = "#16A34A"
                    text_color = "#14532D"
                    tooltip = f"Credible Anchor | Real Contribution: {w:.3f}"
                
                span = (
                    f'<span title="{tooltip}" style="'
                    f'background-color: {bg_color}; '
                    f'border-bottom: 2px solid {border_color}; '
                    f'color: {text_color}; '
                    f'padding: 1px 4px; '
                    f'margin: 0 1px; '
                    f'border-radius: 4px; '
                    f'font-weight: 600; '
                    f'cursor: help;">'
                    f'{escaped_token}'
                    f'</span>'
                )
                output_parts.append(span)
            else:
                output_parts.append(escaped_token)
        else:
            # Punctuation or whitespace
            output_parts.append(html.escape(token))

    body_html = "".join(output_parts)
    
    container_html = f"""
    <div style="
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        line-height: 1.8;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 0.95rem;
        color: #1E293B;
        max-height: 380px;
        overflow-y: auto;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    ">
        {body_html}
    </div>
    <div style="margin-top: 8px; font-size: 0.8rem; color: #64748B; display: flex; gap: 16px; align-items: center;">
        <span><span style="background-color: rgba(239, 68, 68, 0.4); border-bottom: 2px solid #DC2626; color: #7F1D1D; padding: 1px 6px; border-radius: 4px; font-weight: 600;">Red Highlight</span> = Deceptive / Sensationalist Attribution</span>
        <span><span style="background-color: rgba(34, 197, 94, 0.4); border-bottom: 2px solid #16A34A; color: #14532D; padding: 1px 6px; border-radius: 4px; font-weight: 600;">Green Highlight</span> = Factual / Credibility Attribution</span>
        <span style="font-style: italic;">(Hover over highlighted words to see influence scores)</span>
    </div>
    """
    return container_html
