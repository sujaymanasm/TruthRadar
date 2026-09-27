# TruthRadar — System Limitations & Ethical Considerations

## 1. Technological Limitations

1. **Linguistic Stylometry vs. Fact Verification**:
   - TruthRadar analyzes writing style, sensationalist terminology, grammatical patterns, and semantic alignment.
   - It does not query real-time external databases or truth registries (such as Wikipedia, Reuters Wire, or Politifact APIs). An article written in immaculate journalistic prose that contains a fabricated lie may not be detected purely through stylistic analysis.

2. **Domain Adaptation & Temporal Drift**:
   - Language evolves, and political/cultural discourse shifts over time. Models trained on past datasets may exhibit reduced accuracy on newly emerging crises, scientific jargon, or evolving slang without continuous fine-tuning.

3. **Satire & Sarcasm Ambiguity**:
   - Satirical publications (e.g., *The Onion*, *The Babylon Bee*) intentionally mimic legitimate journalistic structure for humor. Automated models may flag satire as fake news due to factual absurdity despite benign intent.

4. **Web Scraping Restrictions**:
   - News outlets with paywalls, bot protections (Cloudflare, CAPTCHAs), or Single Page Applications (React/Vue) that render text purely via JavaScript cannot always be scraped cleanly via HTTP GET requests. In such cases, manual text pasting is required.

---

## 2. Ethical Considerations & Responsible AI

- **No Defamation / Automated Censorship**:
  TruthRadar is engineered as an analytical assistant for researchers, journalists, and students. Its output must never be used for autonomous content takedowns or censorship without human editorial oversight.
- **Fairness & Neutrality**:
  Lexicon and statistical weights should remain politically and ideologically neutral, focusing on evidential and linguistic indicators rather than specific viewpoints.
