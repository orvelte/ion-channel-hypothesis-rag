KNOWN LIMITATION: mean cosine = 0.987 (embedding collapse due to BERT anisotropy).
Absolute cosine scores are not reliable confidence signals.
Retrieval relies on relative ranking only.
If score-based filtering is needed in future, requires contrastive fine-tuning
(e.g. SimCSE) or a natively contrastive model (e.g. text-embedding-3-small).
