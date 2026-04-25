"""
Prompt templates — Stage 5.

Two variants are defined for ablation in Check 5: direct answer and chain-of-thought.
The system prompt establishes the pharmacologist persona and grounding constraint
before any context is injected — this prevents the model from leaning on parametric
memory when retrieved context is present. Templates are strings, not f-strings at
module load time, to avoid accidental early interpolation.
"""

SYSTEM_PROMPT = """You are a computational pharmacologist specialising in voltage-gated ion channels.
Your task is to generate mechanistic hypotheses about drug–channel interactions.

Rules:
- Ground every claim in the retrieved context provided. Do not use knowledge not present in the context.
- Cite the source chunk ID when making a specific claim (e.g. [pubmed_12345678_2]).
- If the context does not contain enough information to answer, say so explicitly.
- Output must be valid JSON matching this exact schema — no extra fields, no prose outside the JSON:

{
  "hypothesis": "<the core mechanistic claim as a single sentence>",
  "supporting_chunk_ids": ["<chunk_id_1>", "<chunk_id_2>"],
  "confidence": "<high|medium|low>",
  "data_gaps": ["<gap 1>", "<gap 2>"],
  "mechanism_type": "<e.g. channel block, gating shift, trafficking — or null>"
}"""

DIRECT_TEMPLATE = """Retrieved context:
{context}

Query: {query}

Generate a hypothesis in JSON format."""

COT_TEMPLATE = """Retrieved context:
{context}

Query: {query}

Think step by step:
1. Identify which retrieved chunks are most relevant to the query.
2. Extract the key mechanistic claims from those chunks.
3. Synthesise a novel hypothesis that connects these claims.
4. Identify any gaps where the context is insufficient.

Then output your final hypothesis in JSON format."""


def build_prompt(
    query: str,
    context_chunks: list[dict],
    variant: str = "cot",
) -> tuple[str, str]:
    """
    Construct (system_prompt, user_prompt) for the given variant.

    Args:
        query: the user query string
        context_chunks: list of chunk dicts with 'text' and 'chunk_id' keys
        variant: "cot" | "direct"

    Returns:
        (system_prompt, user_prompt) tuple ready for LLMClient.complete()
    """
    if variant not in ("cot", "direct"):
        raise ValueError(f"variant must be 'cot' or 'direct', got {variant!r}")

    # Format each chunk as a labelled block so the model can cite by chunk_id.
    # Chunk IDs are included inline so citation tracking in the output parser
    # can match them with a simple substring check.
    chunk_lines = []
    for chunk in context_chunks:
        cid = chunk.get("chunk_id", "unknown")
        source = chunk.get("source", "")
        text = chunk.get("text", "").strip()
        chunk_lines.append(f"[{cid}] (source: {source})\n{text}")
    context_str = "\n\n".join(chunk_lines)

    template = COT_TEMPLATE if variant == "cot" else DIRECT_TEMPLATE
    user_prompt = template.format(context=context_str, query=query)
    return SYSTEM_PROMPT, user_prompt
