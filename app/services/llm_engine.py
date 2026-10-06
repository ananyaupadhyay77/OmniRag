from typing import List, Dict, Any, Tuple
from app.schemas.query import Citation
from app.schemas.document import SourceType


class LLMEngine:
    """Constructs grounded context prompts and builds citation objects."""

    @staticmethod
    def build_prompt_and_citations(
        query: str, 
        reranked_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[Citation]]:
        context_blocks = []
        citations = []

        for idx, chunk in enumerate(reranked_chunks):
            source_num = idx + 1
            text = chunk.get("text", "")
            title = chunk.get("title", "Unknown Source")
            source_type = chunk.get("source_type", SourceType.MARKDOWN)
            path = chunk.get("source_url_or_path", "N/A")
            parent_id = chunk.get("parent_id", "")

            # Build readable context block for LLM
            context_blocks.append(f"[Source {source_num}: {title}]\n{text}")

            # Build Citation object for UI/API
            citations.append(
                Citation(
                    source_title=title,
                    source_type=source_type,
                    source_path=path,
                    excerpt=text[:150] + "...",
                    parent_id=parent_id
                )
            )

        context_str = "\n\n---\n\n".join(context_blocks)

        system_prompt = f"""You are OmniRAG, a precise AI technical assistant. 
Answer the user's question relying strictly on the background context below. 
If the context does not contain the answer, state "I cannot find this in the uploaded documents."

BACKGROUND CONTEXT:
{context_str}

USER QUESTION:
{query}
"""
        return system_prompt, citations