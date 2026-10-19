"""Keyphrase and concept extraction using KeyBERT, spaCy NER, and noun-chunk filtering.

Strictly deterministic NLP — NO LLM call.
Reuses the sentence-transformers embedding model from segmentation.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from lecturenotes.concepts.models import KeyConcept, SectionConcepts
from lecturenotes.config import load_config
from lecturenotes.segmentation.models import TopicSection
from lecturenotes.segmentation.texttiling import get_embedding_model

logger = logging.getLogger(__name__)

_spacy_nlp: Any = None
_keybert_model: Any = None

# Filter out low-value generic terms / pronouns / determiners
_STOP_CHUNKS = {
    "this course",
    "today's lecture",
    "next lecture",
    "the goal",
    "we",
    "it",
    "they",
    "example",
    "course",
    "lecture",
    "summary",
    "page",
    "what",
    "which",
    "this",
    "that",
    "these",
    "those",
}


def get_spacy_model(model_name: str | None = None) -> Any:
    """Load or retrieve the cached spaCy pipeline."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy

        if not model_name:
            model_name = load_config().spacy.model
        logger.info("Loading spaCy model: %s", model_name)
        try:
            _spacy_nlp = spacy.load(model_name)
        except OSError:
            logger.info("spaCy model '%s' not found locally. Auto-downloading...", model_name)
            from spacy.cli import download
            download(model_name)
            _spacy_nlp = spacy.load(model_name)
    return _spacy_nlp


def get_keybert_model(embedding_model_name: str | None = None) -> Any:
    """Initialize or retrieve KeyBERT reusing the sentence-transformers model."""
    global _keybert_model
    if _keybert_model is None:
        from keybert import KeyBERT  # type: ignore[import-untyped]

        emb_model = get_embedding_model(embedding_model_name)
        logger.info("Initializing KeyBERT with embedding model")
        _keybert_model = KeyBERT(model=emb_model)
    return _keybert_model


def _clean_candidate(term: str) -> str:
    """Strip punctuation and leading/trailing articles."""
    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", term).strip()
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def extract_spacy_candidates(text: str) -> tuple[set[str], dict[str, str]]:
    """Extract candidate technical terms using spaCy NER and noun chunks.

    Returns:
        tuple of (candidates set, mapping of candidate -> category).
    """
    nlp = get_spacy_model()
    doc = nlp(text)

    candidates: set[str] = set()
    category_map: dict[str, str] = {}

    # Extract named entities (filter out numerical/date entities)
    excluded_ent_types = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
    for ent in doc.ents:
        if ent.label_ not in excluded_ent_types:
            clean = _clean_candidate(ent.text)
            if len(clean) >= 3 and clean.lower() not in _STOP_CHUNKS:
                candidates.add(clean)
                category_map[clean.lower()] = "named_entity"

    # Extract noun chunks
    for chunk in doc.noun_chunks:
        clean = _clean_candidate(chunk.text)
        lower = clean.lower()
        if (
            len(clean) >= 3
            and len(clean) <= 50
            and lower not in _STOP_CHUNKS
            and not lower.startswith("page ")
            and not lower.startswith("figure ")
        ):
            # Keep if contains meaningful characters
            if any(c.isalpha() for c in clean):
                candidates.add(clean)
                if lower not in category_map:
                    category_map[lower] = "noun_chunk"

    return candidates, category_map


def extract_concepts_for_section(
    section: TopicSection,
    top_n: int | None = None,
    diversity: float | None = None,
    model_name: str | None = None,
) -> list[KeyConcept]:
    """Extract and rank key concepts for a single topic section."""
    cfg = load_config()
    top_n = top_n if top_n is not None else cfg.keybert.top_n
    diversity = diversity if diversity is not None else cfg.keybert.diversity

    text = section.text.strip()
    if not text:
        return []

    # 1. SpaCy NER & noun-chunk candidate extraction
    spacy_candidates, category_map = extract_spacy_candidates(text)

    # 2. KeyBERT ranking reusing the sentence-transformers model
    emb_model = get_embedding_model(model_name)
    raw_keywords: list[tuple[str, float]] = []

    if spacy_candidates:
        from keybert._mmr import mmr  # type: ignore[import-untyped]

        cand_list = sorted(list(spacy_candidates))
        try:
            doc_emb = emb_model.encode([text], normalize_embeddings=True)
            cand_embs = emb_model.encode(cand_list, normalize_embeddings=True)
            raw_keywords = mmr(
                doc_emb,
                cand_embs,
                cand_list,
                top_n=min(top_n, len(cand_list)),
                diversity=diversity,
            )
        except Exception as e:
            logger.warning("Direct MMR candidate scoring failed (%s); falling back to KeyBERT ngrams", e)
            raw_keywords = []

    if not raw_keywords:
        # Fallback to standard KeyBERT n-gram extraction
        kw_model = get_keybert_model(model_name)
        raw_keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n,
            use_mmr=True,
            diversity=diversity,
        )

    # 3. Build KeyConcept models with source sentence attribution
    concepts: list[KeyConcept] = []
    seen_terms: set[str] = set()

    for term, score in raw_keywords:
        clean_term = _clean_candidate(term)
        lower_term = clean_term.lower()

        if lower_term in seen_terms or lower_term in _STOP_CHUNKS:
            continue
        seen_terms.add(lower_term)

        category = category_map.get(lower_term, "keyphrase")

        # Find matching source sentence
        matching_sentence: str | None = None
        for s in section.sentences:
            if re.search(r"\b" + re.escape(clean_term) + r"\b", s.text, re.IGNORECASE):
                matching_sentence = s.text
                break
        if not matching_sentence and section.sentences:
            matching_sentence = section.sentences[0].text

        clamped_score = max(0.0, min(1.0, round(float(score), 4)))
        concepts.append(
            KeyConcept(
                term=clean_term,
                score=clamped_score,
                section_id=section.section_id,
                category=category,  # type: ignore[arg-type]
                source_sentence=matching_sentence,
            )
        )

    # Sort descending by score
    concepts.sort(key=lambda c: c.score, reverse=True)
    return concepts


def extract_lecture_concepts(
    sections: list[TopicSection],
    top_n_per_section: int | None = None,
    diversity: float | None = None,
) -> list[SectionConcepts]:
    """Extract and rank key concepts across all topic sections in a lecture."""
    results: list[SectionConcepts] = []
    for sec in sections:
        sec_concepts = extract_concepts_for_section(
            sec,
            top_n=top_n_per_section,
            diversity=diversity,
        )
        results.append(
            SectionConcepts(
                section_id=sec.section_id,
                section_title=sec.title,
                concepts=sec_concepts,
            )
        )
    return results


def get_all_unique_concepts(
    section_concepts_list: list[SectionConcepts],
) -> list[KeyConcept]:
    """Aggregate unique key concepts across all sections, keeping the highest score."""
    best_by_term: dict[str, KeyConcept] = {}
    for sc in section_concepts_list:
        for c in sc.concepts:
            key = c.term.lower()
            if key not in best_by_term or c.score > best_by_term[key].score:
                best_by_term[key] = c

    sorted_concepts = sorted(best_by_term.values(), key=lambda c: c.score, reverse=True)
    return sorted_concepts
