from . import (
    dedupe_edges,
    dedupe_nodes,
    extract_edge_dates,
    extract_edges,
    extract_nodes,
    extract_attributes,
    invalidate_edges,
    summarize_nodes,
    summarize_episode,
)
from .models import Message, PromptFunction, PromptVersion


class PromptLibrary:
    def __init__(self):
        self.dedupe_edges: dedupe_edges.Prompt = dedupe_edges.versions  # type: ignore
        self.dedupe_nodes: dedupe_nodes.Prompt = dedupe_nodes.versions  # type: ignore
        self.extract_edge_dates: extract_edge_dates.Prompt = (
            extract_edge_dates.versions  # type: ignore
        )
        self.extract_edges: extract_edges.Prompt = extract_edges.versions  # type: ignore
        self.extract_nodes: extract_nodes.Prompt = extract_nodes.versions  # type: ignore
        self.extract_attributes: extract_attributes.Prompt = extract_attributes.versions
        self.invalidate_edges: invalidate_edges.Prompt = invalidate_edges.versions  # type: ignore
        self.summarize_nodes: summarize_nodes.Prompt = summarize_nodes.versions  # type: ignore
        self.summarize_episode: summarize_episode.PromptFunction = (
            summarize_episode.create_summary
        )  # ADDED THIS LINE (assuming create_summary is the function we want to expose directly)


prompt_library = PromptLibrary()

__all__ = ['prompt_library', 'Message', 'PromptFunction', 'PromptVersion']
