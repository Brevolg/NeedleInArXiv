"""Search core for the semantic-search project."""

__all__ = ["SearchEngine", "SearchMode", "SearchResult"]


def __getattr__(name: str):
    if name in __all__:
        from .engine import SearchEngine, SearchMode, SearchResult

        return {
            "SearchEngine": SearchEngine,
            "SearchMode": SearchMode,
            "SearchResult": SearchResult,
        }[name]
    raise AttributeError(name)
