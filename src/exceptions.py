class RepositoryException(Exception):
    """Base exception for repository-related errors."""


class PaperNotFound(RepositoryException):
    """Exception raised when paper data is not found."""


class PaperNotSaved(RepositoryException):
    """Exception raised when paper data is not saved."""


class ParsingException(Exception):
    """Base exception for parsing-related errors."""


class OpenSearchException(Exception):
    """Base exception for OpenSearch-related errors."""


class LLMException(Exception):
    """Base exception for LLM-related errors."""


class ConfigurationError(Exception):
    """Exception raised when configuration is invalid."""