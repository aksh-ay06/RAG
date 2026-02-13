import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError, OpenSearchException
from src.config import Settings, get_settings

from .index_config import ARXIV_PAPERS_INDEX, ARXIV_PAPERS_MAPPING
from .query_builder import PaperQueryBuilder

logger = logging.getLogger(__name__)


# Type definitions for better type safety
class IndexStats(TypedDict):
    index_name: str
    document_count: int
    size_in_bytes: int
    health: str


class SearchResult(TypedDict):
    total: int
    hits: List[Dict[str, Any]]
    error: Optional[str]


class BulkIndexResult(TypedDict):
    success: int
    failed: int


class OpenSearchClient:
    """
    Client for OpenSearch operations including index management and search.

    This client provides methods for creating indices, indexing papers,
    searching with BM25 scoring, and managing OpenSearch cluster operations.
    """

    # Default fields for searching
    DEFAULT_SEARCH_FIELDS = ["title", "abstract", "authors"]

    def __init__(
        self,
        host: str = "http://localhost:9200",
        settings: Optional[Settings] = None,
    ):
        """Initialize OpenSearch client.

        Args:
            host: OpenSearch cluster endpoint URL
            settings: Application settings instance (uses default if None)
        """
        self.host = host
        self.settings = settings or get_settings()
        self.index_name = self.settings.opensearch.index_name or ARXIV_PAPERS_INDEX

        self.client = self._create_client(host)
        logger.info("OpenSearch client initialized with host: %s, index: %s", host, self.index_name)

    def _create_client(self, host: str) -> OpenSearch:
        """Create OpenSearch client with proper configuration."""
        return OpenSearch(
            hosts=[host],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )

    def create_index(self, force: bool = False) -> bool:
        """Create the arxiv-papers index with proper mappings.

        Args:
            force: If True, delete existing index before creating

        Returns:
            True if index was created, False if it already exists

        Raises:
            OpenSearchException: If index creation fails
        """
        try:
            if self._index_exists():
                if force:
                    self._delete_index()
                else:
                    logger.info("Index %s already exists", self.index_name)
                    return False

            return self._create_index()

        except RequestError as e:
            logger.error("Error creating index: %s", e)
            raise
        except OpenSearchException as e:
            logger.error("Unexpected error creating index: %s", e)
            raise

    def _index_exists(self) -> bool:
        """Check if index exists."""
        return self.client.indices.exists(index=self.index_name)

    def _delete_index(self) -> None:
        """Delete existing index."""
        logger.info("Deleting existing index: %s", self.index_name)
        self.client.indices.delete(index=self.index_name)

    def _create_index(self) -> bool:
        """Create index with mappings."""
        response = self.client.indices.create(
            index=self.index_name,
            body=ARXIV_PAPERS_MAPPING
        )

        if response.get("acknowledged"):
            logger.info("Successfully created index: %s", self.index_name)
            return True

        logger.error("Failed to create index: %s", response)
        return False

    def index_paper(self, paper_data: Dict[str, Any]) -> bool:
        """Index a single paper document.

        Args:
            paper_data: Paper data to index

        Returns:
            True if successful, False otherwise
        """
        try:
            validated_data = self._prepare_paper_data(paper_data)
            if not validated_data:
                return False

            response = self.client.index(
                index=self.index_name,
                id=validated_data["arxiv_id"],
                body=validated_data,
                refresh=True,
            )

            success = response.get("result") in ["created", "updated"]
            if success:
                logger.debug("Indexed paper: %s", validated_data["arxiv_id"])
            else:
                logger.error("Failed to index paper: %s", response)

            return success

        except OpenSearchException as e:
            logger.error("Error indexing paper %s: %s", paper_data.get("arxiv_id", "unknown"), e)
            return False

    def _prepare_paper_data(self, paper_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare and validate paper data for indexing."""
        if "arxiv_id" not in paper_data:
            logger.error("Missing arxiv_id in paper data")
            return None

        # Create a copy to avoid modifying original
        prepared_data = paper_data.copy()
        now = datetime.now(timezone.utc).isoformat()

        prepared_data.setdefault("created_at", now)
        prepared_data.setdefault("updated_at", now)

        # Normalize authors field
        if isinstance(prepared_data.get("authors"), list):
            prepared_data["authors"] = ", ".join(prepared_data["authors"])

        return prepared_data

    def bulk_index_papers(self, papers: List[Dict[str, Any]]) -> BulkIndexResult:
        """Bulk index multiple papers.

        Args:
            papers: List of paper data to index

        Returns:
            Dictionary with counts of successful and failed indexing
        """
        results: BulkIndexResult = {"success": 0, "failed": 0}

        for paper in papers:
            if self.index_paper(paper):
                results["success"] += 1
            else:
                results["failed"] += 1

        logger.info(
            "Bulk indexing complete: %d successful, %d failed",
            results["success"],
            results["failed"]
        )
        return results

    def search_papers(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        fields: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        track_total_hits: bool = True,
        latest_papers: bool = False,
    ) -> SearchResult:
        """Search papers using BM25 scoring with query builder.

        Args:
            query: Search query text
            size: Number of results to return
            from_: Offset for pagination
            fields: List of fields to search in (default: title, abstract, authors)
            categories: Filter by categories
            track_total_hits: Whether to track total hits accurately
            latest_papers: Sort by publication date instead of relevance

        Returns:
            Search results with hits and metadata
        """
        try:
            query_builder = PaperQueryBuilder(
                query=query,
                size=size,
                from_=from_,
                fields=fields,
                categories=categories,
                track_total_hits=track_total_hits,
                latest_papers=latest_papers,
            )

            search_body = query_builder.build()
            response = self.client.search(index=self.index_name, body=search_body)

            results = self._format_search_results(response, query)
            return results

        except NotFoundError:
            logger.error("Index %s not found", self.index_name)
            return {"total": 0, "hits": [], "error": "Index not found"}
        except OpenSearchException as e:
            logger.error("Search error: %s", e)
            return {"total": 0, "hits": [], "error": str(e)}

    def _format_search_results(self, response: Dict[str, Any], query: str) -> SearchResult:
        """Format raw search response into SearchResult."""
        total = response["hits"]["total"]["value"]
        hits = []

        for hit in response["hits"]["hits"]:
            paper = hit["_source"]
            paper["score"] = hit["_score"]

            if "highlight" in hit:
                paper["highlights"] = hit["highlight"]

            hits.append(paper)

        logger.info("Search for '%s' returned %d results", query, total)
        return {"total": total, "hits": hits, "error": None}

    def get_index_stats(self) -> IndexStats:
        """Get statistics about the index.

        Returns:
            Dictionary with index statistics
        """
        try:
            stats = self.client.indices.stats(index=self.index_name)
            count = self.client.count(index=self.index_name)
            health = self.client.cluster.health(index=self.index_name)

            return {
                "index_name": self.index_name,
                "document_count": count["count"],
                "size_in_bytes": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"],
                "health": health["status"],
            }
        except OpenSearchException as e:
            logger.error("Error getting index stats: %s", e)
            raise

    def health_check(self) -> bool:
        """Check if OpenSearch is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except OpenSearchException as e:
            logger.error("Health check failed: %s", e)
            return False

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get OpenSearch cluster information.

        Returns:
            Dictionary with cluster info

        Raises:
            OpenSearchException: If unable to retrieve cluster info
        """
        try:
            return self.client.info()
        except OpenSearchException as e:
            logger.error("Error getting cluster info: %s", e)
            raise

    def get_cluster_health(self) -> Dict[str, Any]:
        """Get detailed cluster health information.

        Returns:
            Dictionary with cluster health details

        Raises:
            OpenSearchException: If unable to retrieve cluster health
        """
        try:
            return self.client.cluster.health()
        except OpenSearchException as e:
            logger.error("Error getting cluster health: %s", e)
            raise

    def get_index_mapping(self) -> Dict[str, Any]:
        """Get index mapping.

        Returns:
            Dictionary with index mapping

        Raises:
            OpenSearchException: If unable to retrieve mapping
        """
        try:
            mappings = self.client.indices.get_mapping(index=self.index_name)
            return mappings.get(self.index_name, {}).get("mappings", {})
        except OpenSearchException as e:
            logger.error("Error getting index mapping: %s", e)
            raise

    def get_index_settings(self) -> Dict[str, Any]:
        """Get index settings.

        Returns:
            Dictionary with index settings

        Raises:
            OpenSearchException: If unable to retrieve settings
        """
        try:
            settings = self.client.indices.get_settings(index=self.index_name)
            return settings.get(self.index_name, {}).get("settings", {})
        except OpenSearchException as e:
            logger.error("Error getting index settings: %s", e)
            raise