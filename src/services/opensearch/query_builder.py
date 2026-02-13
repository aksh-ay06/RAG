import logging 
from typing import Any,Dict,List,Optional

logger = logging.getLogger(__name__)

class PaperQueryBuilder:
    def __init__(
            self,
            query: str = None,
            size: int = 10,
            from_: int = 0,
            fields: Optional[List[str]] = None,
            categories: Optional[List[str]] = None,
            track_total_hits: bool = True,
            latest_papers: bool = False,
    ):
        self.query = query
        self.size = size
        self.from_ = from_
        self.fields = fields or ["title^3","abstract^2","authors^1"]
        self.categories = categories
        self.track_total_hits = track_total_hits
        self.latest_papers = latest_papers

    def build(self) -> Dict[str, Any]:
        query_body = {
            "query": self._build_query(),
            "size": self.size,
            "from": self.from_,
            "track_total_hits": self.track_total_hits,
            "_source": self._build_source_fields(),
            "highlight": self._build_highlight(),
        }

       sort = self._build_sort()
       if sort:
           query_body["sort"] = sort

        return query_body

    def _build_query(self) -> Dict[str, Any]:

        must_clauses = []

        if self.query.strip():
            must_clauses.append(self._build_text_query())

        filter_clauses = {}

        bool_query = {}

        if must_clauses:
            bool_query["must"] = must_clauses
        else:
            bool_query["must"] = [{"match_all": {}}]

        if filter_clauses:
            bool_query["filter"] = filter_clauses

        return {"bool": bool_query}
    
    def _build_text_query(self) -> Dict[str, Any]:

        return {
            "multi_match": {
                "query": self.query,
                "fields": self.fields,
                "type": "best_fields",
                "operator": "or",
                "fuzziness": "AUTO",
                "prefix_length": 2,
            }
        }
    
    def _build_filters(self) -> List[Dict[str, Any]]:
        filters = []

        if self.categories:
            filters.append({"terms": {"categories": self.categories}})

        return filters
    
    def _build_source_fields(self) -> List[str]:
        return [
            "arxiv_id",
            "title",
            "authors",
            "abstract",
            "authors",
            "categories",
            "published_date",
            "pdf_url",
        ]
    
    def _build_highlight(self) -> Dict[str, Any]:

        return {
            "fields": {
                "title"; {
                    "fragment_size": 0,
                    "number_of_fragments": 0,
                },
                "abstract": {
                    "fragment_size": 150,
                    "number_of_fragments": 3,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                },
                "authors": {
                    "fragment_size": 0,
                    "number_of_fragments": 0,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                },
            },
            "require_field_match": False,
        }

    def _build_sort(self) -> Optional[List[Dict[str, Any]]]:
        if self.latest_papers:
            return [{"published_date": {"order": "desc"}},"_score"]
        
        if self.query.strip():
            return None
        
        return [{"published_date": {"order": "desc"}},"_score"]
    
    def build_search_query(
            query: str,
            size: int = 10,
            from_: int = 0,
            categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        builder = PaperQueryBuilder(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
        )
        return builder.build()