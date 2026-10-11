from functools import cache, lru_cache

from knowledge.services.import_file_service import ImportFileService
from knowledge.services.query_service import QueryService


@cache
def get_import_file_service() -> ImportFileService:
    """工厂函数  单例模式"""
    return ImportFileService()


@lru_cache
def get_query_service() -> QueryService:
    return QueryService()
