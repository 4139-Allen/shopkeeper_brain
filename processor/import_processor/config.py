"""
导入流程配置管理模块
集中管理所有配置项，支持环境变量
"""
from __future__ import annotations
import os
from dotenv import load_dotenv
from dataclasses import field
from typing import Optional
from dataclasses import dataclass


load_dotenv()

"""
@dataclass:是 Python 3.7+ 引入的装饰器，用于自动生成类的样板代码。
它会自动为类生成以下方法：
__init__() - 构造函数
__repr__() - 字符串表示
__eq__() - 相等比较
__hash__() - 哈希（可选）
"""

@dataclass
class ImportConfig:
    """导入流程模块 所需配置"""

    # 1.文档处理配置
    max_content_length: int = 1000      #切片最大长度
    img_content_length: int = 200       #图片上下文最大长度
    min_merge_content_length: int = 500 #合并短内容的最小长度
    overlap_sentences: int = 1          #句子级切分时的重叠句数
    item_name_chunk_k: int = 3          #商品名识别时使用的切片数量
    item_name_chunk_size: int = 2500    #商品名识别时使用的切片内容长度

    """
    使用 field(default_factory=lambda: ...)，因为：
    ✅ 支持运行时环境变量变化 ✅ 避免模块加载时序问题 ✅ 保持代码风格一致 ✅ 更符合配置管理的最佳实践
    如果直接赋值，可能会导致环境变量修改后不生效的 bug！
    """
    # 2.系统支持的图片扩展名
    image_extensions: set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # 3.LLM配置
    qwen_base_url: str =field(
        default_factory=lambda :os.getenv("QWEN_BASE_URL", "")
    )
    qwen_api_key: str = field(
        default_factory=lambda :os.getenv("QWEN_API_KEY", "")
    )
    vl_model: str = field(
        default_factory=lambda :os.getenv("VL_MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )
    # ==================== 4.Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )

    # ==================== 5.MinIO 配置 ====================
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    minio_secure: bool = False

    # ==================== 6.向量配置 ====================
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )
    embedding_batch_size: int = 8

    # ==================== 速率限制 ====================
    requests_per_minute: int = 10  # 图片总结 API 速率限制

    # 创建实例对象
    @classmethod
    def from_env(cls) -> ImportConfig:
        """从环境变量加载配置"""
        return cls()

    # http://192.168.6.150:9000/
    def get_minio_base_url(self):
        base_protocol = "https://" if self.minio_secure else "http://"
        return base_protocol + f"{self.minio_endpoint}"

# ==================== 全局单例 ====================
# 下划线前缀：Python 约定表示"私有"或"内部使用"
# 暗示不应该直接从模块外部访问，应该通过 get_config() 函数获取
# 注意线程安全：多线程环境需要加锁
_config: Optional[ImportConfig] = None

#_lock = threading.Lock()

# 单例获取函数
def get_config() -> ImportConfig | None:
    """获取配置单例"""
    global _config
    #with _lock:  # 加锁保护
    if _config is None:
        _config = ImportConfig.from_env()
    return _config



