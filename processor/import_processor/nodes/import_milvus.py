"""
向量数据入库节点
采用门面 + 建造者 设计模式
- 门面角色   ImportMilvusNode节点的process()
- 建造者  _MilvusSchemaBuilder   _MilvusIndexBuilder  _MilvusInserter
"""
from dataclasses import dataclass
from pymilvus import DataType, MilvusClient, CollectionSchema
import logging

from pymilvus.milvus_client import IndexParams

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import ValidationError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)

# 写一个存数据的类
# 装饰器作用：未来直接实例化，不用写__init__  __repr__方法
# 构造完就"冻住",任何修改属性/删除属性的操作都会抛异常（防止全局常量被意外篡改,污染所有后续调用）
@dataclass(frozen=True)
class ScalarFieldSpec:
    """标量字段"""
    field_name: str
    datatype: DataType
    max_length: int | None = None

# 预定义的标量字段
_SCALAR_FIELDS: set[ScalarFieldSpec] = {
    ScalarFieldSpec(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
}


# 1.建造者：Schema构建
class _MilvusSchemaBuilder:
    """职责：专门负责构建约束"""

    @staticmethod
    def build(client: MilvusClient, dim: int) -> CollectionSchema:
        logger.info("开始构建约束(schema)...")
        # 1.构建约束对象(动态映射)
        schema = client.create_schema(enable_dynamic_field=True)
        # 2.构建约束字段
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )
        # 3.构建标量字段约束
        for scalar_field in _SCALAR_FIELDS:
            kwargs: dict = {
                "field_name": scalar_field.field_name,
                "datatype": scalar_field.datatype
            }
            if scalar_field.max_length is not None:
                kwargs['max_length'] = scalar_field.max_length
            schema.add_field(**kwargs)
        # 4.构建向量字段约束
        # ① 稠密向量
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim
        )
        # ② 稀疏向量
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        logger.info(f"构建约束(schema)完成")
        return schema


# 2.建造者：索引构建
class _MilvusIndexBuilder:
    """
    职责：负责处理milvus的索引
    为什么只构建向量字段索引：
        向量字段没有索引就"不能搜",标量字段没有索引只是"过滤慢一点"。
    """

    @staticmethod
    def build(client: MilvusClient, collection_name: str) -> IndexParams:
        logger.info(f"开始构建集合({collection_name})索引")
        index = client.prepare_index_params(collection_name=collection_name)
        # ①稠密向量索引
        index.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX", #Milvus 自动选 ANN 算法，在高维空间建"邻居图",搜索时沿图跳跃逼近最近点,跳过绝大多数无关向量
            metric_type="COSINE"    #余弦相似度:比较两个向量的方向夹角,cos = A·B / (|A|·|B|)。结果 [-1,1],越接近 1 越像
        )
        # ②稀疏向量索引
        index.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX", #倒排索引(搜索引擎同款):建立"token → 包含它的文档列表"。稀疏向量 99.9% 位置是 0,倒排只需检查查询中非零 token 对应的少量文档
            metric_type="IP"    #内积(点积):对应位置乘积再求和。对稀疏向量,等价于"共同 token 的权重乘积之和"
        )
        # ③标量索引(file_title 用于按文件删除/过滤,无索引的 VARCHAR 字段不能参与 filter)
        index.add_index(
            field_name="file_title",
            index_name="file_title_index",
            index_type="INVERTED"  # 倒排索引:支持 == 精确匹配过滤
        )
        logger.info(f"构建集合{collection_name}索引完成...")
        return index

# 3.建造者：插入器（数据插入和回填）
class _MilvusInsertBuilder:
    """职责：将数据插入milvus,以及回填chunks_id"""

    def __init__(self, client: MilvusClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def insert(self,chunks: list[dict]) -> list[dict]:
        logger.info(f"开始插入{len(chunks)} 块到milvus")
        # 插入
        # 幂等处理:同一文件重复导入时,先删除该文件的旧记录再插入,避免重复数据
        # 按文件标题过滤，已存在的文件再次导入，先删除库中的旧数据，插入新数据
        file_titles = {chunk.get("file_title") for chunk in chunks if chunk.get("file_title")}
        if file_titles:
            self._delete_existing_file_chunks(file_titles)

        inserted_result = self._client.insert(collection_name=self._collection_name, data=chunks)
        inserted_count = inserted_result.get("insert_count")
        ids = inserted_result.get("ids")
        # 回填状态
        self._fill_chunk_ids(chunks, ids)
        logger.info(f"完成插入{inserted_count}条记录,并且回填chunk_id到chunk中")
        return chunks

    def _delete_existing_file_chunks(self, file_titles):
        """按 file_title 删除集合中同文件的旧记录,保证同一文件重复导入幂等"""
        for file_title in file_titles:
            result = self._client.delete(
                collection_name=self._collection_name,
                filter=f'file_title == "{file_title}"'
            )
            logger.info(f"已清理文件【{file_title}】旧记录:{result.get('delete_count', 0)}条")


    @staticmethod
    def _fill_chunk_ids(chunks, ids):
        # zip 是 Python 内置函数,作用:把多个可迭代对象"按位置拉链式配对",每次迭代产出一个元组。
        for chunk, i_d in zip(chunks, ids):
            chunk["chunk_id"] = i_d
        """
        names = ["张三", "李四", "王五"]
        ages = [20, 30, 40]
        for name, age in zip(names, ages):
            print(name, age)
        # 张三 20
        # 李四 30
        # 王五 40
        """


# 4.门面构建
class ImportMilvusNode(BaseNode):
    """
    向量数据入库节点（门面角色）
    协调 Schema 构建、 索引构建  、 数据插入

    主要职责：
    上一节点传出的 chunks(已有 dense_vector + sparse_vector)
            ↓
    参数校验:过滤无混合向量的无效块,确定向量维度 dim
            ↓
    获取milvus客户端,确保集合存在(不存在则构建 schema + 索引并创建)
            ↓
    幂等处理:按 file_title 删除旧记录,避免重复数据
            ↓
    批量插入 milvus → 回填自增 chunk_id 到每个 chunk
            ↓
    state['chunks'] 更新 → 导入流程结束
    """

    name = "import_milvus_node"

    def process(self, state:ImportGraphState) -> ImportGraphState:
        # 1.参数校验(dim：向量维度)
        validated_chunks, dim = self._validate_get_inputs(state)

        # 2.获取milvus客户端
        milvus_client = StorageClients.get_milvus_client()
        if milvus_client is None:
            return state

        # 3.获取集合名称
        collection_name = getattr(self.config, "chunks_collection")   # 等价self.config.chunks_collection

        # 4.确保集合存在,不存在就创建
        self.ensure_has_collection(milvus_client, collection_name, dim)

        # 5.插入
        inserter = _MilvusInsertBuilder(client=milvus_client, collection_name=collection_name)
        final_chunks = inserter.insert(chunks=validated_chunks)     #保存并回填了chun_id

        # 6.更新状态
        state["chunks"] = final_chunks

        return state

    def _validate_get_inputs(self, state: ImportGraphState) -> tuple[list, int]:
        """参数校验"""
        chunks = state.get("chunks")

        if not chunks:
            raise ValidationError("待入库的chunks不存在", self.name)
        validated_chunks = []
        for chunk in chunks:
            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                validated_chunks.append(chunk)
            else:
                self.logger.error("待入库的切块chunk的混合向量不存在", self.name)

        if not validated_chunks:
            raise ValidationError("入库的chunks无效", self.name)

        dim = len(validated_chunks[0].get("dense_vector"))
        self.logger.info(f"导入milvus向量数据库的有效块：{len(validated_chunks)},且chunk的向量维度:{dim}")
        return validated_chunks, dim

    def ensure_has_collection(self, milvus_client, collection_name: str, dim: int, delete_flag: bool = False):
        """确保集合存在"""
        self.log_step("step2", f"确保集合【{collection_name}】存在")
        # delete_flag = True 可进行删除集合操作
        has_collection = milvus_client.has_collection(collection_name=collection_name)
        # 集合已存在且不强制重建：直接复用
        if has_collection and not delete_flag:
            return
        # 集合已存在且强制重建：先删除
        if has_collection:
            milvus_client.drop_collection(collection_name=collection_name)
            self.logger.info(f"milvus集合【{collection_name}】已被删除")
        # 创建集合
        schema = _MilvusSchemaBuilder.build(milvus_client, dim)
        index = _MilvusIndexBuilder.build(milvus_client, collection_name)
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index
        )
        self.logger.info(f"milvus集合【{collection_name}】创建成功")



if __name__ == "__main__":
    from pathlib import Path
    import json
    setup_logging()

    temp_dir = Path(r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\华为擎云 L420x 用户指南-(华为擎云 L420x-Axxx,UOS&KOS_01,zh-cn)\tem_doc")

    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        logger.error(f"文件找不到：{input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        chunks1 = json.load(f)

    state1: ImportGraphState = {
        "chunks":chunks1,
    }

    node1 = ImportMilvusNode()
    final_chunks1 = node1.process(state1)

    #备份文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks1, f, ensure_ascii=False, indent=4)
        logger.info(f"备份临时文件{output_path}成功")





