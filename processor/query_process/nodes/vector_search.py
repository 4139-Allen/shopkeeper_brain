import json
from pymilvus import AnnSearchRequest

from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import item_names_filter, create_hybrid_search_requests, execute_hybrid_search_query


class VectorSearchNode(BaseNode):
    """向量搜索节点"""

    # 节点名称
    name = "vector_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """节点处理流程"""

        # 1.参数校验
        validated_query, validate_item_names = self._validate_state(state)

        # 2.获取嵌入模型
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败: {e}")
            return state

        # 3.获取milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"milvus客户端获取失败: {e}")
            return state

        # 4.生成混合向量   稠密向量 + 稀疏向量
        try:
            embed_query = generate_bge_m3_hybrid_vectors(bge_m3_client, [validated_query], is_query=True)
        except Exception as e:
            self.logger.error(f"问题[{validated_query}],向量生成失败: {e}")
            return state

        # 5.构建过滤条件
        expr, expr_params = item_names_filter(validate_item_names)

        # 6.创建搜索请求
        hybrid_search_request: list[AnnSearchRequest] = create_hybrid_search_requests(
            dense_vector=embed_query['dense'][0],
            sparse_vector=embed_query['sparse'][0],
            expr=expr,
            expr_params=expr_params,
            limit=5)

        # 7.执行混合检索
        hybrid_search_resp = execute_hybrid_search_query(milvus_client=milvus_client,
                                                         collection_name=self.config.chunks_collection,
                                                         search_requests=hybrid_search_request,
                                                         output_fields=["chunk_id", "content", "item_name", "title"])

        if not hybrid_search_resp or not hybrid_search_resp[0]:
            self.logger.error(f"问题[{validated_query}],混合检索数据不存在")
            return state

        self.logger.info(f"问题[{validated_query}],混合检索数据成功：{hybrid_search_resp[0]}")
        # 8.返回结果
        return {"embedding_chunks": hybrid_search_resp[0]}

    def _validate_state(self, state: QueryGraphState) -> tuple[str, list[str]]:
        """校验输入参数"""
        # 1.获取参数
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")

        # 2.校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 3.返回
        return rewritten_query, item_names


# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    from knowledge.processor.query_process.state import create_default_state
    init_state: QueryGraphState | dict = create_default_state(
        rewritten_query="万用表如何测量电阻",
        item_names=["RS-12 数字万用表"]
    )

    vector_search = VectorSearchNode()
    result = vector_search.process(init_state)
    for r in result.get("embedding_chunks", []):
        print(json.dumps(r, indent=4, ensure_ascii=False))

"""
{
    "chunk_id": 467121384825486677,
    "distance": 0.7390990853309631,
    "entity": {
        "title": "## 电阻测量",
        "chunk_id": 467121384825486677,
        "content": "## 电阻测量\n\n\n警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n\n1. 将功能转盘置于最高电阻Ω位置.\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n\n3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n\n4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n\n![万用表电阻测量接线示意图](http://192.168.6.150:9000/knowledge-base-files/万用表RS-12的使用/de9dde2732fe81a213e8fd32e98b790548145c7c796ec443d5f6f0cb576cd3e1.jpg)\n",
        "item_name": "RS PRO RS-12 数字万用表"
    }
}
{
    "chunk_id": 467121384825486675,
    "distance": 0.42088091373443604,
    "entity": {
        "title": "## 交流电压测量",
        "chunk_id": 467121384825486675,
        "content": "## 交流电压测量\n\n\n警告：谨防触电。\n\n若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n![交流电压测量时表笔正确接入电路的示意图](http://192.168.6.150:9000/knowledge-base-files/万用表RS-12的使用/3e257858115a629b9112ea2e2c75344a2c2d01f2e6e110ab28d41809719fc433.jpg)\n\n1. 将功能转盘置于V AC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。\n\n在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。\n",
        "item_name": "RS PRO RS-12 数字万用表"
    }
}
...
"""

"""
        问题1： embedding_util.py

            encode_queries vs encode_documents 区别
            BGE-M3 采用非对称检索策略，两个方法会在文本前添加不同的指令前缀（instruction prefix），让模型知道当前编码的是"问题"还是"文档"，从而生成更适合匹配的向量。
            方法               用途          适用场景         输入参数名
            ----------------------------------------------------------
            encode_queries   编码查询文本    用户搜索问题      queries
            encode_documents 编码文档内容    入库存储的文档    documents


            # ... existing code ...
            def generate_bge_m3_hybrid_vectors(model: BGEM3EmbeddingFunction, texts: List[str], is_query: bool = True):

                为文本生成混合向量嵌入（稠密 + 稀疏）
                Args:
                    model: BGE-M3嵌入模型
                    texts: 要生成嵌入的文本列表
                    is_query: True用encode_queries（查询场景），False用encode_documents（文档入库场景）
                Returns:
                    {"dense": [...], "sparse": [...]}
                Raises:
                    ValueError: 输入参数无效
                    RuntimeError: 嵌入生成失败

                # 1. 参数校验
                if not texts:
                    raise ValueError("texts 不能为空")

                if not all(isinstance(doc, str) and doc.strip() for doc in texts):
                    raise ValueError("texts 中存在无效元素（空字符串或非字符串类型）")


                # 2. 生成嵌入
                try:
                    if is_query:
                        embedding_result = model.encode_queries(texts)
                    else:
                        embedding_result = model.encode_documents(texts)
                except Exception as e:
                    raise RuntimeError(f"BGE-M3 嵌入生成失败: {e}") from e

                # 3. 校验嵌入结果
                if 'dense' not in embedding_result or 'sparse' not in embedding_result:
                    raise RuntimeError(f"嵌入结果缺少必要字段，实际返回: {list(embedding_result.keys())}")

                # 5. 解析稀疏向量（CSR 矩阵 → dict）
                try:
                    processed_sparse = []
                    csr_array = embedding_result['sparse']

                    for index in range(len(texts)):
                        start = csr_array.indptr[index]
                        end = csr_array.indptr[index + 1]
                        token_ids = csr_array.indices[start:end].tolist()
                        weights = csr_array.data[start:end].tolist()
                        processed_sparse.append(dict(zip(token_ids, weights)))
                except (IndexError, AttributeError) as e:
                    raise RuntimeError(f"稀疏向量解析失败（CSR 矩阵结构异常）: {e}") from e

                # 6. 返回
                return {
                    "dense": [den.tolist() for den in embedding_result["dense"]],
                    "sparse": processed_sparse
                }

            # 查询场景（默认 is_query=True）
            embed_query = generate_bge_m3_hybrid_vectors(bge_m3_client, ["万用表如何测量电阻"])

            # 文档入库场景
            embed_doc = generate_bge_m3_hybrid_vectors(bge_m3_client, ["文档内容..."], is_query=False)


        问题2：item_name_recognition.py  95-96
            # self.logger.info(f"LLM提取到商品名称：{llm_result}")
            # return llm_result

            try:
                parsed = json.loads(llm_result)
                item_name = parsed.get("item_name", "").strip()
                if not item_name:
                    self.logger.info(f"LLM返回JSON中item_name为空,降级使用标题:{file_title}")
                    return file_title
                self.logger.info(f"LLM提取到商品名：{item_name}")
                return item_name

"""