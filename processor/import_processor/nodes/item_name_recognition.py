from langchain_core.messages import SystemMessage, HumanMessage
import json
import os
from pymilvus import DataType
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.prompt.import_process_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    """
    主要职责：
    上一节点传出的 chunks(已有 title + content) + file_title
            ↓
    拼接前 k 个 chunk 内容作为 LLM 识别上下文
            ↓
    LLM 识别商品名(失败/空值降级为 file_title)
            ↓
    BGE-M3 将商品名向量化(dense 1024维 + sparse)
            ↓
    商品名向量存入 milvus(item_name 集合)
            ↓
    回填 item_name 到每个 chunk 与 state → 交给下游节点(向量嵌入)
    """

    name = "item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.参数校验
        self.log_step("step1", "校验上一节点传入参数是否合法")
        file_title, chunks, item_name_chunks_k, item_name_chunk_size = self._validate_state(state)

        # 2.构建商品名识别上下文
        self.log_step("step2", "提取LLM识别商品名所需上下文")
        item_name_recognition_context = self._prepare_item_name_recognition_context(chunks, item_name_chunks_k,item_name_chunk_size)

        # 3.LLM商品名识别
        self.log_step("step3", "调用大模型进行商品识别")
        item_name = self._recognition_name(file_title, item_name_recognition_context)

        # 4.向量化
        self.log_step("step4", "将LLM生成的商品名向量化")
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        # 5.存储到milvus
        self.log_step("step5", "向量化后的商品名存入milvus")
        self._insert_milvus(file_title, item_name, dense_vector, sparse_vector, self.config.item_name_collection)

        # 6.回填
        self.log_step("step6", "将LLM生成的商品名更新到state")
        self._fill_item_name(item_name, state, chunks)

        # 7.备份
        self.log_step("step7",f"本地备份: {state.get('file_dir')}")
        self._backup_chunks(state, chunks)

        # 8.返回
        return state

    def _validate_state(self, state: ImportGraphState) -> tuple[str, list[str], int, int]:
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str)

        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        item_name_chunk_k = self.config.item_name_chunk_k
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError(node_name=self.name, message="item_name_chunk_k为空或者无效")

        item_name_chunk_size = self.config.item_name_chunk_size
        if not item_name_chunk_size or item_name_chunk_size <= 0:
            raise ValidationError(node_name=self.name, message="item_name_chunk_size为空或者无效")

        return file_title, chunks, item_name_chunk_k, item_name_chunk_size

    @staticmethod
    def _prepare_item_name_recognition_context(chunks, item_name_chunks_k, item_name_chunk_size) -> str:
        # 取chunks中前item_name_chunks_k chunk的 content内容进行拼接，长度不能超过item_name_chunk_size
        total = 0
        final_context = []
        for index, chunk in enumerate(chunks[:item_name_chunks_k]):
            if not isinstance(chunk, dict):
                continue

            chunk_content = chunk.get("content")
            content = f"【切片】-{index}-{chunk_content}"
            if total + len(content) > item_name_chunk_size:
                break
            total = total + len(content)
            final_context.append(content)
        return "\n".join(final_context)

    def _recognition_name(self, file_title, item_name_recognition_context) -> str:
        """LLM商品名识别"""
        try:
            llm_client = AIClients.get_llm_openai(response_format=True)
            # import_prompt.py  提示词模板
            user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title,
                                                                context=item_name_recognition_context)

            llm_response = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])

            llm_result = llm_response.content.strip()
            if not llm_result or llm_result == "UNKNOWN":
                self.logger.info(f"LLM未识别出商品名,降级使用标题:{file_title}")
                return file_title
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
            except (json.JSONDecodeError, AttributeError):
                self.logger.warning(f"LLM返回内容无法解析为JSON,原始值:[{llm_result}],降级使用标题:{file_title}")
                return file_title

        except Exception as e:
            self.logger.error(f"LLM商品名识别异常降级使用标题：{file_title},异常:{e}")
            return file_title

    def _embedding_item_name(self, item_name) -> tuple[list[float] | None, dict[int, float] | None]:
        """商品名称向量化"""
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
            vector_result = bge_m3_client.encode_documents([item_name])
            dense_vector = vector_result['dense'][0].tolist()   #稠密向量（1024维，擅长语义相似("万用表" ↔ "测量仪表")）
            start_index = vector_result['sparse'].indptr[0]
            end_index = vector_result['sparse'].indptr[1]
            token_id = vector_result['sparse'].indices[start_index:end_index].tolist()
            weight = vector_result['sparse'].data[start_index:end_index].tolist()
            sparse_vector = dict(zip(token_id, weight))         #稀疏(词项权重) 精确匹配("UT890D+" ↔ "UT890D+")
            return dense_vector, sparse_vector
        except ConnectionError as e:
            self.logger.error(f"BGE-M3客户端获取失败{e}")
            return None, None
        except Exception as e:
            self.logger.error(f"商品名【{item_name}】,BGE-M3向量化异常{e}")
            return None, None

    def _insert_milvus(self, file_title, item_name, dense_vector, sparse_vector, item_name_collection):
        # 1.校验稠密向量和稀疏向量
        if not dense_vector or not sparse_vector:
            self.logger.error(f"文档【{file_title}】对应的商品名【{item_name}】，向量化生成不完整")
            return

        # 2.获取milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"milvus客户端获取失败，异常:{e}")
            return

        # 3.判断集合是否存在，不存在则创建集合
        try:
            if not milvus_client.has_collection(item_name_collection):
                self.logger.info(f"集合【{item_name_collection}】不存在，创建集合")
                self._create_item_name_collection(item_name_collection, milvus_client)

            # 4.保存向量数据到milvus
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }

            result = milvus_client.insert(collection_name=item_name_collection, data=[data])
            self.logger.info(f"向量数据保存结果：{result}")
        except Exception as e:
            self.logger.error(f"向量数据保存异常，异常:{e}")

    def _create_item_name_collection(self, item_name_collection, milvus_client):
        """创建商品名集合"""
        schema = milvus_client.create_schema()
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_param = milvus_client.prepare_index_params()

        index_param.add_index(field_name="dense_vector",
                              index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE")

        index_param.add_index(field_name="sparse_vector",
                              index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        milvus_client.create_collection(collection_name=item_name_collection, schema=schema, index_params=index_param)
        self.logger.info(f"集合【{item_name_collection}】创建成功")

    @staticmethod
    def _fill_item_name(item_name, state, chunks):
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["item_name"] = item_name

    def _backup_chunks(self, state, chunks):
        """
        将切分结果备份到json文件
        :param state:
        :param chunks:
        :return:
        """
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return

        os.makedirs(local_dir, exist_ok=True) #exist_ok=True：如果目录已存在，不会抛出异常，直接跳过
        output_path = os.path.join(local_dir, "chunks2.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"备份失败：{e}")


if __name__ == "__main__":
    setup_logging()

    node1 = ItemNameRecognitionNode()

    chunks_file_path = r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\chunks.json"
    with open(chunks_file_path, "r", encoding="utf-8")as f:
        chunks1 = json.load(f)

    state1:ImportGraphState = {
        "chunks":chunks1,
        "file_title":"万用表RS-12的使用",
        "file_dir":r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    }

    node1.process(state1)




