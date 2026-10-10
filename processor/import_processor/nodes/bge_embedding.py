import json
import os

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import ValidationError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients


class BgeEmbeddingChunksNode(BaseNode):
    """
    主要职责：
    上一节点传出的 chunks(已有 item_name + content)
            ↓
    [content 拼接 item_name] → 分批(8条/批)
            ↓
    BGE-M3 encode_documents(批量)
            ↓
    拆出 dense(1024维) + sparse({token_id: weight})
            ↓
    注入每个 chunk 的 dense_vector（稠密） / sparse_vector（稀疏） 字段
            ↓
    state['chunks'] 更新 → 交给下游节点(写入 milvus)
    """

    name = "bge_embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.参数校验
        validated_chunks = self._validate_get_inputs(state)

        # 2.获取批量嵌入的阈值
        """
        # getattr(config, "embedding_batch_size", 16)
        # 相当于:config.embedding_batch_size   # 属性存在时
        # 但属性不存在时:
        #   config.embedding_batch_size → 直接抛 AttributeError,程序崩溃
        #   getattr(..., 16)             → 安静地返回 16
        """
        embedding_batch_chunk_size = getattr(self.config, "embedding_batch_size", 16)

        # 3. 准备分批嵌入
        total_length = len(validated_chunks)
        final_chunks = []
        for i in range(0, total_length, embedding_batch_chunk_size):
            batch = validated_chunks[i:i + embedding_batch_chunk_size]
            batch_chunks = self._process_batch_chunks(batch, i, total_length)
            final_chunks.extend(batch_chunks)

        # 4.更新&返回state
        state['chunks'] = final_chunks

        # 5.备份json文件
        self.backup_chunks_vector(state, final_chunks)

        return state

    def _validate_get_inputs(self, state):
        """验证参数"""
        self.log_step("step1", "数据校验")
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise ValidationError(f"chunks为空或者无效", self.name)
        self.logger.info(f"chunks数量：{len(chunks)}")
        return chunks

    def _process_batch_chunks(self, batch, start_index: int, total_length: int):
        """处理批量chunks"""
        # 1.循环处理chunk 中需要嵌入的内容，内容拼接=  item_name + \n content
        embedding_contents = []
        for _, chunk in enumerate(batch):
            item_name = chunk.get("item_name")
            content = chunk.get("content")
            embedding_content = f"{item_name}\n{content}"
            embedding_contents.append(embedding_content)

        # 2.批量嵌入
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
            embedding_result = bge_m3_client.encode_documents(documents=embedding_contents)
            if not embedding_result:
                self.logger.warning(f"嵌入后结果不存在")
                return batch
        except Exception as e:
            self.logger.warning(f"生成嵌入向量失败:{str(e)}")

        # 3.循环处理所有chunk的向量，以及注入到每一个chunk中
        for index, chunk in enumerate(batch):
            dense_vector = embedding_result['dense'][index].tolist()
            csr_array = embedding_result['sparse']
            start_index = csr_array.indptr[index]
            end_index = csr_array.indptr[index + 1]
            token_id = csr_array.indices[start_index:end_index].tolist()
            weight = csr_array.data[start_index:end_index].tolist()
            sparse_vector = dict(zip(token_id, weight))
            chunk['dense_vector'] = dense_vector
            chunk['sparse_vector'] = sparse_vector

        self.logger.info(f"开始批量处理 chunk  嵌入；批次{start_index + 1}-{start_index + len(batch)}/{total_length}")

        # 返回处理结果
        return batch

    def backup_chunks_vector(self, state:ImportGraphState, chunks):
        """备份生成的向量json文件"""
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return None

        os.makedirs(local_dir, exist_ok=True)   # exist_ok=True：如果目录已存在，不会抛出异常，直接跳过
        output_path = os.path.join(local_dir, "chunks_vector.json")

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
                self.logger.info(f"备份成功：{local_dir}")
        except Exception as e:
            self.logger.info(f"备份失败：{e}")



if __name__ == "__main__":

    setup_logging()

    node1 = BgeEmbeddingChunksNode()

    chunks_file_path = r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\华为擎云 L420x 用户指南-(华为擎云 L420x-Axxx,UOS&KOS_01,zh-cn)\tem_doc\chunks2_item_name.json"

    with open(chunks_file_path, "r", encoding="utf-8") as f:
        chunks1 = json.load(f)

    state1:ImportGraphState = {
        "chunks":chunks1,
        "file_dir":r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\华为擎云 L420x 用户指南-(华为擎云 L420x-Axxx,UOS&KOS_01,zh-cn)\tem_doc"
    }

    node1.process(state1)




