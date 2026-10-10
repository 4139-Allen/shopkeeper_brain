from typing import List, Dict, Any, Tuple


from knowledge.processor.query_process.base import setup_logging, BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    """RRF融合排序节点"""

    # 节点名称
    name = "rrf_node"

    def __init__(self):
        super().__init__()
        self._rrf_k = self.config.rrf_k  # 60
        self._top_k = self.config.rrf_max_results  # 10

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """RRF融合排序"""
        # 1.获取各路检索结果
        # 1.1 获取向量检索路的结果
        vector_search_chunks = state.get("embedding_chunks") or []

        # 1.2 获取 hyde向量检索路的结果
        hyde_search_chunks = state.get("hyde_embedding_chunks") or []

        # 2.格式规整化 为不同路的搜索结果设置不同的权重
        search_source = {
            "vector_search_chunks": (self._normalize_input(vector_search_chunks), 1.0),
            "hyde_search_chunks": (self._normalize_input(hyde_search_chunks), 1.0),
        }
        # 4.构建完整的rrf_inputs
        rrf_inputs:List[Tuple[List[Dict[str, Any]], float]] = list(search_source.values())

        # 5. 利用RRF的计算公式去获取到所有路查询到的所有chunk对应的score
        rrf_merge_results: List[Tuple[Dict[str, Any], float]] = self._rrf_merge(rrf_inputs, self._rrf_k, self._top_k)

        # 6.获取 rrf_chunks 只要文档，不要分数
        rrf_chunks = [doc for doc, _ in rrf_merge_results]

        # 7.记录分数范围，便于调试
        if rrf_merge_results:
            scores = [s for _, s in rrf_merge_results]
            self.logger.info(f"分数范围：【{min(scores):.6f} - {max(scores):.6f}】")

        # 6.更新状态并返回
        state["rrf_chunks"] = rrf_chunks

        return state

    @staticmethod
    def _normalize_input(rrf_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """统一处理各路检索到的结果"""
        diff_path_result = []

        # 1.判断参数是否存在
        if not rrf_input:
            return []

        # 2. 遍历该路的所有结果
        for doc in rrf_input:
            if not isinstance(doc, dict):
                continue
            entity = doc.get("entity")
            if not entity:
                continue
            diff_path_result.append(entity)

        return diff_path_result

    @staticmethod
    def _rrf_merge(rrf_inputs: List[Tuple[List[Dict[str, Any]], float]], _rrf_k: int, _top_k: int) -> List[
        Tuple[Dict[str, Any], float]]:
        """
        利用RRF公式计算每一个文档的总得分
        :param rrf_inputs: 各路的搜索结果  + 权重
        :param _rrf_k: 平滑参数，通常取60
        :param _top_k: 合并完之后返回的文档数量
        :return: 合并以及排序后的文档列表
        """

        chunk_scores = {}  # 存放所有chunk的RRF计算后: id-> 分数值
        chunk_data = {}  # 存放所有chunk:  id -> 文档数据

        # 1.迭代处理文档分数
        for rrf_input, weight in rrf_inputs:
            for i, doc in enumerate(rrf_input, 1):  # 从序号1开始,表示文档排名
                chunk_id = doc.get("chunk_id")
                if not chunk_id:
                    continue
                # RRF公式  score =+= weight / (k + rank)
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (_rrf_k + i)

                # 使用setdefault 保留首次遇到的文档版本
                chunk_data.setdefault(chunk_id, doc)

        # 2. 按照分数降序排序截取top_k条数据
        sorted_results = sorted([(chunk_data[chunk_id], score) for chunk_id, score in chunk_scores.items()],
                                key=lambda x: x[1], reverse=True)

        return sorted_results[:_top_k] if _top_k else sorted_results

# ================================================================== #
#                        测试入口                                   #
# ================================================================== #



if __name__ == "__main__":

    setup_logging()

    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state:QueryGraphState = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfNode()
    result = rrf_node.process(mock_state)

    print("\n【融合结果】:")
    for j, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{j}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")

"""
============================================================
开始测试: RRF 融合节点
============================================================
【输入状态】:
  embedding_chunks: 3 条
  hyde_embedding_chunks: 3 条
------------------------------------------------------------

【融合结果】:
[1] chunk_1 - 向量搜索结果#1
[2] chunk_2 - 向量搜索结果#2
[3] chunk_3 - 向量搜索结果#3
[4] chunk_4 - HyDE搜索结果#3
------------------------------------------------------------
测试完成
Disconnected from server
"""