import math
from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients

"""
Rerank重排序节点
使用Reranker模型对RRF融合结果和网络搜索结果进行重排序，
并通过优化版断崖检测实现动态 TopK截断。
"""


class RerankNode(BaseNode):
    """Rerank  重排序节点
        流程：
            合并多源文档 -> Reranker计算相关性 ->  断崖检测动态截断
    """

    name = "rerank_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行重排序"""

        # 1.获取query信息
        # 这行代码从 state 字典中获取用户查询语句：优先取 rewritten_query（改写后的查询），
        # 若为空或不存在，则回退取 original_query（原始查询），确保 user_query 始终有值。
        user_query = state.get("rewritten_query", "") or state.get("original_query", "")

        # 2.合并多源文档 【  RRF结果 + Web MCP结果】
        merged_multi_docs: List[Dict[str, Any]] = self._merge_multi_source__docs(state)

        # 3.Rerank精排
        reranked_docs: List[Dict[str, Any]] = self._rerank_merged_docs(user_query, merged_multi_docs)

        # 4.动态Top_k 截取 （断崖检查 + 绝对分数底线过滤）
        cutoff_docs = self._cliff_catoff(reranked_docs=reranked_docs)

        # 5.更新状态返回
        state["reranked_docs"] = cutoff_docs

        return state

    def _merge_multi_source__docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地 RRF文档 和  Web MCP搜索文档"""
        final_docs = []

        for rrf_doc in (state.get("rrf_chunks") or []):
            if not isinstance(rrf_doc, dict):
                continue
            content = rrf_doc.get("content", "").strip()
            if not content:
                continue

            title = rrf_doc.get("title", "").strip()
            chunk_id = rrf_doc.get("chunk_id")
            format_rrf_doc = self._format_rrf_docs(
                content=content,
                title=title,
                chunk_id=chunk_id, source="local"
            )
            final_docs.append(format_rrf_doc)

        for web_doc in (state.get("web_search_docs") or []):
            if not isinstance(web_doc, dict):
                continue
            content = web_doc.get("content", "").strip() or web_doc.get("snippet", "").strip()
            if not content:
                continue
            title = web_doc.get("title", "").strip()
            url = web_doc.get("url", "").strip()
            format_web_doc = self._format_rrf_docs(
                content=content,
                title=title,
                url=url,
                source="web"
            )
            final_docs.append(format_web_doc)

        self.logger.info(f"收集到准备Rerank精排的文档数量:{len(final_docs)}")
        self.logger.info(f"收集到准备Rerank精排的文档集合:{final_docs}")

        return final_docs

    @staticmethod
    def _format_rrf_docs(content: str, title: str, chunk_id=None, url: str = "", source: str = "") -> Dict[str, Any]:
        return {
            "content": content,
            "title": title,
            "chunk_id": chunk_id,
            "url": url,
            "source": source
        }

    def _rerank_merged_docs(self, user_query: str, merged_multi_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用Reranker模型对文档进行精排 """
        if not merged_multi_docs:
            return []

        rerank_client = AIClients.get_bge_m3_rerank_client()
        if not rerank_client:
            self.logger.error("重排序模型获取失败")
            return []

        # 准备问题和文档的句子
        query_doc_content_pairs = [(user_query, doc.get("content")) for doc in merged_multi_docs]

        try:
            # Reranker模型计算句子得分:   负无穷 - 正无穷
            rerank_scores = rerank_client.compute_score(sentence_pairs=query_doc_content_pairs)

            # isinstance(rerank_scores, float) or isinstance(rerank_scores, int)
            # 单文档防护    只有一个文档，返回的可能是float或int值。而不是 list ;   这里如果是一个文档的话，也封装成List。保证后续逻辑处理的类型一致.
            if isinstance(rerank_scores, (float, int)):
                rerank_scores = [rerank_scores]
            print(rerank_scores)  # [5.06640625, -4.7421875, 3.962890625, -9.4765625]

            # 没做归一化处理
            score_docs1 = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]
            print(f"没做归一化处理:{score_docs1}")

            # 归一化处理
            score_docs = [{**doc, "score": self._sigmoid(score)} for doc, score in
                          zip(merged_multi_docs, rerank_scores)]
            print(f"归一化处理:{score_docs}")

            sorted_score_docs = sorted(score_docs, key=lambda x: x["score"], reverse=True)
            print(f"排序后结果返回:{sorted_score_docs}")

            return sorted_score_docs
        except Exception as e:
            self.logger.error(f"重排序模型计算分数失败:{e}")
            return [{**doc, "score": None} for doc in merged_multi_docs]

    def _cliff_catoff(self, reranked_docs: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """断崖检测动态截断
        扫描所有相邻文档分数差，找到最大落差位置进行截断，
        同时保证至少返回 lower_bound 个文档。

        参数:
            reranked_docs: 按分数降序排列的文档列表
            rerank_min_top_k: 最少返回文档数
            rerank_max_top_k: 最多返回文档数

        返回值:
            截断后的文档列表
        """
        upper_bound = min(self.config.rerank_max_top_k, len(reranked_docs))
        lower_bound = min(self.config.rerank_min_top_k, upper_bound)

        if upper_bound <= 1:
            return reranked_docs[:upper_bound]

        cut_off = upper_bound
        max_gap = 0.0

        for i in range(0, upper_bound - 1):
            current_score = reranked_docs[i].get("score")
            next_score = reranked_docs[i + 1].get("score")

            if current_score is None or next_score is None:
                continue

            # 分数差值
            gap = current_score - next_score

            if gap >=self.config.rerank_gap_abs and gap > max_gap:  #self.config.rerank_gap_abs=0.15
                max_gap = gap
                cut_off = i + 1
                self.logger.info(f"位置{cut_off}发生断崖，gap={gap:.4f}")
        # 兜底：不管断崖在哪，至少保留lower_bound个
        cut_off = max(cut_off, lower_bound)
        return reranked_docs[:cut_off] #左闭开区间：包含起始位置，不包含结束位置

    @staticmethod #避免实例方法调用时第一个参数传入self
    def _sigmoid(score: float) -> float:
        """sigmoid归一化，将( -∞ , +∞ )   映射到 （0,1）
            exp(x) 就是以自然常数 e=2.71828 为底的指数函数
            exp(10) ≈ 22026  →  1 / (1 + 22026) ≈ 0.0000454  ≈ 0
            exp(-10) ≈ 0.0000454  →  1 / (1 + 0.0000454) ≈ 0.99995  ≈ 1
        """
        return 1.0 / (1.0 + math.exp(-score))


# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊", "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    for j, doc in enumerate(result["reranked_docs"], 1):
        score1 = doc.get('score')
        score_str = f"{score1:.4f}" if score1 is not None else "N/A"
        print(f"[{j}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")

    print("-" * 60)
    print("测试完成")


"""
============================================================
开始测试: 重排序节点 (RerankNode)
============================================================
【输入状态】:
  查询: 怎么测这块主板的短路问题？
  本地文档: 2 篇
  网络文档: 2 篇
------------------------------------------------------------
2026-06-30 23:51:32 - query.rerank_node - INFO - 收集到准备Rerank精排的文档数量:4
2026-06-30 23:51:32 - query.rerank_node - INFO - 收集到准备Rerank精排的文档集合:[{'content': '主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。', 'title': '主板维修手册', 'chunk_id': 'local_1', 'url': <class 'str'>, 'source': 'local'}, {'content': '今天中午去吃猪脚饭吧，这块主板外观很漂亮。', 'title': '闲聊', 'chunk_id': 'local_2', 'url': <class 'str'>, 'source': 'local'}, {'content': '主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。', 'title': '短路查修指南', 'chunk_id': None, 'url': 'https://example.com/repair', 'source': 'web'}, {'content': '苹果发布新款手机，A系列芯片性能提升20%。', 'title': '科技新闻', 'chunk_id': None, 'url': 'https://example.com/news', 'source': 'web'}]
2026-06-30 23:51:34 - knowledge.utils.client.base - INFO - bge_m3_rerank客户端初始化成功
You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.

[5.06640625, -4.7421875, 3.962890625, -9.4765625]

没做归一化处理:[{'content': '主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。', 'title': '主板维修手册', 'chunk_id': 'local_1', 'url': <class 'str'>, 'source': 'local', 'score': 5.06640625}, 
{'content': '今天中午去吃猪脚饭吧，这块主板外观很漂亮。', 'title': '闲聊', 'chunk_id': 'local_2', 'url': <class 'str'>, 'source': 'local', 'score': -4.7421875}, 
{'content': '主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。', 'title': '短路查修指南', 'chunk_id': None, 'url': 'https://example.com/repair', 'source': 'web', 'score': 3.962890625}, 
{'content': '苹果发布新款手机，A系列芯片性能提升20%。', 'title': '科技新闻', 'chunk_id': None, 'url': 'https://example.com/news', 'source': 'web', 'score': -9.4765625}]

归一化处理:[{'content': '主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。', 'title': '主板维修手册', 'chunk_id': 'local_1', 'url': <class 'str'>, 'source': 'local', 'score': 0.993734466224161}, 
{'content': '今天中午去吃猪脚饭吧，这块主板外观很漂亮。', 'title': '闲聊', 'chunk_id': 'local_2', 'url': <class 'str'>, 'source': 'local', 'score': 0.008644177936723585}, 
{'content': '主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。', 'title': '短路查修指南', 'chunk_id': None, 'url': 'https://example.com/repair', 'source': 'web', 'score': 0.9813464782682386}, 
{'content': '苹果发布新款手机，A系列芯片性能提升20%。', 'title': '科技新闻', 'chunk_id': None, 'url': 'https://example.com/news', 'source': 'web', 'score': 7.662101864956481e-05}]

排序后结果返回:[{'content': '主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。', 'title': '主板维修手册', 'chunk_id': 'local_1', 'url': <class 'str'>, 'source': 'local', 'score': 0.993734466224161}, 
{'content': '主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。', 'title': '短路查修指南', 'chunk_id': None, 'url': 'https://example.com/repair', 'source': 'web', 'score': 0.9813464782682386}, 
{'content': '今天中午去吃猪脚饭吧，这块主板外观很漂亮。', 'title': '闲聊', 'chunk_id': 'local_2', 'url': <class 'str'>, 'source': 'local', 'score': 0.008644177936723585}, 
{'content': '苹果发布新款手机，A系列芯片性能提升20%。', 'title': '科技新闻', 'chunk_id': None, 'url': 'https://example.com/news', 'source': 'web', 'score': 7.662101864956481e-05}]

【重排序结果】:
[1] score=0.9937 | local | 主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。...
[2] score=0.9813 | web   | 主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。...
[3] score=0.0086 | local | 今天中午去吃猪脚饭吧，这块主板外观很漂亮。...
------------------------------------------------------------
测试完成
Disconnected from server
2026-06-30 23:51:39 - query.rerank_node - INFO - 位置2发生断崖，gap=0.9727
"""