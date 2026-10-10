import json
import re
from json import JSONDecodeError
from typing import Dict, Any, List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.config import get_config

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import execute_hybrid_search_query, create_hybrid_search_requests
from knowledge.utils.mongo_history_util import get_recent_messages, update_message_item_names


class ItemNameExtractor:
    """
    商品名称提取器（解决用户提出的模糊问题，llm返回处理后的用户问题，增强检索的准确率）
    基于用户的原始问题和历史对话提取用户真正想问的商品名称
    """

    def __init__(self, logger):
        self.logger = logger

    def extract_item_name(self, original_query: str, history_text: str) -> Dict[str, Any]:
        """LLM根据用户的原始问题和历史会话内容提前商品名称"""

        # 1.默认返回结果
        result: Dict[str, Any] = {
            "item_names": [],
            "rewritten_query": original_query
        }

        # 2.获取llm客户端
        llm_client = AIClients.get_llm_openai()

        # 3.判断客户端是否存在，不存在则直接返回默认值
        if llm_client is None:
            return result

        # 4.调用LLM获取提前的商品名称
        system_prompt = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"
        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(query=original_query,
                                                         history_text=history_text if history_text else '暂无上下文')
        llm_response = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        # 5.判断是否获取到商品名称，没获取到直接返回默认结果
        llm_content = llm_response.content.strip()
        if not llm_content:
            return result

        try:
            # 6.清洗数据
            parsed_result = self._clean_parse(llm_content)

            # 7.清洗后赋值
            result['rewritten_query'] = parsed_result.get('rewritten_query', original_query)
            result['item_names'] = parsed_result.get('item_names', [])
        except Exception as e:
            self.logger.error(f"清洗以及解析LLM的输出失败: {str(e)}")

        # 8.返回结果
        return result

    @staticmethod
    def _clean_parse(llm_content):
        """清洗LLM的输出结果。
        这段代码用于清理 LLM 返回的响应文本：
            第一行：去除开头的 Markdown 代码块标记（如
                     ``` `` 或 `` ```json ``）及紧随的空白字符
            2. **第二行**：去除结尾的 Markdown 代码块标记 `` ``` `` 及其前面的空白字符

            **目的**：从 LLM 返回的 JSON 响应中提取纯 JSON 内容，去掉包裹的 Markdown 代码块格式。
        """
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        try:
            # 将 LLM 输出的json字符串转换为json对象
            parsed_llm_result: Dict[str, Any] = json.loads(content)

            # LLM提取到的商品名称
            raw_item_names = parsed_llm_result.get("item_names")

            if not isinstance(raw_item_names, list):
                clean_item_names = []
            else:
                clean_item_names = [raw_item for raw_item in raw_item_names if raw_item.strip()]

            raw_rewritten_query = parsed_llm_result.get("rewritten_query")
            clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()
            return {
                "item_names": clean_item_names,
                "rewritten_query": clean_rewritten_query
            }

        except JSONDecodeError as e:
            raise ValueError(f"JSON反序列化LLM 的输出结果失败: {str(e)}")


class ItemNameAligner:
    """
    商品对齐器：
    主要职责：
        1.查询向量数据库
        2.评分对齐
        3.分数差异过滤
    """

    def __init__(self, logger):
        self.logger = logger

    def match_align_filter(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        """执行匹配，对齐，过滤三步流程"""

        # 1.查询向量数据库，解析结果，返回查询结果列表：字典为： 提取商品名称 - 得分
        search_result: List[Dict[str, Any]] = self._match_vector(item_names)

        # 2.评分对齐
        confirmed, options = self._item_name_score_align(search_result)

        # 3.分数差异过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)

        return confirmed, options

    def _match_vector(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """根据LLM提取的商品名称，查询向量数据库"""

        # 1.定义默认返回结果
        search_results: List[Dict[str, Any]] = []

        # 2.获取milvus客户端
        milvus_client = StorageClients.get_milvus_client()
        if milvus_client is None:
            self.logger.error(f"获取Milvus客户端失败")
            return search_results

        # 3.获取嵌入模型
        embedding_model = AIClients.get_bge_m3_client()
        if embedding_model is None:
            self.logger.error(f"获取嵌入模型失败")
            return search_results

        # 4.生成搜索条件的混合向量

        valid_item_names = [name for name in item_names if name and len(name.strip()) >= 2]
        if not valid_item_names:
            self.logger.warning(f"过滤后无有效商品名，原始列表：{item_names}")
            return search_results
        try:
            generate_hybrid_embedding = generate_bge_m3_hybrid_vectors(embedding_model, valid_item_names)
        except Exception as e:
            self.logger.error(f"商品名确认-生成混合向量失败: {str(e)}")
            return search_results

        # 5.迭代搜索装填
        config = get_config()
        for index, extract_item_name in enumerate(valid_item_names):
            # 5.1 构建查询条件
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=generate_hybrid_embedding['dense'][index],
                sparse_vector=generate_hybrid_embedding['sparse'][index],
            )

            # 5.2 根据查询条件查询向量数据库数据
            hybrid_search_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=config.item_name_collection,
                search_requests=hybrid_search_requests,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                output_fields=["item_name"]
            )

            # 5.3解析查询结果
            item_name_search_result = {
                "extracted_name": extract_item_name,
                "matches": [
                    {"item_name": h["entity"]["item_name"], "score": h["distance"]} for h in
                    (hybrid_search_result[0] if hybrid_search_result else [])
                ]
            }
            search_results.append(item_name_search_result)

        # 6.返回结果
        return search_results

    @staticmethod
    def _item_name_score_align(search_result: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """根据评分对齐商品名称"""
        confirmed = []
        options = []

        # 1.迭代查询结果
        for item_name_search_result in search_result:
            # 提取的商品名称
            extracted_name = item_name_search_result.get("extracted_name")
            # 提取的商品名称对应的milvus中，名称，分数
            matches = sorted(item_name_search_result.get("matches"), key=lambda x: x['score'], reverse=True)

            # 过滤出高分区
            high = [m for m in matches if m.get('score') >= 0.7]

            if high:
                extract = next((h for h in high if str(h['item_name']) == extracted_name), None)
                if extract:
                    picked = extract["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                elif len(high) == 1:
                    picked = high[0]["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                else:
                    for h in high[:3]:
                        picked = h.get("item_name")
                        if picked not in options and picked not in confirmed:
                            options.append(picked)
            else:
                mid = [m for m in matches if
                       m['score'] >= 0.6 and m.get('item_name') not in options and m.get('item_name') not in confirmed]
                if mid:
                    for m in mid[:3]:
                        picked = m.get('item_name')
                        options.append(picked)

        return confirmed, options[:3]

    @staticmethod
    def _item_name_score_filter(confirmed: List[str], search_results: List[Dict[str, Any]]):
        """分数差异过滤,剔除误判"""
        item_name_score = {}
        for search_result in search_results:
            matches = search_result.get("matches")
            for m in matches:
                score = m.get("score")
                item_name = m.get("item_name")
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name) or 0, score)
        sorted_item_name_score = sorted(item_name_score.items(), key=lambda x: x[1], reverse=True)
        max_item_name_score = sorted_item_name_score[0][1]
        return [name for name, score in item_name_score.items() if max_item_name_score - score <= 0.15]


class ItemNameConfirmNode(BaseNode):
    """
    商品名称确认节点

        流程：获取历史 -> LLM提取商品名称 -> 向量匹配 -> 评分对齐  -> 更新状态  ->   历史回填
    """

    name = "item_name_confirm_node"

    def __init__(self):
        super().__init__()
        self._item_name_extractor = ItemNameExtractor(self.logger)
        self._item_name_aligner = ItemNameAligner(self.logger)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1.获取历史
        # 1.1. 获取原始问题
        original_query = state.get("original_query")
        session_id = state.get("session_id")

        # 1.2获取历史会话并拼串
        chat_history = get_recent_messages(session_id)

        history_text = ""
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("text", "")
            history_text += f"{role}:{content}\n"

        # 2.LLM提取商品名称,并且清理代码块后的结果字符串
        clean_llm_result = self._item_name_extractor.extract_item_name(original_query, history_text)

        item_names = clean_llm_result.get("item_names")
        rewritten_query = clean_llm_result.get("rewritten_query")

        # 3.向量匹配: 查询向量数据库  && 过滤
        if item_names:
            confirmed, options = self._item_name_aligner.match_align_filter(item_names)
        else:
            confirmed, options = [], []

        # 4.决策分支，更新sate
        self._decide(state, item_names, confirmed, options, rewritten_query)

        # 5.历史回填
        if confirmed:
            ids_to_update = [
                str(msg["_id"] for msg in chat_history if not msg.get("item_names"))
            ]
            if ids_to_update:
                try:
                    update_message_item_names(ids_to_update, confirmed)
                except Exception as e:
                    self.logger.warning(f"回填历史 item_name 失败: {e}")

        state['history'] = chat_history

        return state

    @staticmethod
    def _decide(state: QueryGraphState, item_names: List[str], confirmed: List[str], options: List[str],
                rewritten_query: str):
        """根据对齐结果更新state"""
        if confirmed:
            state["rewritten_query"] = rewritten_query
            state["item_names"] = confirmed  # ！！！！！！！！！！！！！！！！！
        elif options:
            state["answer"] = f"我不确定您指的是哪款产品。, 您是在询问以下产品吗: {'、'.join(options)}？"
        else:
            state['answer'] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"


# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    from knowledge.processor.query_process.state import create_default_state
    init_state: QueryGraphState | dict = create_default_state(
        task_id="task_001",
        original_query="RS-12 数字万用表怎么测试电阻？以及华为擎云L420 用户手册 中包含操作环境嘛？"
    )

    print(f"输入: {json.dumps(init_state, ensure_ascii=False, indent=2)}\n")

    node_item_name_confirm = ItemNameConfirmNode()
    result1 = node_item_name_confirm.process(init_state)
    print(f"确认商品: {result1.get('item_names')}")
    print(f"改写查询: {result1.get('rewritten_query')}")
    if result1.get("answer"):
        print(f"拦截回复: {result1.get('answer')}")
