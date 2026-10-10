from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query


class HyDeSearchNode(BaseNode):
    """
    HyDE 检索节点
    流程：
        参数校验 -> LLM生成假设文档 -> 拼接原查询 ->  向量化 ->  混合检索
    """

    # 节点名称
    name = "hyde_search_node"

    # Python 3.10 之后可以用 | 替代 Union，写法更简洁：
    #def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
    def process(self, state: QueryGraphState) -> QueryGraphState | Dict[str, Any]:

        """
        执行HyDE 检索
        :param state: 含  rewritten_query和item_names
        :return: {"hyde_embedding_chunks":[...]} 搜索结果列表
        """

        # 1.参数校验
        validated_query, validate_item_names = self._validate_query_inputs(state)

        # 2.生成假设性文档
        hy_document = self._generate_by_document(validated_query, validate_item_names)

        # 3.获取依赖 + 文档拼接 + 向量化
        bge_m3_client = AIClients.get_bge_m3_client()
        milvus_client = StorageClients.get_milvus_client()
        if not bge_m3_client or not milvus_client:
            return state
        embedding_document = f"{validated_query}\n{hy_document}"

        if not embedding_document:
            self.logger.warning("embedding_document 为空，跳过 HyDE 检索")
            return state

        embedding_result = generate_bge_m3_hybrid_vectors(bge_m3_client, [embedding_document], is_query=True)
        if not embedding_result:
            return state

        # 4.构建过滤条件
        filter_expr = self._item_names_filter_expr(validate_item_names)

        # 5.创建搜索请求 +执行检索
        hybrid_search_requests = create_hybrid_search_requests(dense_vector=embedding_result['dense'][0],
                                                               sparse_vector=embedding_result['sparse'][0],
                                                               expr=filter_expr,
                                                               limit=5)

        resp = execute_hybrid_search_query(milvus_client=milvus_client,
                                           collection_name=self.config.chunks_collection,
                                           search_requests=hybrid_search_requests,
                                           output_fields=["chunk_id", "content", "item_name", "title"])

        if not resp or not resp[0]:
            return state

        # 6.返回结果
        return {"hyde_embedding_chunks": resp[0]}

    def _validate_query_inputs(self, state):
        """参数校验"""
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        return rewritten_query, item_names

    def _generate_by_document(self, validated_query: str, validate_item_names: List[str]) -> str:
        """使用LLM生成假设性文档"""
        # 1.获取LLM客户端
        llm_client = AIClients.get_llm_openai(False)

        # 2.判断客户端是否存在
        if not llm_client:
            return ""

        # 3.获取系统提示词和用户提示词
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(rewritten_query=validated_query, item_names=validate_item_names)
        system_prompt = f"您是一位{validate_item_names}的技术文档领域的专家，主要擅长编写技术文档，操作手册，文档规格说明。"
        try:
            # 4.获取AIMessage
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 5.获取内容
            llm_content = getattr(llm_response, "content", "")

            # 6.判断是否存在
            if not llm_content:
                return ""

            # 7.返回
            return llm_content
        except Exception as e:
            self.logger.error(f"LLM调用失败：{e}")
            return ""

    @staticmethod
    def _item_names_filter_expr(validate_item_names: List[str]) -> str:
        """构建商品名过滤表达式"""
        quoted = ",".join(f'"{v}"' for v in validate_item_names)
        return f"item_name in [{quoted}]"


# ================================================================== #
#                        测试入口                                   #
# ================================================================== #

if __name__ == "__main__":
    from knowledge.processor.query_process.base import setup_logging

    setup_logging()

    print("=" * 60)
    print("开始测试: HyDE 检索节点 (HydeSearchNode)")
    print("=" * 60)

    mock_state:QueryGraphState = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS-12 数字万用表"]
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  商品: {mock_state['item_names']}")
    print("-" * 60)

    node = HyDeSearchNode()
    result = node.process(mock_state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n【HyDE 检索结果】: {len(chunks)} 条")
    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(f"  [{i}] chunk_id={entity.get('chunk_id')} "
              f"item_name={entity.get('item_name')} "
              f"distance={chunk.get('distance', 'N/A')}")
        content = entity.get("content", "")
        print(f"      内容: {content[:80]}...")

    print("-" * 60)
    print("测试完成")


"""
============================================================
开始测试: HyDE 检索节点 (HydeSearchNode)
============================================================
【输入状态】:
  查询: RS-12 数字万用表如何测量直流电压？
  商品: ['RS PRO RS-12 数字万用表']
------------------------------------------------------------
INFO:knowledge.utils.client.base:ChatOpenAI LLM 客户端初始化成功
INFO:httpx:HTTP Request: POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions "HTTP/1.1 200 OK"
INFO:FlagEmbedding.finetune.embedder.encoder_only.m3.runner:loading existing colbert_linear and sparse_linear---------
INFO:knowledge.utils.client.base:bge_m3客户端初始化成功
You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
INFO:knowledge.utils.milvus_util:Milvus 混合搜索完成，共处理 1 个查询，总计找到 5 个结果

【HyDE 检索结果】: 5 条
  [1] chunk_id=467121384825486674 item_name=RS PRO RS-12 数字万用表 distance=0.8118578195571899
      内容: ## 直流电压测量


注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。

1. 将功能转盘置于V DC的位置。

2. 将黑色表笔插入负极...
  [2] chunk_id=467121384825486676 item_name=RS PRO RS-12 数字万用表 distance=0.8106058835983276
      内容: ## 直流电流测量


注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。

![直流电流测量接线示意图（10A档位限时不超过30秒）](h...
  [3] chunk_id=467121384825486662 item_name=RS PRO RS-12 数字万用表 distance=0.7905207872390747
      内容: ## 安全手册


为了您的安全，请在使用本仪表之前仔细阅读该手册:

使用本表时，请勿将输入的测量值超出其所允许的量程范围。



- 【功能】：输入量程为最...
  [4] chunk_id=467121384825486675 item_name=RS PRO RS-12 数字万用表 distance=0.46906429529190063
      内容: ## 交流电压测量


警告：谨防触电。

若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示...
  [5] chunk_id=467121384825486677 item_name=RS PRO RS-12 数字万用表 distance=0.46114856004714966
      内容: ## 电阻测量


警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。

1. 将功能转盘置于最高电阻Ω位置.

2. 将黑色表笔插入负极...
------------------------------------------------------------
测试完成
Disconnected from server
"""

"""
问题1：
    ERROR:query.hyde_search_node:LLM调用失败：Error code: 400 - {'error': {'message': "<400> InternalError.Algo.InvalidParameter: 'messages' must contain the word 'json' in some form, to use 'response_format' of type 'json_object'.", 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_parameter_error'}, 'id': 'chatcmpl-e7e131ef-4c45-9f2f-b8b8-455bca598560', 'request_id': 'e7e131ef-4c45-9f2f-b8b8-455bca598560'}
        # 1.获取LLM客户端
        llm_client = AIClients.get_llm_openai(False)
问题2：
ERROR:query.hyde_search_node:LLM调用失败：BaseChatModel.invoke() missing 1 required positional argument: 'input'
invoke() 方法的第一个参数名是 input，不是 messages。代码中使用了 messages= 作为关键字参数，导致 input 参数缺失。
# ❌ 错误：invoke 没有 messages 参数
llm_response = llm_client.invoke(messages=[...])

# ✅ 正确：直接传位置参数
llm_response = llm_client.invoke([...])
"""