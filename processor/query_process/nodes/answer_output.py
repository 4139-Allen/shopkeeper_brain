from typing import List, Dict, Any, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import set_task_result


class AnswerOutputNode(BaseNode):
    """答案输出节点
        流程：
            检查已有答案 -> 构建提示词 -> LLM生成答案  -> 写入历史 ->  输出答案
    """

    # 节点名称
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        执行答案生成
        :param state: 查询状态：包含  session_id、task_id、reranked_docs
        :return: 更新后的state，包含： answer 和 prompt
        """

        task_id = state.get("task_id", "")
        is_stream = state.get("is_stream")

        # 1.已有答案直接返回
        if state.get("answer"):
            self._push_existing_answer(state)
        else:
            # 2.没有答案，构建提示词，调用LLM生成答案
            prompt = self._build_prompt(state)
            state['prompt'] = prompt
            self._generate_answer(state, prompt)

        # 3.写入历史记录
        self._write_history(state)

        # 4.流式模式发送结束事件
        if is_stream:
            push_sse_event(task_id, SSEEvent.FINAL, {"answer": state.get("answer", "")})

        # 5.返回结果
        return state

    # ================================================================== #
    #                    已有答案推送                                       #
    # ================================================================== #
    @staticmethod
    def _push_existing_answer(state: QueryGraphState):
        """非流式模式：存入任务结果；"""
        set_task_result(state['task_id'], "answer", state['answer'])

    # ================================================================== #
    #                    历史记录                                           #
    # ================================================================== #
    def _write_history(self, state: QueryGraphState):
        """将用户问题和组手回答写入 MongoDB 历史记录"""
        session_id = state.get("session_id", "")
        rewritten_query = state.get("rewritten_query", "") or state.get("original_query", "")
        item_names = state.get("item_names") or []
        try:
            # 1.写用户问题
            save_chat_message(session_id=session_id,
                              role="user",
                              text=state["original_query"],
                              rewritten_query=rewritten_query,
                              item_names=item_names)

            # 2.写组手回复
            if state.get("answer"):
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=state["answer"],
                    rewritten_query=rewritten_query,
                    item_names=item_names
                )
        except Exception as e:
            self.logger.warning(f"保存历史记录出错：{e}")

    # ================================================================== #
    #                    LLM 生成                                          #
    # ================================================================== #
    def _generate_answer(self, state: QueryGraphState, prompt: str):
        """调用LLM生成答案(流式、非流式)"""
        self.log_step("generating answer", "生成答案")
        llm_client = AIClients.get_llm_openai(False)
        if llm_client is None:
            raise ValueError("LLM客户端初始化失败")
        task_id = state.get("task_id", "")

        if state.get("is_stream"):
            state["answer"] = self._stream_generate(llm_client, prompt, task_id)
        else:
            state["answer"] = self._invoke_generate(llm_client, prompt)
            set_task_result(task_id, "answer", state["answer"])

    def _stream_generate(self, llm_client, prompt, task_id):
        """ 流式输出"""
        accumulate_answer = ""
        try:
            for chunk in llm_client.stream(prompt):
                delta_text = getattr(chunk, "content", "") or ""
                if delta_text:
                    accumulate_answer += delta_text
                    push_sse_event(task_id, SSEEvent.DELTA, {"delta": delta_text})
        except Exception as e:
            self.logger.error(f"流式生成答案出错! {e}")
        return accumulate_answer

    def _invoke_generate(self, llm_client, prompt):
        """ 非流式输出"""
        self.log_step("_invoke_generate", "生成答案")
        try:
            response = llm_client.invoke(prompt)
            return response.content
        except Exception as e:
            self.logger.error(f"生成回答出错:{e}")
            return "抱歉,生成回答时出错!"

    # ================================================================== #
    #                    提示词构建                                         #
    # ================================================================== #
    def _build_prompt(self, state: QueryGraphState):
        """根据搜索结果，历史对话组装调用LLM 提示词"""

        # 获取最大字符数预算
        char_budget = self.config.max_context_chars

        # 1.获取问题和商品名称
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state.get("item_names", [])

        # 2.格式化上下文文档
        context_str, char_budget = self._format_reranked_docs(state.get("reranked_docs"), char_budget)

        # 3.格式化历史对话
        history_str, char_budget = self._format_chat_history(state.get("history", []), char_budget)

        # 4.组装提示词
        return ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str or "无历史记录",
            item_names=",".join(item_names),
            question=question
        )

    @staticmethod
    def _format_reranked_docs(reranked_docs: List[Dict[str, Any]], char_budget: int) -> Tuple[str, int]:
        """格式化重排序文档，带字符预算控制"""

        used_chars = 0
        formatted_lines = []

        for idx, doc in enumerate(reranked_docs, 1):  # 索引从1开始，不是从0开始啦
            content = doc.get("content", "").strip()
            if not content:
                continue
            meta_tags = [f"[{idx}]"]
            for field, template in [
                {"source", "[source={}]"},
                {"chunk_id", "[chunk_id={}]"},
                {"url", "[url={}]"},
                {"title", "[title={}]"}
            ]:

                field_value = str(doc.get(field, "")).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))
            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={relevance_score}:.4f]")
            doc_entry = " ".join(meta_tags) + "\n" + content
            if used_chars + len(doc_entry) > char_budget:
                break
            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2
        return "\n\n".join(formatted_lines), char_budget - used_chars

    @staticmethod
    def _format_chat_history(chat_history: List[Dict], char_budget) -> Tuple[str, int]:
        """格式化历史对话，带字符预算控制"""
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "组手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue
            formatted_line = f"{role_label_map[role]}:{text}"
            if used_chars + len(formatted_line) > char_budget:
                break
            formatted_lines.append(formatted_line)
            used_chars += len(formatted_line) + 1

        return "\n".join(formatted_lines), char_budget - used_chars


if __name__ == "__main__":
    from dotenv import load_dotenv
    import json

    load_dotenv()

    from knowledge.processor.query_process.base import setup_logging
    setup_logging()

    print("=" * 60)
    print("开始测试: 答案生成节点 (AnswerOutputNode)")
    print("=" * 60)

    # 构造模拟状态
    mock_state = {
        "task_id": "test_task_001",
        "session_id": "test_session_001",
        "is_stream": True,
        "original_query": "万用表怎么测电压？",
        "rewritten_query": "RS-12数字万用表如何测量电压？",
        "item_names": ["RS-12数字万用表"],
        "reranked_docs": [
            {
                "content": "数字万用表测量电压步骤：1. 将旋钮转到V档位；2. 黑表笔插COM孔，红表笔插V孔；3. 将表笔并联到被测点两端。",
                "source": "local",
                "chunk_id": "chunk_001",
                "title": "万用表使用手册",
                "score": 0.9234
            },
            {
                "content": "测量直流电压时需注意正负极性，红表笔接正极，黑表笔接负极。",
                "source": "web",
                "url": "https://example.com/guide",
                "title": "电压测量指南",
                "score": 0.8756
            }
        ],
        "history": [
            {"role": "user", "text": "万用表是什么？"},
            {"role": "assistant", "text": "万用表是一种多功能电子测量仪器..."}
        ],
    }

    print("【输入状态】:")
    print(f"  query: {mock_state['rewritten_query']}")
    print(f"  item_names: {mock_state['item_names']}")
    print(f"  reranked_docs: {len(mock_state['reranked_docs'])} 篇")
    print("-" * 60)

    # 执行答案生成
    node = AnswerOutputNode()
    result = node.process(mock_state)

    # 打印结果
    print("\n【生成结果】:")
    print("-" * 60)
    print(result.get("answer", "无答案"))
    print("-" * 60)

    print("\n测试完成")