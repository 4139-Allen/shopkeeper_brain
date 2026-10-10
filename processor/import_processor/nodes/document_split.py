import re
import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    """
    主要职责：
    上一节点传出的 md_content + file_title
            ↓
    参数校验,统一换行符(\r\n → \n)
            ↓
    按 h1-h6 标题切分为 section(记录 title/parent_title)
            ↓
    长 section 递归切分 / 短 section 贪心合并(保证语义完整)
            ↓
    组装 chunks(title + content) → state['chunks'] 更新
            ↓
    备份 chunks.json → 交给下游节点(商品名识别)
    """
    name = "document_split_node"

    def process(self, state:ImportGraphState):

        # 1.校验参数
        md_content, file_title, max_content_length, min_content_length = self._validate_state(state)

        # 2.根据标题切割，得到段落
        sections = self._split_by_heading(md_content, file_title)

        # 3.切分与合并
        final_chunks = self._split_and_merge(sections, max_content_length, min_content_length)

        # 4.组装
        chunks = self._assemble_chunks(final_chunks)

        # 5.更新状态
        state["chunks"] = chunks

        # 6.备份
        self._backup_chunks(state, chunks)

        return state

    # 校验参数，顺带统一文本换行符
    def _validate_state(self, state:ImportGraphState):
        """
        校验参数，换行符统一
        :param state:
        :return:
        """
        # 1.打印日志
        self.log_step(f"step1","参数校验以及获取文档内容")
        # 2.获取文档内容
        md_content = state.get("md_content")
        # 3.统一换行符
        # md 文件里的换行符按来源不同有三种写法：\r\n、\r、\n
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        # 4.获取文件标题
        file_title = state.get("file_title")
        # 5.校验最大最小值 不能小于0，而且min < max   ValueError
        if (self.config.min_content_length < 0
            or self.config.max_content_length < 0
            or self.config.min_content_length > self.config.max_content_length):
            raise ValueError("切片长度校验失败，请检查参数配置")

        # 6.返回数据
        return md_content, file_title, self.config.max_content_length, self.config.min_content_length

    # 通过标题切块
    def _split_by_heading(self, md_content, file_title):
        """
        根据 h1 - h6 标题切分
        :param md_content:
        :param file_title:
        :return:
            sections: 列表，每个元素格式
            {
                "title": "# 第一章",
                "body": "正文内容...",
                "file_title": "万用表",
                "parent_title": "# 第一章"
            }
        """
        self.log_step(f"step2", "按标题开始切分文档")

        # 1.定义变量
        in_fence = False        # 是否在代码围栏中（'''或~~~）
        hierarchy = [""] * 7    # 7个长度列表，第一个为空（当前正处在第 i 级标题的章节内）
        body_line = []
        sections = []
        current_title = ""
        current_level = 0

        # 2.定义正则表达式
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # 3.按行切分
        content_lines: list[str] = md_content.split("\n")

        def _flush():
            """封装section对象"""
            body = "\n".join(body_line)
            if current_title or body:
                parent_title = ""
                # 获取父级标题
                for p in range(current_level - 1, 0, -1):
                    if hierarchy[p]:
                        parent_title = hierarchy[p]
                        break
                if not parent_title:
                    parent_title = current_title if current_title else file_title
                sections.append({
                    "title": current_title if current_title else file_title,
                    "body": body,
                    "file_title": file_title,
                    "parent_title": parent_title
                })

        for content_line in content_lines:
            # 判断是否存在代码围栏 startswith() endswith() 用来判断字符串的开头/结尾,返回布尔值(True/False):
            if content_line.strip().startswith("```") or content_line.strip().endswith("~~~"):
                in_fence = not in_fence     # 反转bool值

            # 匹配标题(过滤了代码块中的#，防止认作标题处理)
            match = heading_re.match(content_line) if not in_fence else None
            if match:
                _flush()
                level = len(match.group(1))
                current_level = level
                current_title = content_line
                hierarchy[level] = current_title

                # 清空下级标题(清除知道自己所在章节)
                for i in range(level + 1, 7):
                    hierarchy[i] = ""
                # 标题不属于正文
                body_line = []
            else:
                # 处理标题以外，都收集
                body_line.append(content_line)
        _flush()

        return sections

    def _split_and_merge(self, sections, max_content_length, min_content_length):
        """
        超长的 section 先扣除标题占用的长度,再对正文按"
        段落 → 换行 → 句子标点 → 空格"的优先级递归切成不超过上限的多块,
        每块冠以"标题 - 序号";若切不动则原样保留。
        """
        self.log_step("step3","正在切分较长段落")
        # 1.先切长的（chunk 太长(超过 max)切掉）
        current_sections = []
        for section in sections:
            # 使用extend将切分结果合并到current_sections列表
            # extend 的作用:把一个可迭代对象(如列表)里的元素逐个追加到当前列表末尾,相当于"列表的合并扩展"。
            current_sections.extend(self.split_long_section(section, max_content_length))

        # 2.在合并短的（chunk 太短(低于 min)语义不全，合并）
        final_sections = self.merge_short_section(current_sections, min_content_length)
        return final_sections

    def split_long_section(self, section, max_content_length):
        # 1.获取section的属性
        title = section.get("title")
        body = section.get("body")
        file_title = section.get("file_title")
        parent_title = section.get("parent_title")

        # 2.表格降维处理
        if '<table>' in body:
            self.logger.info("***** 表格降维处理 *****")
            body = MarkdownTableLinearizer.process(body)
            section['body'] = body
            self.logger.info(f"{body}")

        # 3.标题过长截断
        if len(title) > 50:
            title = title[:50]

        # 4.计算总长度
        title_prefix = f"{title}\n\n"
        total_length = len(title_prefix) + len(body)

        # 5.不超过阈值，无需切分
        if total_length <= max_content_length:
            return [section]

        # 6.计算body可用空间
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]

        # 7.递归切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,  # 每块最大长度
            chunk_overlap=0,  # 块之间无重叠
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],  # 按优先级尝试分隔符（段落→句子→标点→空格）
            keep_separator=False  # 不保留分隔符
        )
        texts = text_splitter.split_text(body)  # 得到列表
        if len(texts) == 1:
            return [section]

        # 8.生成子section
        sub_sections = []
        for index, text in enumerate(texts):  # 返回(index, text)元组
            sub_sections.append({
                "title": f"{title} - {index + 1}",  # 标题带序号
                "body": text,
                "file_title": file_title,
                "parent_title": parent_title
            })
        return sub_sections

    def merge_short_section(self, current_sections, min_content_length):
        """贪心累加算法：将同源的短section合并
            同一父标题下,当前块只要还没到 min 就不断吞并后续块,一旦够长或遇到跨章节的块就封箱,短块落单也照单全收。
        """
        self.log_step("step4", "正在合并短文本")
        # 1.初始化
        current_section = current_sections[0]
        final_sections = []

        # 2.遍历集合
        for next_section in current_sections[1:]:
            same_parent = current_section['parent_title'] == next_section['parent_title']
            if same_parent and len(current_section.get("body").rstrip()) < min_content_length:
                current_section['body'] = current_section.get("body").rstrip() + "\n\n" + next_section.get(
                    "body").lstrip()
                current_section['title'] = current_section['parent_title']
            else:
                final_sections.append(current_section)
                current_section = next_section

        # 3.最后一个封箱
        final_sections.append(current_section)

        return final_sections

    def _assemble_chunks(self, final_chunks):
        self.log_step("step5", "组合标题和内容")
        """
        最终组合 chunks
        :param final_chunks:
        :return:
        """
        chunks = []

        for chunk in final_chunks:
            chunks.append({
                "title": chunk.get("title"),
                "file_title": chunk.get("file_title"),
                "parent_title": chunk.get("parent_title"),
                "content": f"{chunk.get('title')}\n\n{chunk.get('body')}"
            })
        return chunks

    def _backup_chunks(self, state, chunks):
        self.log_step("step6", "备份")
        """
        将切分结果备份到json文件
        :param state:
        :param chunks:
        :return:
        """
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return

        os.makedirs(local_dir, exist_ok=True)  # exist_ok=True：如果目录已存在，不会抛出异常，直接跳过
        output_path = os.path.join(local_dir, "chunks1_split.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"备份失败：{e}")

if __name__ == "__main__":
    setup_logging()

    node = DocumentSplitNode()

    file_path = r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\华为擎云 L420x 用户指南-(华为擎云 L420x-Axxx,UOS&KOS_01,zh-cn)\auto\华为擎云 L420x 用户指南-(华为擎云 L420x-Axxx,UOS&KOS_01,zh-cn).md"

    with open(file_path, "r", encoding="utf-8")as f:
        md_content_1 = f.read()

    doc_state:ImportGraphState = {
        "file_dir":r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\华为擎云 L420x 用户指南-(华为擎云 L420x-Axxx,UOS&KOS_01,zh-cn)\tem_doc",
        "file_title":"华为擎云 L420x 用户指南-(华为擎云 L420x-Axxx,UOS&KOS_01,zh-cn)",
        "md_content":md_content_1
    }

    node.process(doc_state)

