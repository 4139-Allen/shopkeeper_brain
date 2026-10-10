import asyncio
import json
from json import JSONDecodeError
from typing import List, Tuple, Union, Dict, Any

from agents.mcp import MCPServerStreamableHttp

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState


class WebSearchMcpNode(BaseNode):
    """MCP网络搜索节点"""

    # 节点名称
    name = "web_search_mcp_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, dict]:
        """执行网络搜索"""
        # 1.参数校验
        validated_rewritten_query, validate_item_names = self._validate_query_inputs(state)

        # 2.执行MCP检索
        mcp_result = asyncio.run(self._create_execute_web_search(validated_rewritten_query))

        if not mcp_result:
            return state

        # 3.返回状态
        return {"web_search_docs": mcp_result}

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """参数校验"""
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        return rewritten_query, item_names

    async def _create_execute_web_search(self, validated_rewritten_query: str) -> List[Dict[str, Any]]:
        """
        异步执行MCP网络检索
        :param validated_rewritten_query:
        :return:  搜索列表   每项包含：  title   url   snippet
        """
        # 1.创建MCP 客户端(Streamable HTTP)模式, 自动管理连接
        async with MCPServerStreamableHttp(
                name="bailian_web_search",
                params={
                    "url": self.config.mcp_dashscope_base_url,
                    "headers": {"Authorization": f"Bearer {self.config.dashscope_api_key}"},
                    "timeout": 300,
                    "terminate_on_close": True
                },
                max_retry_attempts=2,
                cache_tools_list=True
        ) as client:
            # 2.调用工具获取结果
            execute_tool_result = await client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": validated_rewritten_query, "count": 3})

            print(execute_tool_result)

            """
            meta=None content=[TextContent(type='text', text='{"pages":[
                {
                    "snippet":"403 10 巴西青年隔空喊话雷军:小米汽车出海,不止是爆款更是 游资艳姐 06-21 19:09 20 0 玩具汽车 股友16I25710F7 06-17 12:30 17 0 今天有种好像我又行了的感觉,扶我起来 兔饱饱 06-09 13:46 26 0 牛! 钩得稳稳 06-05 17:12 109 0 $绿的谐波(SH688017)$$小米汽车(BK1155)$$稀土永磁(BK 美梦成真666 06-05 15:29 10012 11 小米四年首现业绩双降,汽车与AI布局蓄力成长 每日财报 06-03 00:26 566 0 小米汽车发文:小米汽车5月交付超过30000台!#小米汽 中国经营报 06-01 19:55 84 0 【小米汽车发布世界模型全新框架:重建+生成一体化, 猫头鹰车志 05-26 13:26 166 0 从“生而移动”到“向硬核进化”,小米万亿生态质变 科记汇 05-22 21:11 845 0 23.35万元起!雷军重磅发布YU7 GT、YU7标准版,小米YU 老冀说科技 05-21 22:52 2549 0 YU7 GT 小米人车家全生态新品发布会 小米公司 05-20 18:13 97 0 数央网观察:小米YU7 GT连上热搜,雷军口中的贵” 数央网 05-20 16:58 94 0 星网宇达和小米汽车有哪些合作?明天走势会如何? 大家的好耶耶 05-19 15:28 67 0 小米汽车是否有星网宇达供货? 大家的好耶耶 05-19 10:46 45993 76 小米汽车闯进前五!65万辆交付背后的投资逻辑再审视 港股老斯基 05-19 07:43 44 0 雷军自己把股价打下来了要跌停了 河来财涨停龙2828 05-15 08:05 41072 43 超3.6万辆!小米汽车4月销量杀入国产前五,信心再被点 一点财经 05-13 16:07 698 0 小米汽车北京国际车展发布会 小米公司 04-23 10:46 53 1 先造谣再辟谣,辟谣比造谣还先到,现在品牌传播策略之 坐庄散户 04-22 18:53 47 0 $小米汽车(BK1155)$ 凡凡高涨 04-21 22:08 568 0 雷军直播开SU7跨城挑战 1265公里只充一次电 直言不做 每日财报 04-21 15:01 87 0 【雷军:小米未来几年内都不会做10万元内车型】财联社 首席消费官 04-17 14:55 1455 0 雷军将亲自直播1265公里只冲一次电,新SU7从北京到上 蓝鲸新闻 04-16 14:38 45 0 首相访问小米SU7机会多多,小米汽车进军欧美市场西班 牛气冲天神1 04-14 13:07 127 1 马去了赛力斯,赛力斯火了。郑去了小米汽车,小米汽车 走遍大好河山 04-13 13:41 49 1 你速度60我速度60,等于咱俩速度120。 钩得稳稳 04-13 12:06 49 0 速成鸡 精准的章杭素 04-12 15:51 84 1 财联社4月2日电,乘联分会数据显示,3月特斯拉中国批 牛气冲天神1 04-11 02:34 2294 2 雷军:安全是小米汽车的基础和前提。“小 97 0 数央网观察:小米YU7 GT连上热搜,雷军口中的“小贵” 数央网 0",
                    "hostname":"东方财富网",
                    "hostlogo":"http://gubaf10.eastmoney.com/favicon.ico",
                    "title":"小米汽车股吧_小米汽车分析讨论社区-东方财富网",
                    "url":"http://gubaf10.eastmoney.com/list,bk1155.html"
                },
                {
                    "snippet":"6月30日小米汽车概念上涨0.98%,板块个股德迈仕、泉峰汽车涨幅居前 6月30日,截至收盘,小米汽车概念上涨0.98%,板块资金流出139513.04万。上涨个股家数63个,下跌个股家数27个。 板块涨幅居前的十大牛股分别是:德迈仕(15.55%)、泉峰汽车(10.03%)、天汽模(6.81%)、徕木股份(6.76%)、豪能股份(6.4%)、东方中科(5.07%)、厦门信达(4.9%)、电工合金(4.04%)、贵航股份(4.01%)、会通股份(3.55%)、卡倍亿(3.55%)、中捷精工(3.12%)、江波龙(2.99%)、凌云股份(2.79%)、春秋电子(2.75%)、隆利科技(2.65%)、富特科技(2.49%)、航天智造(2.49%)、四维图新(2.15%)、银轮股份(1.85%) <table><tr><th>序</th><th>代码</th><th>股票名称</th><th>现价</th><th>涨跌幅</th><th>主力资金净额</th><th>主力资金净占比</th></tr><tr><td>1</td><td>301007</td><td>德迈仕</td><td>30.99</td><td>15.55</td><td>4798.97万</td><td>6.34</td></tr><tr><td>2</td><td>603982</td><td>泉峰汽车</td><td>11.08</td><td>10.03</td><td>9749.54万</td><td>37.08</td></tr><tr><td>3</td><td>002510</td><td>天汽模</td><td>7.06</td><td>6.81</td><td>1.01亿</td><td>8.16</td></tr><tr><td>4</td><td>603633</td><td>徕木股份</td><td>8.69</td><td>6.76</td><td>569.01万</td><td>1.90</td></tr><tr><td>5</td><td>603809</td><td>豪能股份</td><td>15.3</td><td>6.4</td><td>6341.70万</td><td>9.81</td></tr><tr><td>6</td><td>002819</td><td>东方中科</td><td>31.3</td><td>5.07</td><td>3411.29万</td><td>4.39</td></tr><tr><td>7</td><td>000701</td><td>厦门信达</td><td>6.64</td><td>4.9</td><td>4175.47万</td><td>6.88</td></tr><tr><td>8</td><td>300697</td><td>电工合金</td><td>14.43</td><td>4.04</td><td>-2139.68万</td><td>-1.78</td></tr><tr><td>9</td><td>600523</td><td>贵航股份</td><td>14.79</td><td>4.01</td><td>-2356.58万</td><td>-5.78</td></tr><tr><td>10</td><td>688219</td><td>会通股份</td><td>12.55</td><td>3.55</td><td>-1320.72万</td><td>-6.79</td></tr><tr><td>11</td><td>300863</td><td>卡倍亿</td><td>38.54</td><td>3.55</td><td>932.79万</td><td>3.64</td></tr><tr><td>12</td><td>301072</td><td>中捷精工</td><td>21.47</td><td>3.12</td><td>-563.82万</td><td>-5.91</td></tr><tr><td>13</td><td>301308</td><td>江波龙</td><",
                    "hostname":"金融界",
                    "hostlogo":"https://i0.jrjimg.cn/ad/pic300.jpg",
                    "title":"6月30日小米汽车概念上涨0.98%,板块个股德迈仕、泉峰汽车涨幅居前",
                    "url":"https://m.jrj.com.cn/madapter/finance/2025/06/30183951390815.shtml"
                },
                {"snippet":"2026/06/29/一16:00 价 21.860 均 21.949 量 875.00万 幅 2.05% ▲▼ 成交 VOL: 8750000.00 MA10: 1676640.00 ▲▼ 量比 LB: 0.01 ▲▼ 无 量比 MACD BOLL RSI BBIBOLL ROC 均价/波幅 价格(元) 成交(手) <table><tr><th>12.940</th><th>2.6万手</th></tr><tr><td>12.940</td><td>2.6万手</td></tr><tr><td>12.940</td><td>2.6万手</td></tr><tr><td>12.940</td><td>2.6万手</td></tr><tr><td>12.940</td><td>2.6万手</td></tr><tr><td>12.940</td><td>2.6万手</td></tr></table> 大手成交分布 市价上 79.5% 市价下","hostname":"新浪网","hostlogo":"https://n.sinaimg.cn/finance/hqcenter_sharepic/hqcenter_171x171.jpg","title":"小米集团-W","url":"https://stock.finance.sina.com.cn/hkstock/quotes/1810.html"}],"request_id":"5f654015-ba32-97cc-b809-cacaba41ebcb","status":0}', annotations=None, meta=None)] structuredContent=None isError=False
            """

            # 3.解析结果

            # 3.1 获取最外层的对象
            if not execute_tool_result:
                return []

            # 3.2 获取对象的content属性
            if not execute_tool_result.content[0]:
                return []

            # 3.3 获取TextContent对象的text
            content_text = execute_tool_result.content[0].text
            if not content_text:
                return []
            try:
                # 3.4 反序列化
                content_text_json = json.loads(content_text)
                # 1) 获取pages
                pages = content_text_json.get("pages")
                if not pages:
                    return []

                # 2) 遍历每一个结果
                search_result = []
                for page in pages:
                    title = page.get("title", "").strip()
                    snippet = page.get("snippet", "").strip()
                    url = page.get("url", "").strip()
                    search_result.append({"title": title, "url": url, "snippet": snippet})

                # 4.返回结果
                return search_result
            except JSONDecodeError as e:
                self.logger.error(f"反序列化MCP结果失败：{str(e.msg)} - 原文 {e.doc} - 位置 {e.pos}")
                return []


if __name__ == '__main__':

    state1 = {
        "rewritten_query": "今天的小米汽车的股价是多少",
        "item_names": ["RS-12 数字万用表"]
    }

    web_mcp_search = WebSearchMcpNode()
    result = web_mcp_search.process(state1)

    for r in result.get('web_search_docs', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))
