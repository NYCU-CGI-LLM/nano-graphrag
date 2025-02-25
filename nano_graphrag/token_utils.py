import tiktoken
import re
from typing import Dict, List, Union, Optional, Any

class TokenCounter:
    """用於計算和追踪 RAG 系統中各部分 Token 使用情況的工具類"""

    def __init__(self, model_name: str = "gpt-4o"):
        """初始化 TokenCounter

        Args:
            model_name: 用於 tiktoken 編碼的模型名稱
        """
        self.model_name = model_name
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.stats = {
            "query_tokens": 0,
            "system_prompt_tokens": 0,
            "retrieved_data_tokens": 0,
            "total_prompt_tokens": 0,
            "completion_tokens": 0
        }

    def count_tokens(self, text: str) -> int:
        """計算文本的 token 數量

        Args:
            text: 要計算的文本

        Returns:
            token 數量
        """
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """批量計算多個文本的 token 數量

        Args:
            texts: 要計算的文本列表

        Returns:
            每個文本的 token 數量列表
        """
        return [self.count_tokens(text) for text in texts]

    def update_query_tokens(self, query: str) -> None:
        """更新查詢的 token 數量

        Args:
            query: 用戶查詢
        """
        self.stats["query_tokens"] = self.count_tokens(query)

    def update_system_prompt_tokens(self, system_prompt: str) -> None:
        """更新系統提示的 token 數量

        Args:
            system_prompt: 系統提示
        """
        self.stats["system_prompt_tokens"] = self.count_tokens(system_prompt)

    def update_retrieved_data_tokens(self, chunks: List[str]) -> None:
        """更新檢索數據的 token 數量

        Args:
            chunks: 檢索到的文本塊列表
        """
        self.stats["retrieved_data_tokens"] = sum(self.count_tokens_batch(chunks))

    def update_total_prompt_tokens(self, full_prompt: str) -> None:
        """更新完整提示的 token 數量

        Args:
            full_prompt: 完整的提示（包括系統提示、上下文和查詢）
        """
        self.stats["total_prompt_tokens"] = self.count_tokens(full_prompt)

    def update_completion_tokens(self, completion: str) -> None:
        """更新回應的 token 數量

        Args:
            completion: LLM 生成的回應
        """
        self.stats["completion_tokens"] = self.count_tokens(completion)

    def get_stats(self) -> Dict[str, int]:
        """獲取所有 token 統計信息

        Returns:
            包含各部分 token 數量的字典
        """
        return self.stats.copy()

    def reset_stats(self) -> None:
        """重置所有統計信息"""
        for key in self.stats:
            self.stats[key] = 0

# 提供一個簡單的估算函數，用於快速估算（不需要精確計算時使用）
def estimate_tokens(text: str) -> int:
    """估算文本的 token 數量

    中文字符: 約 1.5 tokens/字符
    非中文字符: 約 0.25 tokens/字符

    Args:
        text: 要估算的文本

    Returns:
        估算的 token 數量
    """
    if not text:
        return 0

    # 使用正則表達式匹配中文和非中文字符
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    non_chinese_chars = len(re.findall(r"[^\u4e00-\u9fff]", text))

    # 計算估算的 token 數量
    tokens = chinese_chars * 1.5 + non_chinese_chars * 0.25

    return int(tokens)