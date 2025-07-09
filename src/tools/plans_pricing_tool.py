"""
产品推荐工具
用于处理C类意图（推荐课程），基于数据库中的产品信息提供个性化推荐
"""
import asyncio
import logging
import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# 产品推荐的System Prompt
PRODUCT_RECOMMENDATION_PROMPT = """You are a professional course and product recommendation specialist at HKCA Learning Media.

Your task is to analyze the user's query and recommend the most suitable products from our available offerings.

Based on the user's needs, provide personalized recommendations considering:
1. User's background and experience level
2. Specific subjects or areas of interest mentioned
3. Budget considerations if mentioned
4. Learning goals and timeline

For each recommendation:
- Explain why this product suits their needs
- Highlight key benefits
- Mention pricing in both CNY and HKD
- Suggest a learning path if multiple products are relevant

Be professional, helpful, and focus on matching products to user needs rather than just listing all available options.

Available Products:
{products_info}

User Query: {user_query}

Please provide a comprehensive recommendation in {language}."""

class ProductRecommendationInput(BaseModel):
    """产品推荐输入模型"""
    query: str = Field(..., description="用户的课程推荐查询")
    lang: str = Field(default="Eng", description="用户期望的回复语言")

class PlansAndPricingTool:
    """产品推荐和定价工具"""
    
    def __init__(self):
        """初始化工具，设置数据库连接和LLM"""
        # 初始化数据库连接
        try:
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                logger.error("DATABASE_URL环境变量未设置")
                raise ValueError("DATABASE_URL环境变量未设置")
            
            self.engine = create_engine(db_url)
            logger.info("数据库连接初始化成功")
        except Exception as e:
            logger.error(f"数据库连接初始化失败: {e}")
            raise
        
        # 初始化异步LLM客户端
        from src.utils.llm_client import llm_client
        self.llm_client = llm_client
        logger.info("产品推荐工具LLM客户端初始化完成")
    
    async def get_products_from_db(self, user_language: str = None) -> List[Dict]:
        """从数据库获取产品信息，根据用户语言过滤（异步版本）"""
        try:
            # 根据用户语言构建查询条件
            if user_language:
                # 查询指定语言的产品
                query = text("""
                    SELECT name, price_cny, price_hkd, introduction, type, ai_balance, language
                    FROM products 
                    WHERE is_deleted = false AND language = :language
                    ORDER BY type, name
                """)
                params = {"language": user_language}
                logger.info(f"异步查询语言为 {user_language} 的产品")
            else:
                # 查询所有产品
                query = text("""
                    SELECT name, price_cny, price_hkd, introduction, type, ai_balance, language
                    FROM products 
                    WHERE is_deleted = false
                    ORDER BY type, name
                """)
                params = {}
                logger.info("异步查询所有产品")
            
            # 使用线程池执行同步数据库操作
            loop = asyncio.get_event_loop()
            
            def _execute_query():
                with self.engine.connect() as conn:
                    if user_language:
                        return conn.execute(query, params).fetchall()
                    else:
                        return conn.execute(query).fetchall()
            
            # 在线程池中执行数据库查询
            results = await loop.run_in_executor(None, _execute_query)
                
            products = []
            for row in results:
                product = {
                    "name": row[0],
                    "price_cny": float(row[1]) if row[1] else 0.0,
                    "price_hkd": float(row[2]) if row[2] else 0.0,
                    "introduction": row[3] or "",
                    "type": row[4] or "",
                    "ai_balance": row[5] if row[5] else 0,
                    "language": row[6] or ""
                }
                products.append(product)
            
            logger.info(f"异步从数据库获取到{len(products)}个产品")
            return products
            
        except Exception as e:
            logger.error(f"异步获取产品信息失败: {e}")
            return []
    
    def get_products_from_db_sync(self, user_language: str = None) -> List[Dict]:
        """从数据库获取产品信息，根据用户语言过滤（同步版本，向后兼容）"""
        try:
            # 根据用户语言构建查询条件
            if user_language:
                # 查询指定语言的产品
                query = text("""
                    SELECT name, price_cny, price_hkd, introduction, type, ai_balance, language
                    FROM products 
                    WHERE is_deleted = false AND language = :language
                    ORDER BY type, name
                """)
                params = {"language": user_language}
                logger.info(f"同步查询语言为 {user_language} 的产品")
            else:
                # 查询所有产品
                query = text("""
                    SELECT name, price_cny, price_hkd, introduction, type, ai_balance, language
                    FROM products 
                    WHERE is_deleted = false
                    ORDER BY type, name
                """)
                params = {}
                logger.info("同步查询所有产品")
            
            with self.engine.connect() as conn:
                if user_language:
                    results = conn.execute(query, params).fetchall()
                else:
                    results = conn.execute(query).fetchall()
                
            products = []
            for row in results:
                product = {
                    "name": row[0],
                    "price_cny": float(row[1]) if row[1] else 0.0,
                    "price_hkd": float(row[2]) if row[2] else 0.0,
                    "introduction": row[3] or "",
                    "type": row[4] or "",
                    "ai_balance": row[5] if row[5] else 0,
                    "language": row[6] or ""
                }
                products.append(product)
            
            logger.info(f"同步从数据库获取到{len(products)}个产品")
            return products
            
        except Exception as e:
            logger.error(f"同步获取产品信息失败: {e}")
            return []
    
    def convert_lang_format(self, lang: str) -> str:
        """将意图识别的语言格式转换为数据库语言格式"""
        lang_mapping = {
            "zh-cn": "zh-cn",  # 简体中文
            "zh-hk": "zh-hk",  # 繁体中文（香港）
            "en": "en",        # 英文
            # 兼容旧格式
            "Sim": "zh-cn",
            "Trad": "zh-hk", 
            "Eng": "en"
        }
        return lang_mapping.get(lang, "en")  # 默认英文
    
    def format_products_info(self, products: List[Dict], user_lang: str) -> str:
        """格式化产品信息为文本"""
        if not products:
            return "暂无可用产品信息。"
        
        formatted_products = []
        
        # 根据用户语言调整格式
        if user_lang in ["zh-cn", "Sim"]:  # 简体中文
            header = "可用产品列表："
            for i, product in enumerate(products, 1):
                product_text = f"""
{i}. 产品名称：{product['name']}
   产品类型：{product['type']}
   价格：¥{product['price_cny']:.2f} / HK${product['price_hkd']:.2f}
   产品介绍：{product['introduction']}"""
                
                if product['ai_balance'] > 0:
                    product_text += f"\n   AI回答次数：{product['ai_balance']}"
                
                if product['language']:
                    product_text += f"\n   支持语言：{product['language']}"
                    
                formatted_products.append(product_text)
                
        elif user_lang in ["zh-hk", "Trad"]:  # 繁体中文
            header = "可用產品列表："
            for i, product in enumerate(products, 1):
                product_text = f"""
{i}. 產品名稱：{product['name']}
   產品類型：{product['type']}
   價格：¥{product['price_cny']:.2f} / HK${product['price_hkd']:.2f}
   產品介紹：{product['introduction']}"""
                
                if product['ai_balance'] > 0:
                    product_text += f"\n   AI回答次數：{product['ai_balance']}"
                
                if product['language']:
                    product_text += f"\n   支援語言：{product['language']}"
                    
                formatted_products.append(product_text)
                
        else:  # 英文
            header = "Available Products:"
            for i, product in enumerate(products, 1):
                product_text = f"""
{i}. Product Name: {product['name']}
   Product Type: {product['type']}
   Price: ¥{product['price_cny']:.2f} / HK${product['price_hkd']:.2f}
   Description: {product['introduction']}"""
                
                if product['ai_balance'] > 0:
                    product_text += f"\n   AI Responses Included: {product['ai_balance']}"
                
                if product['language']:
                    product_text += f"\n   Supported Language: {product['language']}"
                    
                formatted_products.append(product_text)
        
        return header + "\n" + "\n".join(formatted_products)
    
    def get_language_name(self, lang: str) -> str:
        """获取语言的完整名称"""
        lang_mapping = {
            "zh-cn": "Simplified Chinese",
            "zh-hk": "Traditional Chinese", 
            "en": "English",
            # 兼容旧格式
            "Sim": "Simplified Chinese",
            "Trad": "Traditional Chinese", 
            "Eng": "English"
        }
        return lang_mapping.get(lang, "English")
    
    async def recommend_products(self, query: str, lang: str = "en") -> Dict[str, Any]:
        """
        基于用户查询推荐产品（异步版本，支持token和成本统计）
        
        Args:
            query: 用户的课程推荐查询
            lang: 用户期望的回复语言 (zh-cn, zh-hk, en)
            
        Returns:
            Dict: 包含产品推荐内容和使用统计的字典
        """
        try:
            # 转换语言格式
            db_lang = self.convert_lang_format(lang)
            logger.info(f"用户语言: {lang}, 数据库查询语言: {db_lang}")
            
            # 获取指定语言的产品信息（异步）
            products = await self.get_products_from_db(user_language=db_lang)
            if not products:
                error_msg = ""
                if lang in ["zh-cn", "Sim"]:
                    error_msg = "抱歉，暂时无法获取产品信息，请稍后再试。"
                elif lang in ["zh-hk", "Trad"]:
                    error_msg = "抱歉，暫時無法獲取產品信息，請稍後再試。"
                else:
                    error_msg = "Sorry, unable to retrieve product information at the moment. Please try again later."
                
                return {
                    "response": error_msg,
                    "usage_info": {
                        "request_id": "error",
                        "model_used": "database_query",
                        "provider": "local",
                        "total_cost": 0.0,
                        "currency": "CNY",
                        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                        "cost_breakdown": {"input_cost": 0, "output_cost": 0, "total_cost": 0},
                        "processing_time": 0.0
                    }
                }
            
            # 格式化产品信息
            products_info = self.format_products_info(products, lang)
            language_name = self.get_language_name(lang)
            
            # 构建prompt
            full_prompt = PRODUCT_RECOMMENDATION_PROMPT.format(
                products_info=products_info,
                user_query=query,
                language=language_name
            )
            
            user_message = f"Please provide product recommendations for: {query}"
            
            # 调用异步LLM生成推荐，使用qwen-max-latest模型
            response_content, usage_info = await self.llm_client.simple_chat(
                user_message=user_message,
                system_message=full_prompt,
                model_name="qwen-max-latest"
            )
            
            logger.info("产品推荐生成成功")
            
            # 构建包含使用统计的完整结果
            result = {
                "response": response_content,
                "usage_info": {
                    "request_id": usage_info.request_id,
                    "model_used": usage_info.model_used,
                    "provider": usage_info.provider,
                    "total_cost": usage_info.total_cost,
                    "currency": usage_info.currency,
                    "token_usage": usage_info.token_usage,
                    "cost_breakdown": usage_info.cost_breakdown,
                    "processing_time": usage_info.processing_time
                }
            }
            
            logger.info(f"产品推荐完成 - 模型: {usage_info.model_used}, "
                       f"Token: {usage_info.token_usage.get('total_tokens', 0)}, "
                       f"成本: {usage_info.total_cost}{usage_info.currency}")
            
            return result
                
        except Exception as e:
            logger.error(f"产品推荐生成失败: {e}")
            # 根据语言返回错误信息
            error_msg = ""
            if lang in ["zh-cn", "Sim"]:
                error_msg = "抱歉，产品推荐服务暂时不可用，请稍后再试。如有紧急需求，请联系我们的客服团队。"
            elif lang in ["zh-hk", "Trad"]:
                error_msg = "抱歉，產品推薦服務暫時不可用，請稍後再試。如有緊急需求，請聯繫我們的客服團隊。"
            else:
                error_msg = "Sorry, the product recommendation service is temporarily unavailable. Please try again later or contact our customer service team for urgent inquiries."
            
            return {
                "response": error_msg,
                "usage_info": {
                    "request_id": "error",
                    "model_used": "fallback",
                    "provider": "local",
                    "total_cost": 0.0,
                    "currency": "CNY",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "cost_breakdown": {"input_cost": 0, "output_cost": 0, "total_cost": 0},
                    "processing_time": 0.0
                }
            }
    
    def recommend_products_sync(self, query: str, lang: str = "en") -> str:
        """
        基于用户查询推荐产品（同步版本，向后兼容），使用同步HTTP客户端
        
        Args:
            query: 用户的课程推荐查询
            lang: 用户期望的回复语言 (zh-cn, zh-hk, en)
            
        Returns:
            str: 产品推荐内容
        """
        try:
            # 使用同步LLM客户端
            from src.utils.sync_llm_client import sync_llm_client
            
            # 转换语言格式
            db_lang = self.convert_lang_format(lang)
            logger.info(f"用户语言: {lang}, 数据库查询语言: {db_lang}")
            
            # 获取指定语言的产品信息（同步）
            products = self.get_products_from_db_sync(user_language=db_lang)
            if not products:
                return self._get_fallback_recommendation(lang)
            
            # 格式化产品信息
            products_info = self.format_products_info(products, lang)
            language_name = self.get_language_name(lang)
            
            # 构建prompt
            full_prompt = PRODUCT_RECOMMENDATION_PROMPT.format(
                products_info=products_info,
                user_query=query,
                language=language_name
            )
            
            user_message = f"Please provide product recommendations for: {query}"
            
            # 调用同步LLM生成推荐，使用qwen-max-latest模型
            response_content, usage_info = sync_llm_client.simple_chat(
                user_message=user_message,
                system_message=full_prompt,
                model_name="qwen-max-latest"
            )
            
            logger.info("同步产品推荐生成成功")
            logger.info(f"同步产品推荐完成 - 模型: {usage_info.model_used}, "
                       f"Token: {usage_info.token_usage.get('total_tokens', 0)}, "
                       f"成本: {usage_info.total_cost}{usage_info.currency}")
            
            return response_content
                
        except Exception as e:
            logger.error(f"同步产品推荐生成失败: {e}")
            return self._get_fallback_recommendation(lang)
    
    def _get_fallback_recommendation(self, lang: str) -> str:
        """获取默认产品推荐"""
        if lang in ["zh-cn", "Sim"]:
            return "抱歉，产品推荐服务暂时不可用，请稍后再试。如有紧急需求，请联系我们的客服团队获取详细的课程和定价信息。"
        elif lang in ["zh-hk", "Trad"]:
            return "抱歉，產品推薦服務暫時不可用，請稍後再試。如有緊急需求，請聯繫我們的客服團隊獲取詳細的課程和定價信息。"
        else:
            return "Sorry, the product recommendation service is temporarily unavailable. Please try again later or contact our customer service team for detailed course and pricing information."

# 创建工具实例
plans_pricing_tool_instance = PlansAndPricingTool()

# 封装为StructuredTool（同步版本，向后兼容）
plans_pricing_tool = StructuredTool.from_function(
    func=plans_pricing_tool_instance.recommend_products_sync,
    name="plans_pricing",
    description="基于用户需求从数据库产品信息中提供个性化的课程和产品推荐，包括价格信息。",
    args_schema=ProductRecommendationInput
) 