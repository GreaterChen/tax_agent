"""
产品推荐工具
用于处理C类意图（推荐课程），基于数据库中的产品信息提供个性化推荐
"""
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
        
        # 初始化LLM
        try:
            from config.llm_config import llm_config
            
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="qwen-max-latest", 
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0.3
            )
                
        except Exception as e:
            logger.error(f"LLM初始化失败: {e}")
            # 回退到默认配置
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="gpt-4o-mini", 
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                temperature=0.3
            )
    
    def get_products_from_db(self, user_language: str = None) -> List[Dict]:
        """从数据库获取产品信息，根据用户语言过滤"""
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
                logger.info(f"查询语言为 {user_language} 的产品")
            else:
                # 查询所有产品
                query = text("""
                    SELECT name, price_cny, price_hkd, introduction, type, ai_balance, language
                    FROM products 
                    WHERE is_deleted = false
                    ORDER BY type, name
                """)
                params = {}
                logger.info("查询所有产品")
            
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
            
            logger.info(f"从数据库获取到{len(products)}个产品")
            return products
            
        except Exception as e:
            logger.error(f"获取产品信息失败: {e}")
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
    
    def recommend_products(self, query: str, lang: str = "en") -> str:
        """
        基于用户查询推荐产品
        
        Args:
            query: 用户的课程推荐查询
            lang: 用户期望的回复语言 (zh-cn, zh-hk, en)
            
        Returns:
            str: 产品推荐内容
        """
        try:
            # 转换语言格式
            db_lang = self.convert_lang_format(lang)
            logger.info(f"用户语言: {lang}, 数据库查询语言: {db_lang}")
            
            # 获取指定语言的产品信息
            products = self.get_products_from_db(user_language=db_lang)
            if not products:
                if lang in ["zh-cn", "Sim"]:
                    return "抱歉，暂时无法获取产品信息，请稍后再试。"
                elif lang in ["zh-hk", "Trad"]:
                    return "抱歉，暫時無法獲取產品信息，請稍後再試。"
                else:
                    return "Sorry, unable to retrieve product information at the moment. Please try again later."
            
            # 格式化产品信息
            products_info = self.format_products_info(products, lang)
            language_name = self.get_language_name(lang)
            
            # 构建prompt
            full_prompt = PRODUCT_RECOMMENDATION_PROMPT.format(
                products_info=products_info,
                user_query=query,
                language=language_name
            )
            
            # 调用LLM生成推荐
            from langchain_core.messages import SystemMessage, HumanMessage
            
            messages = [
                SystemMessage(content=full_prompt),
                HumanMessage(content=f"Please provide product recommendations for: {query}")
            ]
            
            response = self.llm.invoke(messages)
            
            logger.info("产品推荐生成成功")
            return response.content
                
        except Exception as e:
            logger.error(f"产品推荐生成失败: {e}")
            # 根据语言返回错误信息
            if lang in ["zh-cn", "Sim"]:
                return "抱歉，产品推荐服务暂时不可用，请稍后再试。如有紧急需求，请联系我们的客服团队。"
            elif lang in ["zh-hk", "Trad"]:
                return "抱歉，產品推薦服務暫時不可用，請稍後再試。如有緊急需求，請聯繫我們的客服團隊。"
            else:
                return "Sorry, the product recommendation service is temporarily unavailable. Please try again later or contact our customer service team for urgent inquiries."

# 创建工具实例
plans_pricing_tool_instance = PlansAndPricingTool()

# 封装为StructuredTool
plans_pricing_tool = StructuredTool.from_function(
    func=plans_pricing_tool_instance.recommend_products,
    name="plans_pricing",
    description="基于用户需求从数据库产品信息中提供个性化的课程和产品推荐，包括价格信息。",
    args_schema=ProductRecommendationInput
) 