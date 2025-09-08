import dashscope
import json
import re
import os
import time
import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("并行AI分析服务")

class AIAnalysisService:
    def __init__(self, api_keys: List[str] = None):
        """初始化并行AI分析服务"""
        env_api_key = os.getenv('QWEN_API_KEY')
        
        if api_keys:
            self.api_keys = api_keys
        elif env_api_key:
            self.api_keys = [key.strip() for key in env_api_key.split(',') if key.strip()]
            logger.info(f"使用环境变量中的{len(self.api_keys)}个API密钥")
        else:
            # 不使用默认API密钥，而是提示用户配置环境变量
            self.api_keys = []
            logger.warning("未找到环境变量QWEN_API_KEY，请在.env文件中配置API密钥。")
            logger.warning("请参考.env.example文件创建.env文件并添加您的API密钥。")
            logger.warning("在未配置API密钥的情况下，部分功能可能无法正常使用。")
        
        self.max_retries = 3
        self.retry_delay = 2
        self.save_dir = "history_track"
        os.makedirs(self.save_dir, exist_ok=True)
        
        logger.info(f"并行AI分析服务初始化完成，共{len(self.api_keys)}个API密钥可供使用")

    def extract_json(self, text: str) -> Optional[str]:
        """增强JSON提取逻辑"""
        match = re.search(r'```json\s*([\s\S]+?)\s*```', text)
        if match:
            json_text = match.group(1)
        else:
            match = re.search(r'(\{[\s\S]*\})', text)
            if not match:
                return None
            json_text = match.group(1)

        try:
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError as e:
            logger.debug(f"初步JSON解析失败: {e}，尝试清理...")
            # 移除注释和多余的逗号等常见错误
            json_text = re.sub(r"//.*", "", json_text)
            json_text = re.sub(r",\s*([\}\]])", r"\1", json_text)
            try:
                json.loads(json_text)
                return json_text
            except json.JSONDecodeError as e2:
                logger.warning(f"清理后JSON解析仍然失败: {e2}")
                return None

    def call_qwen_single(self, messages: List[Dict[str, Any]], api_key: str, 
                        model: str = "qwen-turbo", timeout: int = 60) -> str:
        """单个API调用，使用指定密钥"""
        try:
            dashscope.api_key = api_key
            response = dashscope.Generation.call(
                model=model,
                messages=messages,
                result_format='text', 
                temperature=0.3,
                timeout=timeout
            )
            if response.status_code == 200 and response.output and response.output.text:
                return response.output.text
            else:
                error_message = f"API调用异常: 状态码 {response.status_code}, 响应: {response.message}"
                logger.warning(error_message)
                return error_message
        except Exception as e:
            error_message = f"API调用失败: {str(e)}"
            logger.error(error_message)
            return error_message

    def save_analysis_results(self, results: Dict[str, Any], filename: str = "recent_ai_analysis.json"):
        """保存分析结果到JSON文件"""
        filepath = os.path.join(self.save_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"分析结果已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存分析结果失败: {str(e)}")

    def process_single_batch(self, batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个内容批次"""
        batch_blocks = batch_info['blocks']
        batch_index = batch_info['index']
        model = batch_info['model']
        api_key = batch_info['api_key']
        
        system_prompt = (
            "你是一个内容摘要助手，请为以下内容生成客观摘要。严格遵守JSON格式输出。\n"
            "输入输出格式要求：\n"
            "1. 必须按照输入内容的顺序逐个生成摘要，不能调换顺序。\n"
            "2. 输出必须是一个包含'blocks'键的JSON对象，'blocks'的值是一个列表。\n"
            "3. 列表中的每个JSON对象必须包含'id'和'summary'两个键，对应输入的原始内容。\n"
            "4. 输出的block数量必须与输入内容的数量完全一致。\n"
            "摘要要求：\n"
            "1. 对主要内容进行客观概括，不包含个人评价。\n"
            "2. 短内容（少于100字）用一句话概括。\n"
            "3. 长内容（300字以上）的摘要字数约为原文字数的15%,必须逐条分点列清楚。\n"
        )

        batch_content = []
        for block in batch_blocks:
            content = f"ID: {block['id']}\n标题: {block.get('title', '无')}\n内容: {block.get('content', '无')}\n---\n"
            batch_content.append(content)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "".join(batch_content)}
        ]

        for attempt in range(self.max_retries):
            reply = self.call_qwen_single(messages, api_key, model)
            
            if not reply.startswith("API调用"):
                json_text = self.extract_json(reply)
                if json_text:
                    try:
                        result_data = json.loads(json_text)
                        if isinstance(result_data, dict) and "blocks" in result_data and len(result_data["blocks"]) == len(batch_blocks):
                            return {"blocks": result_data["blocks"], "success": True}
                        else:
                            logger.warning(f"批次{batch_index} JSON格式或数量不匹配, 将重试...")
                    except json.JSONDecodeError as e:
                        logger.warning(f"批次{batch_index} JSON解析失败: {e}, 将重试...")
            
            if attempt < self.max_retries - 1:
                logger.info(f"批次{batch_index} 第{attempt + 1}次尝试失败，将在{self.retry_delay}秒后重试")
                time.sleep(self.retry_delay)
        
        logger.error(f"批次{batch_index} 处理失败，已达最大重试次数。")
        error_blocks = [{"id": block['id'], "summary": "错误：AI摘要生成失败"} for block in batch_blocks]
        return {"blocks": error_blocks, "success": False}

    def process_user_sequentially(self, user_task: Dict[str, Any]) -> Dict[str, Any]:
        """【串行处理】单个用户的所有内容批次"""
        user_id = user_task['user_id']
        blocks = user_task['blocks']
        api_key = user_task['api_key']
        model = user_task['model']
        batch_size = user_task['batch_size']
        
        logger.info(f"用户 {user_id} 开始处理 {len(blocks)} 条内容 (使用 API Key: ...{api_key[-4:]})")
        
        batches = [
            {
                'blocks': blocks[i:i+batch_size],
                'index': i // batch_size,
                'model': model,
                'api_key': api_key
            }
            for i in range(0, len(blocks), batch_size)
        ]
        
        all_processed_blocks = []
        for batch_info in batches:
            result = self.process_single_batch(batch_info)
            all_processed_blocks.extend(result.get("blocks", []))
        
        logger.info(f"用户 {user_id} 处理完成")
        return {"user_id": user_id, "blocks": all_processed_blocks, "success": True}

    def analyze_recent_track(self, user_blocks_dict: Dict[str, List[Dict[str, Any]]], 
                           model: str = "qwen-turbo",
                           batch_size: int = 15,
                           save_results: bool = True) -> Dict[str, Any]:
        """【核心改进】使用“任务队列”模型并行分析所有用户"""
        if not user_blocks_dict:
            return {}
        
        # 【已修正】使用生成器表达式来正确检查所有API密钥
        if not self.api_keys or all(k.startswith("sk-2a9c") for k in self.api_keys):
            logger.error("API密钥无效或未配置，请在环境变量中设置 QWEN_API_KEY。如果是多个key，请用逗号隔开。")
            return {}

        task_queue = Queue()
        for user_id, blocks in user_blocks_dict.items():
            if blocks:
                task_queue.put({
                    'user_id': user_id,
                    'blocks': blocks,
                    'model': model,
                    'batch_size': batch_size
                })
        
        total_tasks = task_queue.qsize()
        if total_tasks == 0:
            logger.info("没有需要处理的用户。")
            return {}
        
        logger.info(f"并行分析开始，共{total_tasks}个用户任务，使用{len(self.api_keys)}个API密钥作为并发工作线程。")
        
        results = {}

        def worker(api_key: str):
            """每个工作线程使用一个独立的API Key，不断从队列中取任务并处理"""
            while not task_queue.empty():
                try:
                    user_task = task_queue.get_nowait()
                except Empty:
                    break
                
                user_task['api_key'] = api_key
                
                try:
                    result = self.process_user_sequentially(user_task)
                    results[result['user_id']] = {"blocks": result['blocks']}
                except Exception as e:
                    user_id = user_task.get('user_id', '未知用户')
                    logger.error(f"处理用户 {user_id} 时发生严重错误: {e}")
                    results[user_id] = {"blocks": []}
                finally:
                    task_queue.task_done()

        num_workers = min(len(self.api_keys), total_tasks)
        with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='API_Worker') as executor:
            executor.map(worker, self.api_keys[:num_workers])

        if save_results and results:
            self.save_analysis_results(results, "recent_ai_analysis.json")
        
        logger.info(f"并行分析完成，成功处理{len(results)}/{total_tasks}个用户。")
        return results

    def analyze_from_json_file(self, json_file_path: str = None) -> Dict[str, Any]:
        """从JSON文件读取数据并进行AI分析
        
        Args:
            json_file_path: JSON文件路径，如果为None则使用默认路径
            
        Returns:
            分析结果字典
        """
        # 统一使用与前端相同的文件路径
        if json_file_path is None:
            json_file_path = "history_track/recent_user_track.json"
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"成功加载JSON文件: {json_file_path}")
            
            # 格式化数据为分析所需格式，确保每个block都有id字段
            user_blocks_dict = {}
            for user_id, blocks in data.items():
                # 确保每个block都有必要的字段
                formatted_blocks = []
                for i, block in enumerate(blocks):
                    formatted_block = dict(block)
                    # 确保有id字段
                    if 'id' not in formatted_block:
                        formatted_block['id'] = f"{user_id}_{i}"
                    # 确保有title和content字段
                    if 'title' not in formatted_block:
                        formatted_block['title'] = formatted_block.get('title', '无标题')
                    if 'content' not in formatted_block:
                        formatted_block['content'] = formatted_block.get('content', '')
                    formatted_blocks.append(formatted_block)
                
                if formatted_blocks:
                    user_blocks_dict[user_id] = formatted_blocks
            
            if not user_blocks_dict:
                logger.warning("没有找到有效的用户数据进行AI分析")
                return {}
            
            logger.info(f"准备分析 {len(user_blocks_dict)} 个用户的数据")
            
            # 执行AI分析
            results = self.analyze_recent_track(user_blocks_dict, save_results=True)
            return results
            
        except FileNotFoundError:
            logger.error(f"JSON文件未找到: {json_file_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON文件格式错误: {e}")
            return {}
        except Exception as e:
            logger.error(f"读取JSON文件时发生错误: {e}")
            return {}


def analyze_from_json_file(json_file_path: str = None) -> Dict[str, Any]:
    """从JSON文件读取数据并进行AI分析的便捷函数
    
    Args:
        json_file_path: JSON文件路径，如果为None则使用默认路径
        
    Returns:
        分析结果字典
    """
    service = AIAnalysisService()
    return service.analyze_from_json_file(json_file_path)

if __name__ == "__main__":
    print("测试并行AI分析服务...")

    test_data = {}
    user_counts = {'小': 15, '中': 7, '大': 3}
    content_ranges = {'小': (5, 15), '中': (20, 40), '大': (50, 100)}

    for user_type, count in user_counts.items():
        for i in range(count):
            user_id = f"{user_type}用户_{i+1}"
            num_contents = random.randint(*content_ranges[user_type])
            test_data[user_id] = [
                {
                    'id': f"{user_id}_{j}",
                    'title': f"测试标题 {j}",
                    'content': f"这是用户 {user_id} 的第 {j} 条测试内容。" * random.randint(1, 5)
                }
                for j in range(num_contents)
            ]

    print(f"测试数据准备完成：{len(test_data)}个用户")

    # 提示：请确保设置了有效的 QWEN_API_KEY 环境变量，否则程序会使用默认的测试密钥并可能因无效而失败。
    # Windows PowerShell: $env:QWEN_API_KEY="sk-key1,sk-key2"
    # Linux/macOS: export QWEN_API_KEY="sk-key1,sk-key2"
    service = AIAnalysisService()
    start_time = time.time()
    
    results = service.analyze_recent_track(test_data, batch_size=15, save_results=True)
    
    end_time = time.time()
    
    print(f"并行分析完成！")
    print(f"总耗时: {end_time - start_time:.2f}秒")
    print(f"处理用户数: {len(results)}个")
    
    successful_users = 0
    for user_id, data in results.items():
        if data.get('blocks') and "错误" not in data['blocks'][0].get('summary', ''):
            successful_users += 1
        print(f"  - {user_id}: {len(data.get('blocks', []))} 条内容已处理")
    print(f"成功摘要的用户数: {successful_users} / {len(results)}")
