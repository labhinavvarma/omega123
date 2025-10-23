import streamlit as st
import asyncio
import json
import yaml
import requests
import uuid
import urllib3
from typing import List, Dict, Any, Optional, Sequence

from mcp.client.sse import sse_client
from mcp import ClientSession
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable

# Import configuration
from config import config

# Disable SSL warnings if configured
if not config.verify_ssl:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Page config
st.set_page_config(page_title=config.app_title, page_icon=config.app_icon)
st.title(config.app_title)

server_url = st.sidebar.text_input("MCP Server URL", config.mcp_default_url)
show_server_info = st.sidebar.checkbox("🛡 Show MCP Server Info", value=False)


# === Custom SF Assist LLM Wrapper ===
class ChatSnowflakeCortexAPI(BaseChatModel):
    """Custom LangChain wrapper for Snowflake Cortex API via SF Assist"""
    
    session_id: str = None
    model_name: str = config.model
    bound_tools: List[Any] = []
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())
    
    class Config:
        arbitrary_types_allowed = True
    
    def bind_tools(
        self,
        tools: Sequence[BaseTool | dict],
        **kwargs: Any,
    ) -> Runnable:
        """Bind tools to the model - required for ReAct agent"""
        return self.__class__(
            session_id=self.session_id,
            model_name=self.model_name,
            bound_tools=list(tools)
        )
    
    def _format_messages_with_tools(self, messages: List[BaseMessage]) -> List[Dict]:
        """Convert LangChain messages to API format, including tool information"""
        api_messages = []
        
        # Add system message with tool information if tools are bound
        if self.bound_tools:
            tool_descriptions = []
            for tool in self.bound_tools:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    tool_descriptions.append(f"- {tool.name}: {tool.description}")
            
            if tool_descriptions:
                tools_text = "\n".join(tool_descriptions)
                system_content = f"{config.sys_msg}\n\nYou have access to the following tools:\n{tools_text}\n\nTo use a tool, respond with a tool call in your reasoning."
                api_messages.append({"role": "system", "content": system_content})
        
        # Convert messages
        for msg in messages:
            if isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                api_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                if not any(m.get("role") == "system" for m in api_messages):
                    api_messages.append({"role": "system", "content": msg.content})
            else:
                api_messages.append({"role": "user", "content": str(msg.content)})
        
        return api_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation - converts messages and calls SF Assist API"""
        
        # Format messages with tool information
        api_messages = self._format_messages_with_tools(messages)
        
        # Build payload using config
        payload = {
            "query": {
                "aplctn_cd": config.aplctn_cd,
                "app_id": config.app_id,
                "api_key": config.api_key,
                "method": "cortex",
                "model": self.model_name,
                "sys_msg": config.sys_msg,
                "limit_convs": "0",
                "prompt": {
                    "messages": api_messages
                },
                "app_lvl_prefix": "",
                "user_id": "",
                "session_id": self.session_id
            }
        }
        
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "Authorization": f'Snowflake Token="{config.api_key}"'
        }
        
        try:
            response = requests.post(
                config.api_url, 
                headers=headers, 
                json=payload, 
                verify=config.verify_ssl
            )
            
            if response.status_code == 200:
                raw = response.text
                
                # Parse response
                if "end_of_stream" in raw:
                    answer, _, _ = raw.partition("end_of_stream")
                    bot_reply = answer.strip()
                else:
                    bot_reply = raw.strip()
                
                # Return as LangChain ChatResult
                message = AIMessage(content=bot_reply)
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            
            else:
                raise Exception(f"API Error {response.status_code}: {response.text}")
                
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation"""
        return self._generate(messages, stop, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM"""
        return "snowflake_cortex_api"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters"""
        return {
            "model_name": self.model_name,
            "session_id": self.session_id,
        }


# === Server Info ===
if show_server_info:
    async def fetch_mcp_info():
        result = {"resources": [], "tools": [], "prompts": [], "yaml": [], "search": []}
        try:
            async with sse_client(url=server_url) as sse_connection:
                async with ClientSession(*sse_connection) as session:
                    await session.initialize()
 
                    # --- Resources ---
                    resources = await session.list_resources()
                    if hasattr(resources, 'resources'):
                        for r in resources.resources:
                            result["resources"].append({"name": r.name})
                   
                    # --- Tools (filtered) ---
                    tools = await session.list_tools()
                    hidden_tools = {"add-frequent-questions", "add-prompts", "suggested_top_prompts"}
                    if hasattr(tools, 'tools'):
                        for t in tools.tools:
                            if t.name not in hidden_tools:
                                result["tools"].append({"name": t.name})
 
                    # --- Prompts ---
                    prompts = await session.list_prompts()
                    if hasattr(prompts, 'prompts'):
                        for p in prompts.prompts:
                            args = []
                            if hasattr(p, 'arguments'):
                                for arg in p.arguments:
                                    args.append(f"{arg.name} ({'Required' if arg.required else 'Optional'}): {arg.description}")
                            result["prompts"].append({
                                "name": p.name,
                                "description": getattr(p, 'description', ''),
                                "args": args
                            })
 
                    # --- YAML Resources ---
                    try:
                        yaml_content = await session.read_resource("schematiclayer://cortex_analyst/schematic_models/hedis_stage_full/list")
                        if hasattr(yaml_content, 'contents'):
                            for item in yaml_content.contents:
                                if hasattr(item, 'text'):
                                    parsed = yaml.safe_load(item.text)
                                    result["yaml"].append(yaml.dump(parsed, sort_keys=False))
                    except Exception as e:
                        result["yaml"].append(f"YAML error: {e}")
 
                    # --- Search Objects ---
                    try:
                        content = await session.read_resource("search://cortex_search/search_obj/list")
                        if hasattr(content, 'contents'):
                            for item in content.contents:
                                if hasattr(item, 'text'):
                                    objs = json.loads(item.text)
                                    result["search"].extend(objs)
                    except Exception as e:
                        result["search"].append(f"Search error: {e}")
 
        except Exception as e:
            st.sidebar.error(f"❌ MCP Connection Error: {e}")
        return result
 
    mcp_data = asyncio.run(fetch_mcp_info())
 
    # Display Resources
    with st.sidebar.expander("📦 Resources", expanded=False):
        for r in mcp_data["resources"]:
            if "cortex_search/search_obj/list" in r["name"]:
                display_name = "Cortex Search"
            else:
                display_name = r["name"]
            st.markdown(f"**{display_name}**")
    
    # --- YAML Section ---
    with st.sidebar.expander("Schematic Layer", expanded=False):
        for y in mcp_data["yaml"]:
            st.code(y, language="yaml")
 
    # --- Tools Section (Filtered) ---
    with st.sidebar.expander("🛠 Tools", expanded=False):
        for t in mcp_data["tools"]:
            st.markdown(f"**{t['name']}**")
 
    # Display Prompts
    with st.sidebar.expander("🧐 Prompts", expanded=False):
        for p in mcp_data["prompts"]:
            st.markdown(f"**{p['name']}**")
else:
    # Chat functionality with SF Assist
    @st.cache_resource
    def get_model():
        """Get SF Assist LLM model"""
        return ChatSnowflakeCortexAPI(model_name=config.model)
    
    prompt_type = st.sidebar.radio("Select Prompt Type", ["Calculator", "HEDIS Expert", "Weather", "No Context"])
    prompt_map = {
        "Calculator": "calculator-prompt",
        "HEDIS Expert": "hedis-prompt",
        "Weather": "weather-prompt",
        "No Context": None
    }
 
    examples = {
        "Calculator": ["(4+5)/2.0", "sqrt(16) + 7", "3^4 - 12"],
        "HEDIS Expert": [],
        "Weather": [
            "What is the present weather in Richmond?",
            "What's the weather forecast for Atlanta?",
            "Is it raining in New York City today?"
        ],
        "No Context": ["Who won the world cup in 2022?", "Summarize climate change impact on oceans"]
    }
 
    if prompt_type == "HEDIS Expert":
        try:
            async def fetch_hedis_examples():
                async with sse_client(url=server_url) as sse_connection:
                    async with ClientSession(*sse_connection) as session:
                        await session.initialize()
                        content = await session.read_resource("genaiplatform://hedis/frequent_questions/Initialization")
                        if hasattr(content, "contents"):
                            for item in content.contents:
                                if hasattr(item, "text"):
                                    examples["HEDIS Expert"].extend(json.loads(item.text))
   
            asyncio.run(fetch_hedis_examples())
        except Exception as e:
            examples["HEDIS Expert"] = [f"⚠️ Failed to load examples: {e}"]
 
    if "messages" not in st.session_state:
        st.session_state.messages = []
 
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
 
    with st.sidebar.expander("Example Queries", expanded=True):
        for example in examples[prompt_type]:
            if st.button(example, key=example):
                st.session_state.query_input = example
 
    query = st.chat_input("Type your query here...")
    if "query_input" in st.session_state:
        query = st.session_state.query_input
        del st.session_state.query_input
 
    async def process_query(query_text):
        st.session_state.messages.append({"role": "user", "content": query_text})
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.text("Processing...")
            try:
                # Initialize MCP client WITHOUT context manager
                client = MultiServerMCPClient(
                    {config.mcp_server_name: {"url": server_url, "transport": config.mcp_transport}}
                )
                
                # Get tools from client
                tools = await client.get_tools()
                
                # Get model
                model = get_model()
                
                # Create agent
                agent = create_react_agent(model=model, tools=tools)
                
                # Get prompt from server if available
                prompt_name = prompt_map[prompt_type]
                if prompt_name:
                    prompt_from_server = await client.get_prompt(
                        server_name=config.mcp_server_name,
                        prompt_name=prompt_name,
                        arguments={}
                    )
                    if "{query}" in prompt_from_server[0].content:
                        formatted_prompt = prompt_from_server[0].content.format(query=query_text)
                    else:
                        formatted_prompt = prompt_from_server[0].content + query_text
                else:
                    formatted_prompt = query_text
                
                # Invoke agent
                response = await agent.ainvoke({"messages": formatted_prompt})
                result = list(response.values())[0][1].content
                message_placeholder.text(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.text(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
 
    if query:
        asyncio.run(process_query(query))
 
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
 
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"{config.app_title} v{config.app_version} (SF Assist)")
