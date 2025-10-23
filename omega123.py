import json
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)
import requests
import uuid
import urllib3

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.utils import _build_model_kwargs
from pydantic import Field, SecretStr, model_validator

# Disable SSL warnings for internal/dev environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === SF Assist API Configuration ===
API_URL = "https://sfassist.edagenaidev.awsdns.internal.das/api/cortex/complete"
API_KEY = "78a799ea-a0f6-11ef-a0ce-15a449f7a8b0"
APP_ID = "edadip"
APLCTN_CD = "edagnai"
SYS_MSG = "You are a powerful AI assistant. Provide accurate, concise answers based on context."

SUPPORTED_ROLES: List[str] = [
    "system",
    "user",
    "assistant",
]

import re


class ChatSnowflakeCortexError(Exception):
    """Error with Snowpark client."""


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any] = {
        "content": message.content,
    }

    # Populate role and additional message data
    if isinstance(message, ChatMessage) and message.role in SUPPORTED_ROLES:
        message_dict["role"] = message.role
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _truncate_at_stop_tokens(
    text: str,
    stop: Optional[List[str]],
) -> str:
    """Truncates text at the earliest stop token found."""
    if stop is None:
        return text

    for stop_token in stop:
        stop_token_idx = text.find(stop_token)
        if stop_token_idx != -1:
            text = text[:stop_token_idx]
    return text


class ChatSnowflakeCortex(BaseChatModel):
    """Snowflake Cortex based Chat model - Modified to use SF Assist API

    To use the chat model, you must have the ``snowflake-snowpark-python`` Python
    package installed and either:

        1. environment variables set with your snowflake credentials or
        2. directly passed in as kwargs to the ChatSnowflakeCortex constructor.

    Example:
        .. code-block:: python

            from llmobject_wrapper import ChatSnowflakeCortex
            chat = ChatSnowflakeCortex()
    """

    test_tools: Dict[str, Union[Dict[str, Any], Type, Callable, BaseTool]] = Field(
        default_factory=dict
    )

    session: Any = None
    """Snowpark session object (kept for compatibility but not used with SF Assist)."""

    model: str = "llama3.1-70b"
    """Model name for SF Assist API."""

    cortex_function: str = "complete"
    """Cortex function to use, defaulted to `complete`."""

    temperature: float = 0
    """Model temperature. Value should be >= 0 and <= 1.0"""

    max_tokens: Optional[int] = None
    """The maximum number of output tokens in the response."""

    top_p: Optional[float] = 0
    """top_p adjusts the number of choices for each predicted tokens based on
        cumulative probabilities. Value should be ranging between 0.0 and 1.0.
    """

    # Keep these for compatibility but they're not used with SF Assist
    snowflake_username: Optional[str] = Field(default=None, alias="username")
    snowflake_password: Optional[SecretStr] = Field(default=None, alias="password")
    snowflake_account: Optional[str] = Field(default=None, alias="account")
    snowflake_database: Optional[str] = Field(default=None, alias="database")
    snowflake_schema: Optional[str] = Field(default=None, alias="schema")
    snowflake_warehouse: Optional[str] = Field(default=None, alias="warehouse")
    snowflake_role: Optional[str] = Field(default=None, alias="role")

    # SF Assist session ID
    sf_assist_session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = "auto",
        **kwargs: Any,
    ) -> "ChatSnowflakeCortex":
        """Bind tool-like objects to this chat model, ensuring they conform to
        expected formats."""

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        formatted_tools_dict = {
            tool["name"]: tool for tool in formatted_tools if "name" in tool
        }
        self.test_tools.update(formatted_tools_dict)

        print("Tools bound to chat model:")
        print(formatted_tools)
        print(formatted_tools_dict)
        print(self.test_tools)
        print("########################################")
        return self

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        values = _build_model_kwargs(values, all_required_field_names)
        return values

    def __del__(self) -> None:
        if getattr(self, "session", None) is not None:
            self.session.close()

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return f"snowflake-cortex-{self.model}-sfassist"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using SF Assist API instead of Snowflake Cortex"""
        
        # Convert messages to API format
        message_dicts = [_convert_message_to_dict(m) for m in messages]

        # Add tool information to system message if tools are bound
        if self.test_tools:
            tool_descriptions = []
            for tool_name, tool_def in self.test_tools.items():
                if isinstance(tool_def, dict):
                    description = tool_def.get('description', '')
                    tool_descriptions.append(f"- {tool_name}: {description}")
            
            if tool_descriptions:
                tools_text = "\n".join(tool_descriptions)
                system_msg = f"{SYS_MSG}\n\nYou have access to the following tools:\n{tools_text}"
                # Add or update system message
                has_system = any(m.get("role") == "system" for m in message_dicts)
                if not has_system:
                    message_dicts.insert(0, {"role": "system", "content": system_msg})

        # Build SF Assist API payload
        payload = {
            "query": {
                "aplctn_cd": APLCTN_CD,
                "app_id": APP_ID,
                "api_key": API_KEY,
                "method": "cortex",
                "model": self.model,
                "sys_msg": SYS_MSG,
                "limit_convs": "0",
                "prompt": {
                    "messages": message_dicts
                },
                "app_lvl_prefix": "",
                "user_id": "",
                "session_id": self.sf_assist_session_id
            }
        }

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "Authorization": f'Snowflake Token="{API_KEY}"'
        }

        try:
            # Call SF Assist API
            response = requests.post(API_URL, headers=headers, json=payload, verify=False)

            if response.status_code == 200:
                raw = response.text
                
                # Parse response
                if "end_of_stream" in raw:
                    answer, _, _ = raw.partition("end_of_stream")
                    ai_message_content = answer.strip()
                else:
                    ai_message_content = raw.strip()

                content = _truncate_at_stop_tokens(ai_message_content, stop)
                message = AIMessage(
                    content=content,
                    response_metadata={"model": self.model, "session_id": self.sf_assist_session_id},
                )
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            else:
                raise ChatSnowflakeCortexError(
                    f"SF Assist API Error {response.status_code}: {response.text}"
                )

        except Exception as e:
            raise ChatSnowflakeCortexError(
                f"Error while making request to SF Assist API: {e}"
            )

    def _stream_content(
        self, content: str, stop: Optional[List[str]]
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream the output of the model in chunks to return ChatGenerationChunk.
        """
        chunk_size = 50  # Define a reasonable chunk size for streaming
        truncated_content = _truncate_at_stop_tokens(content, stop)

        for i in range(0, len(truncated_content), chunk_size):
            chunk_content = truncated_content[i : i + chunk_size]

            # Create and yield a ChatGenerationChunk with partial content
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_content))

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model in chunks to return ChatGenerationChunk."""
        
        # Use the same _generate method and stream its output
        result = self._generate(messages, stop, run_manager, **kwargs)
        content = result.generations[0].message.content
        
        for chunk in self._stream_content(content, stop):
            yield chunk
