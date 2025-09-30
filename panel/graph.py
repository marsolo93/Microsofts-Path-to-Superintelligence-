from typing import List

from abc import ABC
import abc

from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, \
    SystemMessage
from langgraph.graph import MessagesState
from langchain_openai import AzureChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from .configs import LLMConfig


class Graph(ABC):
    """Abstract base class for building LangGraph subgraphs with an LLM backend.

    This class initializes an LLM client (AzureChatOpenAI) from a given
    configuration and provides utility methods for child graphs to build
    and preprocess message histories.

    Parameters
    ----------
    name : str
        Name of the graph instance.
    llm_config : LLMConfig
        Configuration for the underlying LLM (model, endpoint, API key, etc.).

    Attributes
    ----------
    name : str
        Name of the graph.
    llm_config : LLMConfig
        Configuration used to create the model.
    model : AzureChatOpenAI
        LLM client bound to the given Azure configuration.
    """

    def __init__(
            self,
            name: str,
            llm_config: LLMConfig,
            description: str
    ) -> None:
        self.name = name
        self.llm_config = llm_config
        self.description = description
        self._build_model()

    def __str__(self) -> str:
        name_str = f"NAME: {self.name}"
        description_str = f"TASK: {self.description}"
        return name_str + "\n" + description_str + "\n\n"

    def _build_model(self):
        """Instantiate the AzureChatOpenAI model based on the LLM config.

        Notes
        -----
        This method is called automatically during initialization and
        stores the resulting model in ``self.model``.
        """
        self.model = AzureChatOpenAI(
            model=self.llm_config.llm,
            temperature=self.llm_config.temperature,
            api_key=self.llm_config.api_key,  # type: ignore
            azure_endpoint=self.llm_config.endpoint,
            api_version=self.llm_config.api_version,
            timeout=self.llm_config.time_out,
            max_tokens=self.llm_config.max_tokens,
        )

    @abc.abstractmethod
    def initialize(self) -> CompiledStateGraph:
        """Abstract method to build and compile the state graph.

        Returns
        -------
        CompiledGraph
            A compiled LangGraph state machine specialized by subclasses.
        """
        raise NotImplementedError()

    @staticmethod
    def __get_messages(
            state: MessagesState,
            system_prompt: str
    ) -> List[HumanMessage | AIMessage | ToolMessage | SystemMessage]:
        """Assemble conversation messages with a system prompt.

        Parameters
        ----------
        state : MessagesState
            Current graph state containing the list of conversation messages.
        system_prompt : str
            System-level instruction prepended to the conversation.

        Returns
        -------
        list of (HumanMessage | AIMessage | ToolMessage | SystemMessage)
            Messages formatted for LLM input, beginning with a
            SystemMessage, followed by all messages from the state.
        """
        messages = [SystemMessage(content=system_prompt)] + [
            msg if isinstance(msg, (HumanMessage, AIMessage, ToolMessage))
            else HumanMessage(content=msg["content"])
            for msg in state['messages']
        ]
        return messages
