from typing import Optional
from langgraph.graph import StateGraph
from langchain_core.messages import ToolMessage, ToolCall
from langchain_core.tools import BaseTool
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from typing import Any
from .configs import LLMConfig
from .states import ToolState, StateKeys
from .graph import Graph


class Verification(BaseModel):
    """Schema for verifying the relevance of tool responses."""
    relevance: float = Field(...,
                             description="An estimated float value between "
                                         "0 and 1, describing how well the "
                                         "tool call responses fit to the "
                                         "context."
                             )


class ToolGraph(Graph):

    """Graph for handling tool invocations and optional verification.

    This subgraph is responsible for:
    - Letting the LLM decide which tools to call.
    - Executing the selected tools.
    - Optionally verifying tool responses before ending.

    Parameters
    ----------
    name : str
        Name of the graph instance.

    llm_config : LLMConfig
        Configuration for the underlying LLM backend.

    tools : list[Any]
        List of available tools that must implement ``BaseTool``.

    relevance_verification : bool, default=True
        Whether to run a verification step on tool responses.

    system_prompt : str, optional
        Optional system message prepended to the conversation.
    """
    def __init__(
            self,
            name: str,
            llm_config: LLMConfig,
            description: str,
            tools: list[Any],
            relevance_verification: bool = True,
            system_prompt: Optional[str] = None,
    ) -> None:
        super(Graph).__init__(name=name, llm_config=llm_config,
                              description=description)
        self.tools = tools
        self.system_prompt = system_prompt
        self._build_model()
        self.relevance_verification = relevance_verification
        if self.tools:
            self.tool_library: dict[str, BaseTool] = {}
            for tool in self.tools:
                assert isinstance(tool, BaseTool), ("Sorry, your tool is not a "
                                                    "BaseTool implementation.")
                assert tool.name, f"There exists no tool name for tool {tool}"
                self.tool_library[tool.name] = tool
            self.tool_model = self.model.bind_tools(self.tools)
            self.verification_model = self.model.with_structured_output(
                Verification
            )

    def initialize(self) -> CompiledStateGraph:
        """Build and compile the state graph for tool execution.

        Returns
        -------
        CompiledGraph
            A compiled LangGraph state machine based on ``ToolState``.
        """
        graph = StateGraph(ToolState)
        graph.add_node("tool_node", self.tool_node)
        graph.set_entry_point("tool_node")

        if self.relevance_verification:
            graph.add_node("verification_node", self.verification_node)
            graph.add_edge(start_key="tool_node", end_key="verification_node")
            graph.add_conditional_edges(
                source="verification_node",
                path=self.__verifier,
                path_map={True: END, False: "tool_node"}
            )
        else:
            graph.add_edge(start_key="tool_node", end_key=END)
        return graph.compile()

    async def tool_node(
            self,
            state: ToolState
    ) -> dict[str, list[ToolMessage]]:
        """Select and execute tools proposed by the LLM.

        Parameters
        ----------
        state : ToolState
            Current tool state, including messages from the conversation.

        Returns
        -------
        dict of str to list of ToolMessage
            Tool messages containing the results of executed tools.
        """
        messages = self.__get_messages(
            state=state,
            system_prompt=self.system_prompt
        )

        selected_tool_calls = await self.tool_model.ainvoke(
            {
                StateKeys.MESSAGES: messages
            }
        )
        tool_results = []
        for tool_call in selected_tool_calls.tool_calls:
            tool_results.append(
                await self.__execute_single_tool(selected_tool_call=tool_call)
            )
        return {StateKeys.MESSAGES: tool_results}

    async def verification_node(
            self,
            state: ToolState
    ) -> dict[str, float | int]:
        """Verify the relevance of tool call results.

        Parameters
        ----------
        state : ToolState
            Current tool state, including messages and cycle counters.

        Returns
        -------
        dict
            Updated state fields:
            - ``remaining_cycles`` : int
                Decremented by one.
            - ``verification_score`` : float
                Score from the verification model.
        """
        remaining_cycles = state[StateKeys.REMAINING_CYCLES]
        messages = self.__get_messages(
            state=state,
            system_prompt=self.system_prompt
        )

        verification = await self.verification_model.ainvoke(
            {
                StateKeys.MESSAGES: messages
            }
        )
        remaining_cycles -= 1
        return {
            StateKeys.REMAINING_CYCLES: remaining_cycles,
            StateKeys.VERIFICATION_SCORE: verification.relevance
        }

    def __verifier(self, state: ToolState):
        """Decide whether to stop or continue tool execution cycles.

        Parameters
        ----------
        state : ToolState
            Current tool state.

        Returns
        -------
        bool
            True if verification passes or cycles are exhausted, else False.
        """
        if state[StateKeys.REMAINING_CYCLES] == 0:
            return True
        if state[StateKeys.VERIFICATION_SCORE] > 0.5:
            return True
        else:
            return False

    async def __execute_single_tool(
            self,
            selected_tool_call: ToolCall
    ) -> ToolMessage:
        """Execute a single tool call and wrap the result in a ToolMessage.

        Parameters
        ----------
        selected_tool_call : ToolCall
            Tool call object containing name, args, and ID.

        Returns
        -------
        ToolMessage
            Message containing the tool result content and artifact.
        """
        tool_name, args = selected_tool_call["name"], selected_tool_call["args"]
        tool_result = await self.tool_library[tool_name].ainvoke(args)
        tool_msg = ToolMessage(
            tool_call_id=selected_tool_call["id"],
            content=tool_result.content,
            artifact=tool_result.artifact
        )
        return tool_msg
