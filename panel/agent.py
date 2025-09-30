from typing import Optional, cast, List

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import AIMessage
from langgraph.graph import END
from pydantic import BaseModel, Field

from typing import Any
from .configs import LLMConfig
from .states import AgentState, ToolState, StateKeys
from .graph import Graph
from .tool_graph import ToolGraph


# TODO: exchange the Optional[str] to a Literal with the agent names
class AgentDecision(BaseModel):
    """Structured output schema for an panel agent's decision."""
    answer: str = Field(..., description="What this agent says to the panel.")
    next_agent: Optional[str] = Field(
        None,
        description="Exact name of the next agent to speak; null to end."
    )
    reasoning: Optional[str] = Field(
        None, description="(Optional) brief rationale for routing and answer."
    )


class AgentGraph(Graph):
    """Parent graph that orchestrates tool usage and agent responses.

    This graph optionally calls a ToolGraph subgraph first to handle
    tool invocations, then passes the conversation state to an LLM
    that outputs a structured `AgentDecision`.

    Parameters
    ----------
    name : str
        Name of the agent graph.

    llm_config : LLMConfig
        Configuration object for the LLM backend.

    tools : list[Any]
        List of tools available to this agent.

    tool_response_verification : bool, default=True
        Whether to verify tool call responses before finalizing.

    system_prompt : str, optional
        Optional system message prepended to the conversation.
    """

    def __init__(
            self,
            name: str,
            llm_config: LLMConfig,
            description: str,
            tools: list[Any],
            tool_response_verification: bool = True,
            system_prompt: Optional[str] = None,
    ) -> None:
        super(Graph).__init__(name=name, llm_config=llm_config,
                              description=description)
        self.system_prompt = system_prompt
        self.tools = tools
        self._build_model()
        if self.tools:
            self.tool_graph = ToolGraph(
                name=self.name.join("_tool_graph"),
                llm_config=llm_config,
                tools=tools,
                relevance_verification=tool_response_verification,
                system_prompt=system_prompt,
                description="",
            ).initialize()
        self.structured_model = self.model.with_structured_output(AgentDecision)

    def register_panel_partners(self, panel_partners: List[Graph]):
        self.register = [agent.name for agent in panel_partners]
        self.partners_info = ""
        for agent in panel_partners:
            self.partners_info += str(agent)

    def initialize(self) -> CompiledStateGraph:
        """Initialize the LangGraph state machine for this panel agent.

        Returns
        -------
        CompiledGraph
            A compiled graph object that can be invoked with an AgentState.
        """
        graph = StateGraph(AgentState)
        graph.add_node(f"{self.name}_response_node", self.response_node)
        if self.tools:
            graph.add_node(f"{self.name}_tool_graph", self.execute_tool_graph)
            graph.set_entry_point(f"{self.name}_tool_graph")
            graph.add_edge(start_key=f"{self.name}_tool_graph",
                           end_key=f"{self.name}_response_node")
            graph.add_edge(start_key=f"{self.name}_response_node", end_key=END)
        else:
            graph.set_entry_point(f"{self.name}_response_graph")
            graph.add_edge(start_key=f"{self.name}_response_key", end_key=END)
        return graph.compile()

    async def execute_tool_graph(self, state: AgentState):
        """Run the ToolGraph subgraph to execute any tool calls.

        Parameters
        ----------
        state : AgentState
            Current agent state, including conversation messages.

        Returns
        -------
        dict
            Dictionary with updated messages returned from the ToolGraph.
        """
        messages = self.__get_messages(
            state=state,
            system_prompt=self.system_prompt
        )

        tool_state: ToolState = cast(
            ToolState, {
                StateKeys.MESSAGES: messages,
                StateKeys.VERIFICATION_SCORE: 0.0,
                StateKeys.REMAINING_CYCLES: 3
            }
        )
        result = await self.tool_graph.ainvoke(tool_state)
        return {StateKeys.MESSAGES: result[StateKeys.MESSAGES]}

    async def response_node(self, state: AgentState):
        """Generate a structured agent decision after tool execution.

        Parameters
        ----------
        state : AgentState
            Current agent state with messages from prior turns.

        Returns
        -------
        dict
            Dictionary containing:
            - ``messages`` : list of the message history with the agent's
                                answer.
            - ``next_agent`` : str or None, indicating the next agent.
            - ``reasoning`` : str or None, rationale for the decision.
        """
        system_prompt = self.system_prompt + "\n\n" + self.__get_partner_info()
        messages = self.__get_messages(
            state=state,
            system_prompt=system_prompt
        )
        result = await self.structured_model.ainvoke(messages)

        ai_msg = AIMessage(content=result.answer)

        return {
            StateKeys.MESSAGES: [ai_msg],
            StateKeys.NEXT_AGENT: result.next_agent,
            StateKeys.REASONING: result.reasoning
        }

    def __get_partner_info(self) -> str:
        if self.register:
            partner_info_string = (f"You can route your response to the next "
                                   f"agent for critical evaluation or for "
                                   f"extending the knowledge about other "
                                   f"aspects among the following "
                                   f"candidates: {self.register}. Considering "
                                   f"the following descriptions: \n "
                                   f"{self.partners_info}")
            return partner_info_string
