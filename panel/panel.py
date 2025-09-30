from typing import List, Any, cast, Optional, Dict

from langgraph.types import Command
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import END, START

from .agent import AgentGraph
from .states import PanelState, AgentState, StateKeys


class PanelAgentExecutor:
    """Adapter node that executes a single agent subgraph within the panel.

    This executor:
    - Takes the current ``PanelState`` as input.
    - Builds an ``AgentState`` and invokes the compiled agent subgraph.
    - Decrements the number of remaining hops.
    - Decides which node to route to next (another agent or END).
    - Returns a ``Command`` object with the updated state.

    Parameters
    ----------
    agent : AgentGraph
        The agent graph to be wrapped and executed as a node.
    """

    def __init__(self, agent: AgentGraph):
        assert agent.register, (
            "Please register the other panel members before using this adapter "
            "to the agent."
        )
        self.name = agent.name + "_executor"
        self._agent_name = agent.name
        self.agent: CompiledStateGraph = agent.initialize()

    async def __call__(self, state: PanelState) -> Command:
        """Invoke the wrapped agent graph and produce the next command.

        Parameters
        ----------
        state : PanelState
            The current state of the panel including messages, visited agents,
            and remaining hops.

        Returns
        -------
        Command
            A command containing:
            - ``goto`` : str or END, next node to execute.
            - ``update`` : dict with updated messages, visited list, and
                            hops_left.
        """
        hops_left: int = state[StateKeys.HOPS_LEFT]
        agent_state: AgentState = cast(
            AgentState,
            {
                StateKeys.MESSAGES: state[StateKeys.MESSAGES],
                StateKeys.NEXT_AGENT: None,
                StateKeys.REASONING: None
            }
        )
        results: AgentState = await self.agent.ainvoke(agent_state)

        next_agent: Optional[str] = results.get(StateKeys.NEXT_AGENT)
        hops_left -= 1
        goto_node: str
        if hops_left > 0 and next_agent:
            goto_node = next_agent
        else:
            goto_node = END

        new_messages = results[StateKeys.MESSAGES]
        visited = state.get(StateKeys.VISITED, [])
        visited = visited + [self._agent_name]

        return Command(
            goto=goto_node,
            update={
                StateKeys.MESSAGES: new_messages,
                StateKeys.VISITED: visited,
                StateKeys.HOPS_LEFT: hops_left,
            },
        )


class Panel:
    """Panel graph orchestrating multiple agents with a primary entry point.

    The panel:
    - Wraps each ``AgentGraph`` in a ``PanelAgentExecutor`` node.
    - Starts execution from the ``primary_member``.
    - Routes between agents according to their outputs (``next_agent``).
    - Terminates when hops are exhausted or no next agent is provided.

    Parameters
    ----------
    panel_name : str
        Name of the panel graph.
    panel_members : list of AgentGraph
        All agent graphs participating in the panel.
    primary_member : AgentGraph
        The agent that receives the initial input and starts the chain of
        debate.
    max_hops : int
        Maximum number of agent handoffs allowed before forced termination.
    actions : list, optional
        Additional actions or metadata associated with the panel.
    """

    def __init__(
            self,
            panel_name: str,
            panel_members: List[AgentGraph],
            primary_member: AgentGraph,
            max_hops: int,
            actions: List[Any] | None
    ):
        self.panel_name = panel_name
        self.panel_member_names: List[str] = [
            member.name for member in panel_members
        ]
        self.max_hops = max_hops
        self.actions = actions or []
        assert primary_member.name in self.panel_member_names, (
            f"Sorry, your primary member {primary_member.name} is not included "
            f"in the list of all panel members!"
        )
        self.primary_member = primary_member
        self.agent_executors: Dict[str, PanelAgentExecutor] = {}

        for member in panel_members:
            list_members_self_exclusion = [
                other_member for other_member in panel_members
                if other_member.name != member.name
            ]
            member.register_panel_partners(list_members_self_exclusion)
            self.agent_executors[member.name] = PanelAgentExecutor(agent=member)

    def build_executable_panel(self) -> CompiledStateGraph:
        """Build and compile the executable panel graph.

        Returns
        -------
        CompiledGraph
            A compiled LangGraph state machine for panel orchestration.
        """
        graph = StateGraph(PanelState)
        for member_name, member_executor in self.agent_executors.items():
            graph.add_node(member_name, member_executor)

        graph.add_edge(START, self.primary_member.name)
        return graph.compile()
