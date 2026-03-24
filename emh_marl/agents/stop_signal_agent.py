"""
emh_marl/agents/stop_signal_agent.py

StopSignalAgent — Tick-level synchronizer for multi-agent RL within ABIDES.

This agent does not place orders on its own behalf.  On every simulation tick
it orchestrates a fixed roster of MARL agents:

  1. Computes each agent's reward from the previous step.
  2. Collects each agent's current observation.
  3. Checks terminal/auxiliary conditions.
  4. Calls each agent's DQN policy to get a discrete action.
  5. Translates actions to ABIDES-compatible Order objects.
  6. Submits orders to the ExchangeAgent in FIFO registration order.
  7. Notifies each agent of the complete (s, a, r, done, info) tuple.

All interactions with registered MARL agents are direct synchronous method
calls — no message-passing.  Because ABIDES is single-threaded, the kernel is
naturally paused for the entire duration of wakeup(), giving complete
tick-level synchronization across all registered MARL agents with no gym
wrapper required.

MARL agents must satisfy the interface documented in MARLAgentProtocol.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from abides_core import Agent, NanosecondTime
from abides_core.utils import str_to_ns, fmt_ts
from abides_markets.agents import ExchangeAgent
from abides_markets.messages.order import LimitOrderMsg, MarketOrderMsg
from abides_markets.orders import LimitOrder, MarketOrder, Order

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MARL agent interface documentation
# ---------------------------------------------------------------------------

class MARLAgentProtocol:
    """
    Documents the interface every MARL agent registered with StopSignalAgent
    must implement.  This class is never instantiated directly; it exists
    purely for reference and type-checking.

    All methods are called synchronously inside StopSignalAgent.wakeup()
    once per tick.  StopSignalAgent calls them in the order listed below.

    Call order per tick
    -------------------
    1. compute_reward()          — reward for the *previous* tick's action
    2. get_observation()         — observation for the *current* tick
    3. check_done()              — is this agent's episode over?
    4. get_info()                — auxiliary diagnostic data (may be empty)
    5. compute_action(obs)       — DQN forward pass → discrete action index
    6. action_to_order(a, t)     — translate action index → ABIDES Order
    7. notify_step_result(...)   — deliver completed (s, a, r, done, info)

    Methods
    -------
    get_observation() -> Any
        Return the current observation vector / feature dict.
        Must reflect simulation state as of the current tick.

    compute_reward() -> float
        Return the scalar reward for the action taken in the *previous* tick.
        Called before get_observation() so the reward reflects consequences
        of the last action before the next state is presented.

    check_done() -> bool
        Return True if this agent's episode should be considered terminal
        (e.g. insolvency, end-of-day logic, or environment signal).

    get_info() -> Dict[str, Any]
        Return auxiliary diagnostic data (e.g. portfolio value, spread).
        May be an empty dict.

    compute_action(observation: Any) -> int
        Run the DQN policy on *observation* and return a discrete action index.

    action_to_order(action: int, current_time: NanosecondTime) -> Optional[Order]
        Translate a discrete action index to an ABIDES Order instance.
        Return None for the "hold / do nothing" action.

    notify_step_result(
        observation: Any,
        action: Optional[int],
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None
        Receive the completed (s, a, r, done, info) transition after all
        orders for the tick have been submitted (e.g. to store to a replay
        buffer or update agent-internal logs).
    """


# ---------------------------------------------------------------------------
# StopSignalAgent
# ---------------------------------------------------------------------------

class StopSignalAgent(Agent):
    """
    Tick-level synchronizer for a collection of MARL agents within ABIDES.

    This agent is the single point of control for all registered MARL agents.
    It inherits from abides_core.Agent and registers with the kernel like any
    other ABIDES agent, but never submits orders for its own benefit.

    Parameters
    ----------
    id : int
        Unique agent ID assigned by the experiment configuration.
    marl_agents : list
        Ordered list of MARL agent objects satisfying MARLAgentProtocol.
        Registration order is preserved and determines FIFO priority for
        order submission when multiple agents act in the same tick.
    tick_size : NanosecondTime
        Simulation duration of one tick (time between consecutive wakeups).
        Default: 5 simulated seconds — str_to_ns("00:00:05").
        Configurable per experiment; constant once the simulation starts.
    exchange_id : int | None
        ID of the ExchangeAgent to route orders to.
        If None, the first ExchangeAgent found via the kernel is used.
        Override this when running multiple exchange agents.
    name : str | None
        Human-readable agent name (defaults to "StopSignalAgent_<id>").
    type : str | None
        Agent type string for kernel aggregation.
    random_state : np.random.RandomState | None
        Seeded RNG forwarded to the Agent base class.
    log_events : bool
        Enable event logging (forwarded to Agent base class).
    log_to_file : bool
        Write event log to disk at termination (forwarded to Agent base class).

    Attributes
    ----------
    current_tick : int
        Number of ticks fully processed so far (read-only via property).
    exchange_id : int
        Resolved ID of the target ExchangeAgent (set in kernel_starting).
    marl_agents : list
        Registered MARL agents in FIFO priority order (immutable after init).
    """

    def __init__(
        self,
        id: int,
        marl_agents: List[Any],
        tick_size: NanosecondTime = str_to_ns("00:00:05"),
        exchange_id: Optional[int] = None,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_events: bool = True,
        log_to_file: bool = True,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            type=type,
            random_state=random_state,
            log_events=log_events,
            log_to_file=log_to_file,
        )

        # Registration order is FIFO submission priority — list is immutable.
        self.marl_agents: List[Any] = list(marl_agents)

        # Tick duration: constant throughout the experiment.
        self.tick_size: NanosecondTime = tick_size

        # Optionally pre-specify the exchange agent's ID.
        self._exchange_id_override: Optional[int] = exchange_id

        # Resolved during kernel_starting once all agents are instantiated.
        self.exchange_id: Optional[int] = None

        # Number of fully completed ticks.
        self._tick_count: int = 0

        # Last discrete action per agent index (None before the first tick).
        # Stored so agents can compute reward w.r.t. the previous action if
        # they need a reference outside their own state.
        self._prev_actions: Dict[int, Optional[int]] = {
            i: None for i in range(len(self.marl_agents))
        }

        # Orders submitted in the most recently completed tick (diagnostic).
        self._last_tick_orders: List[Tuple[int, Order]] = []

        logger.info(
            "[StopSignalAgent %d] Initialized | marl_agents=%d | tick_size=%d ns",
            self.id,
            len(self.marl_agents),
            self.tick_size,
        )

    # -----------------------------------------------------------------------
    # Kernel lifecycle hooks
    # -----------------------------------------------------------------------

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        """
        Called once by the kernel after all agents exist.

        Resolves the ExchangeAgent ID, then delegates to the base class which
        schedules the first wakeup at *start_time*.  Do NOT call set_wakeup
        separately — the base class already does so.
        """
        # Resolve exchange agent ID before super() so it's ready at tick 0.
        if self._exchange_id_override is not None:
            self.exchange_id = self._exchange_id_override
        else:
            matches = self.kernel.find_agents_by_type(ExchangeAgent)
            if not matches:
                raise RuntimeError(
                    "StopSignalAgent: no ExchangeAgent found in the kernel. "
                    "Add one to the experiment config or pass exchange_id "
                    "explicitly to StopSignalAgent.__init__."
                )
            self.exchange_id = matches[0]

        logger.info(
            "[StopSignalAgent %d] Exchange resolved → id=%d | "
            "first wakeup at %s.",
            self.id,
            self.exchange_id,
            fmt_ts(start_time),
        )

        # Base class schedules wakeup at start_time — must come after exchange
        # resolution so the agent is fully configured before the first tick.
        super().kernel_starting(start_time)

    def kernel_stopping(self) -> None:
        """Called once before simulation terminates.  Logs a final summary."""
        super().kernel_stopping()
        logger.info(
            "[StopSignalAgent %d] Stopping after %d ticks.",
            self.id,
            self._tick_count,
        )
        self.logEvent(
            "STOP_SIGNAL_SUMMARY",
            {
                "total_ticks": self._tick_count,
                "num_marl_agents": len(self.marl_agents),
            },
        )

    # -----------------------------------------------------------------------
    # Core tick handler
    # -----------------------------------------------------------------------

    def wakeup(self, current_time: NanosecondTime) -> None:
        """
        Main tick handler — called by the kernel once per tick.

        Because ABIDES is single-threaded, no other agent messages are
        processed while this method executes, providing natural tick-level
        synchronization across all registered MARL agents.

        Tick flow
        ---------
        1. Update agent clock (via super); schedule the *next* tick.
        2. Compute each agent's reward for its previous action.
        3. Collect each agent's current observation.
        4. Check terminal/auxiliary conditions for each agent.
        5. Compute a discrete action for each non-terminal agent.
        6. Translate each action to an ABIDES Order (None = hold).
        7. Submit orders to the ExchangeAgent in FIFO registration order.
        8. Notify each agent of the full (s, a, r, done, info) transition.
        9. Increment tick counter and log the tick event.
        """
        super().wakeup(current_time)

        # ── 1. Schedule next tick ─────────────────────────────────────────
        self.set_wakeup(current_time + self.tick_size)
        self._last_tick_orders.clear()

        logger.debug(
            "[StopSignalAgent %d] Tick %d at %s",
            self.id,
            self._tick_count,
            fmt_ts(current_time),
        )

        # Per-agent results collected this tick.
        observations: Dict[int, Any] = {}
        rewards: Dict[int, float] = {}
        dones: Dict[int, bool] = {}
        infos: Dict[int, Dict[str, Any]] = {}
        actions: Dict[int, Optional[int]] = {}
        # (agent_idx, Order|None) — preserved in registration (FIFO) order.
        pending_orders: List[Tuple[int, Optional[Order]]] = []

        # ── 2–4. Reward / observation / done / info ───────────────────────
        for idx, agent in enumerate(self.marl_agents):
            rewards[idx] = self._compute_reward(idx, agent)
            observations[idx] = self._get_observation(idx, agent)
            dones[idx] = self._check_done(idx, agent)
            infos[idx] = self._get_info(idx, agent)

        # ── 5–6. Action → order (FIFO registration order) ─────────────────
        for idx, agent in enumerate(self.marl_agents):
            if dones[idx]:
                # Terminal agents submit no orders this tick.
                actions[idx] = None
                pending_orders.append((idx, None))
                continue

            action = self._compute_action(idx, agent, observations[idx])
            actions[idx] = action
            self._prev_actions[idx] = action

            order = self._action_to_order(idx, agent, action, current_time)
            pending_orders.append((idx, order))

        # ── 7. Submit to exchange ─────────────────────────────────────────
        self._submit_orders(pending_orders, current_time)

        # ── 8. Notify agents of completed transition ──────────────────────
        for idx, agent in enumerate(self.marl_agents):
            self._notify_step_result(
                idx,
                agent,
                observations[idx],
                actions[idx],
                rewards[idx],
                dones[idx],
                infos[idx],
            )

        # ── 9. Advance tick counter and log ───────────────────────────────
        self._tick_count += 1
        self.logEvent(
            "TICK",
            {
                "tick": self._tick_count,
                "sim_time": current_time,
                "orders_submitted": len(self._last_tick_orders),
            },
        )

    # -----------------------------------------------------------------------
    # Message handler — receives exchange confirmations
    # -----------------------------------------------------------------------

    def receive_message(
        self,
        current_time: NanosecondTime,
        sender_id: int,
        message: Any,
    ) -> None:
        """
        Handles acknowledgment messages from the ExchangeAgent
        (OrderAccepted, OrderExecuted, OrderCancelled, …).

        Currently logs the message type and takes no further action.
        Extend here to route execution confirmations back to originating
        MARL agents or incorporate fill information into the reward signal.
        """
        super().receive_message(current_time, sender_id, message)

        logger.debug(
            "[StopSignalAgent %d] %s from agent %d at %s",
            self.id,
            type(message).__name__,
            sender_id,
            fmt_ts(current_time),
        )

        # TODO: route order execution messages to the originating MARL agent.
        # Pattern: message.order.agent_id → agent index → agent callback.

    # -----------------------------------------------------------------------
    # Order submission
    # -----------------------------------------------------------------------

    def _submit_orders(
        self,
        pending_orders: List[Tuple[int, Optional[Order]]],
        current_time: NanosecondTime,
    ) -> None:
        """
        Submit non-None orders to the ExchangeAgent.

        Iterates *pending_orders* in FIFO registration order (the order agents
        were passed to __init__), which determines submission priority at the
        exchange for same-tick same-price-level conflicts between agents.
        """
        for agent_idx, order in pending_orders:
            if order is None:
                continue  # Hold / no-op — nothing to submit.

            msg = self._wrap_order(order)
            if msg is None:
                logger.warning(
                    "[StopSignalAgent %d] Tick %d: unrecognised order type "
                    "'%s' from agent_idx=%d — skipped.",
                    self.id,
                    self._tick_count,
                    type(order).__name__,
                    agent_idx,
                )
                continue

            self.send_message(self.exchange_id, msg)
            self._last_tick_orders.append((agent_idx, order))

            logger.debug(
                "[StopSignalAgent %d] Submitted %s from agent_idx=%d",
                self.id,
                order,
                agent_idx,
            )

    @staticmethod
    def _wrap_order(order: Order) -> Optional[Any]:
        """
        Wrap an Order in the correct ABIDES message dataclass.

        Supports LimitOrder and MarketOrder.  Returns None for unrecognised
        order types so the caller can log and skip gracefully.
        """
        if isinstance(order, LimitOrder):
            return LimitOrderMsg(order)
        if isinstance(order, MarketOrder):
            return MarketOrderMsg(order)
        return None

    # -----------------------------------------------------------------------
    # Stubbed MARL agent interaction helpers
    #
    # Each helper wraps one method of the MARLAgentProtocol interface.
    # Replace the TODO lines with real calls once agents are implemented.
    # The agent_idx parameter is available for per-agent diagnostics/logging.
    # -----------------------------------------------------------------------

    def _get_observation(self, agent_idx: int, agent: Any) -> Any:
        """
        Retrieve the current observation from *agent*.

        Stub: returns None until MARL agents are implemented.
        Real call: return agent.get_observation()
        """
        # TODO: return agent.get_observation()
        return None

    def _compute_reward(self, agent_idx: int, agent: Any) -> float:
        """
        Retrieve the reward for *agent*'s action in the previous tick.

        Called before _get_observation() so reward reflects consequences of
        the last action before the next state is presented.

        Stub: returns 0.0 until MARL agents are implemented.
        Real call: return agent.compute_reward()
        """
        # TODO: return agent.compute_reward()
        return 0.0

    def _check_done(self, agent_idx: int, agent: Any) -> bool:
        """
        Check whether *agent*'s episode is terminal.

        Stub: returns False until MARL agents are implemented.
        Real call: return agent.check_done()
        """
        # TODO: return agent.check_done()
        return False

    def _get_info(self, agent_idx: int, agent: Any) -> Dict[str, Any]:
        """
        Retrieve auxiliary diagnostic data from *agent*.

        Stub: returns {} until MARL agents are implemented.
        Real call: return agent.get_info()
        """
        # TODO: return agent.get_info()
        return {}

    def _compute_action(
        self,
        agent_idx: int,
        agent: Any,
        observation: Any,
    ) -> Optional[int]:
        """
        Run *agent*'s DQN policy on *observation* and return a discrete action
        index.

        Stub: returns None (no-op) until MARL agents are implemented.
        Real call: return agent.compute_action(observation)
        """
        # TODO: return agent.compute_action(observation)
        return None

    def _action_to_order(
        self,
        agent_idx: int,
        agent: Any,
        action: Optional[int],
        current_time: NanosecondTime,
    ) -> Optional[Order]:
        """
        Translate *agent*'s discrete action index to an ABIDES Order.

        The MARL agent owns the action-space definition and performs the
        translation; StopSignalAgent only routes the resulting Order.

        Stub: returns None (hold) until MARL agents are implemented.
        Real call: return agent.action_to_order(action, current_time)
        """
        # TODO: return agent.action_to_order(action, current_time)
        return None

    def _notify_step_result(
        self,
        agent_idx: int,
        agent: Any,
        observation: Any,
        action: Optional[int],
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        """
        Deliver the completed (s, a, r, done, info) transition to *agent*
        after all tick orders have been submitted.

        Stub: no-op until MARL agents are implemented.
        Real call: agent.notify_step_result(observation, action, reward, done, info)
        """
        # TODO: agent.notify_step_result(observation, action, reward, done, info)
        pass

    # -----------------------------------------------------------------------
    # Read-only diagnostics / properties
    # -----------------------------------------------------------------------

    @property
    def num_agents(self) -> int:
        """Number of MARL agents registered with this synchronizer."""
        return len(self.marl_agents)

    @property
    def tick_count(self) -> int:
        """Number of ticks fully processed so far."""
        return self._tick_count
