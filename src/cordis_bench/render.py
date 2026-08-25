RULES = """Dynamic-composition semantics:
- The shared state is x=(x0,...), with all arithmetic modulo m.
- A component is enabled when desired by the orchestrator and active when its effect is currently installed.
- An inactive enabled component may activate when all of its required keys are provided by active enabled components.
- An active component must deactivate when it is disabled or when a required provider is disabled/missing.
- A provider cannot deactivate while an active dependent still requires one of its keys, so dependents unload first.
- Activating a component applies its effect operations from top to bottom.
- Deactivating applies the exact inverse operations in reverse order.
- If several lifecycle actions are legal, the runtime may choose any of them.
"""


def v1_rules(theory):
    if theory.undo_order == "lifo":
        undo = "On deactivation, replace each primitive by its inverse and apply those inverses bottom-to-top (LIFO)."
    else:
        undo = "On deactivation, replace each primitive by its inverse and apply those inverses top-to-bottom (FIFO)."
    return f"""Dynamic-composition semantics:
- The shared state is x=(x0,...), with all arithmetic modulo m.
- `enable(C)` and `disable(C)` are external orchestration directives: they change only whether C is enabled.
- A component is active exactly while its effect is installed.
- Support is recursive: an enabled component is supported only if every required provider is itself supported; enabled components with no requirements are supported.
- An inactive supported component may activate when all of its required providers are active.
- An active unsupported component must deactivate.
- A provider cannot deactivate while an active dependent still requires one of its keys; dependents unload first.
- Activating a component applies its listed effect operations top-to-bottom.
- {undo}
- If several lifecycle actions are legal, the runtime may choose any of them.
"""


def render_world(world):
    lines = [f"Modulus m = {world.modulus}", f"State width = {world.width}", "Components:"]
    for component in world.components:
        requires = ", ".join(sorted(component.requires)) or "none"
        provides = ", ".join(sorted(component.provides)) or "none"
        lines.append(f"  {component.name}: requires {{{requires}}}; provides {{{provides}}}")
        for op in component.effect:
            lines.append(f"    {op.render()}")
    return "\n".join(lines)


def render_state(state):
    return (
        f"x={state.values}; enabled={{{', '.join(sorted(state.enabled))}}}; "
        f"active={{{', '.join(sorted(state.active))}}}"
    )


def render_trace_task(world, start, trace, choices):
    actions = "\n".join(f"  {i + 1}. {action.render()}" for i, action in enumerate(trace))
    options = "\n".join(f"  {letter}. {value}" for letter, value in choices.items())
    return f"""{RULES}
{render_world(world)}

Initial runtime state:
{render_state(start)}

The runtime takes this exact lifecycle trace:
{actions}

What is the final shared state x?
{options}

Answer with only the option letter."""


def render_confluence_task(world, start):
    return f"""{RULES}
{render_world(world)}

Initial runtime state:
{render_state(start)}

Let the runtime take lifecycle steps until quiescence. Is the quiescent observable outcome (shared state x and active component set) the same for every legal lifecycle schedule?
  A. YES
  B. NO

Answer with only A or B."""


def render_intervention_task(world, start, target_values, options):
    choices = "\n".join(f"  {letter}. Disable component {name}" for letter, name in options.items())
    return f"""{RULES}
{render_world(world)}

The system is currently quiescent:
{render_state(start)}

Choose one intervention. After that intervention, the runtime may take lifecycle steps in any legal order until quiescence. Which intervention guarantees that every legal schedule ends with shared state x={target_values}?
{choices}

Answer with only the option letter."""


def render_dynamic_trace_task(world, start, trace, choices, theory):
    actions = "\n".join(f"  {i + 1}. {action.render()}" for i, action in enumerate(trace))
    options = "\n".join(f"  {letter}. {value}" for letter, value in choices.items())
    return f"""{v1_rules(theory)}
{render_world(world)}

The system starts quiescent:
{render_state(start)}

The following exact mixed orchestration/lifecycle trace occurs. External enable/disable directives happen exactly where shown; lifecycle actions shown are the scheduler choices:
{actions}

What is the final shared state x?
{options}

Answer with only the option letter."""


def render_dynamic_confluence_task(world, start, directive, theory):
    return f"""{v1_rules(theory)}
{render_world(world)}

The system starts quiescent:
{render_state(start)}

The orchestrator now executes exactly one directive:
  {directive.render()}

The runtime then takes lifecycle steps until quiescence. Is the final observable outcome (shared state x and active component set) the same for every legal lifecycle schedule?
  A. YES
  B. NO

Answer with only A or B."""


def render_plan_task(world, start, target_values, options, theory):
    choices = "\n".join(
        f"  {letter}. disable({first}); settle to quiescence; disable({second}); settle to quiescence"
        for letter, (first, second) in options.items()
    )
    return f"""{v1_rules(theory)}
{render_world(world)}

The system starts quiescent:
{render_state(start)}

Choose one two-directive plan. After each disable directive, the runtime may take lifecycle actions in any legal order until quiescence before the next directive is issued. Which plan guarantees that every legal schedule ends with shared state x={target_values}?
{choices}

Answer with only the option letter."""
