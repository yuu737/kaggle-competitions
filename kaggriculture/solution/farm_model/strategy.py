"""Farm-level capital strategy, separate from legal actions and job scheduling.

The first strategy experiment changes reinvestment tempo only. Production type
still competes on whole-farm season cash, demand, own price impact, rival supply,
capacity and funding. It does not force an animal or copy a published action tape.
"""


def phase(state):
    final_day = state.settings.last_action_step // state.settings.turns_per_day
    if state.day == final_day:
        return "liquidate"
    if state.day >= final_day - 4:
        return "realize"
    if state.day <= 4:
        return "bootstrap"
    return "compound"


def investment_due(state, policy, last_plan_day, last_plan_step):
    if policy.investment_strategy == "daily":
        return last_plan_day != state.day
    if policy.investment_strategy != "compound":
        raise ValueError("Unknown investment strategy")
    if state.day == state.settings.last_action_step // state.settings.turns_per_day:
        return False
    # Daily reset reopens the budget; within a day, deploy idle cash again after
    # the previous project finishes. A quarter-day interval bounds repeat work.
    interval = max(1, state.settings.turns_per_day // 4)
    return last_plan_day != state.day or state.step - last_plan_step >= interval
