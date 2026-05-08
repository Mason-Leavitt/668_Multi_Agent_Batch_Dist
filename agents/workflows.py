SUPPORTED_WORKFLOWS = {
    "solve_D_given_W0_x0_xDavg": {
        "label": "Target average distillate composition",
        "problem_type": "solve_D_given_W0_x0_xDavg",
        "description": (
            "Use this workflow when you know the initial charge, the initial "
            "ethanol mole fraction, and the average distillate composition you want."
        ),
        "required_inputs": ["W0", "x0", "xDavg_target"],
        "outputs": ["D", "B", "xB", "xDavg"],
        "example_user_prompt": (
            "I have 1000 mol of ethanol-water at 5 mol% ethanol. "
            "I want the average distillate to be 20 mol% ethanol. "
            "How much distillate can I collect?"
        ),
    },
    "solve_batch_given_W0_x0_xB": {
        "label": "Target final still composition",
        "problem_type": "solve_batch_given_W0_x0_xB",
        "description": (
            "Use this workflow when you know the initial charge, the initial "
            "ethanol mole fraction, and the final still composition you want to reach."
        ),
        "required_inputs": ["W0", "x0", "xB"],
        "outputs": ["D", "B", "xB", "xDavg"],
        "example_user_prompt": (
            "I start with 1000 mol of ethanol-water at 5 mol% ethanol and distill "
            "until the still is 1 mol% ethanol. How much distillate do I collect?"
        ),
    },
    "check_batch_consistency": {
        "label": "Check an existing proposed result",
        "problem_type": "check_batch_consistency",
        "description": (
            "Use this workflow when you already have a proposed batch result and "
            "want to check total balance, component balance, and Rayleigh consistency."
        ),
        "required_inputs": ["W0", "B", "D", "x0", "xB", "xDavg"],
        "outputs": [
            "is_total_balance_consistent",
            "is_component_balance_consistent",
            "is_rayleigh_consistent",
            "is_fully_consistent",
        ],
        "example_user_prompt": (
            "Check whether this batch result is consistent: W0=1000 mol, "
            "B=763.986 mol, D=236.014 mol, x0=0.05, xB=0.003661, and xDavg=0.20."
        ),
    },
}

VARIABLE_DESCRIPTIONS = {
    "W0": "initial charge amount in the still",
    "D": "distillate amount collected",
    "B": "amount remaining in the still",
    "x0": "initial ethanol mole fraction in the feed",
    "xB": "final ethanol mole fraction in the still",
    "xDavg_target": "target average ethanol mole fraction in the distillate",
    "xDavg": "average ethanol mole fraction in the distillate",
}
