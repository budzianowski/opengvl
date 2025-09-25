def get_prompt(instruction: str) -> str:
    """Default prompt for temporal progress estimation (inline template)."""
    return (
        "You are an expert roboticist tasked to predict task completion percentages for frames of a robot "
        f"for the task of {instruction}. The task completion percentages are between 0 and 100, where 100 "
        "corresponds to full task completion. The frames may be in random order; reason about each frame "
        "independently when estimating completion.\n"
    )


def format_prompt(template: str, *, instruction: str) -> str:
    """Format a prompt template with required placeholders.

    Required placeholders:
    - {instruction}
    """
    return template.format(instruction=instruction)
