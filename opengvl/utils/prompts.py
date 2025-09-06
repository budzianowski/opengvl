def get_prompt(instruction: str) -> str:
    """Get the prompt for the model."""
    return f"""You are an expert roboticist tasked to predict task completion percentages for frames of a robot for the task of {instruction}. The task completion percentages are between 0 and 100, where 100 corresponds to full task completion. We provide several examples of the robot performing the task at various stages and their corresponding task completion percentages. Note that these frames are in random order, so please pay attention to the individual frames when reasoning about task completion percentage.\n"""


# We provide several examples of the robot performing the task at various stages and their corresponding task completion percentages.
