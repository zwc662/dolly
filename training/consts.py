DEFAULT_TRAINING_DATASET = "tatsu-lab/alpaca"
WIKISQL_TRAINING_DATASET = "wikisql"
DEFAULT_INPUT_MODEL = "EleutherAI/gpt-j-6B"
END_KEY = "### End"
INSTRUCTION_KEY = "### Instruction:"
HEADER_KEY = "### Headers: "
RESPONSE_KEY_NL = f"### Response:\n"
WIKISQL_RESPONSE_KEY_NL = f'human_readable'
DEFAULT_SEED = 42

# The format of the instruction the model has been trained on.
PROMPT_FORMAT = """%s

%s
{instruction}

%s""" % (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    INSTRUCTION_KEY,
    RESPONSE_KEY_NL,
)
