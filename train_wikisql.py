# Databricks notebook source
# MAGIC %md
# MAGIC ## Train Dolly
# MAGIC
# MAGIC This fine-tunes the [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) model on
# MAGIC the [wikisql](https://huggingface.co/datasets/wikisql) dataset.
# MAGIC
# MAGIC ```
# MAGIC   No licence
# MAGIC ```
# MAGIC
# MAGIC Please note that while GPT-J 6B is [Apache 2.0 licensed](https://huggingface.co/EleutherAI/gpt-j-6B),
# MAGIC the wikisql dataset has unknown license (https://huggingface.co/datasets/wikisql).

# COMMAND ----------
# MAGIC %md
# MAGIC ## Installation:
# MAGIC 
# MAGIC Please run the Makefile by `make` to install the dependencies.
 
# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

# COMMAND ----------

import os
from datetime import datetime
from training.trainer import load_training_dataset, load_tokenizer

#dbutils.widgets.text("num_gpus", "", "num_gpus")
#dbutils.widgets.text("local_training_root", "", "local_training_root")
#dbutils.widgets.text("dbfs_output_root", "", "dbfs_output_root")

# COMMAND ----------

# Cache data and tokenizer locally before creating a bunch of deepspeed processes and make sure they succeeds.
load_training_dataset()
load_tokenizer()

# COMMAND ----------

timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
model_name = "dolly"
checkpoint_dir_name = f"{model_name}__{timestamp}"

root_path = os.getcwd()
deepspeed_config = os.path.join(root_path, "config/ds_z3_bf16_config.json")

model_training_dir_name = "gpt-j-6b_wikisql_training"

# Use the local training root path if it was provided.  Otherwise try to find a sensible default.
local_training_root = None #dbutils.widgets.get("local_training_root")
if not local_training_root:
    # Use preferred path when working in a Databricks cluster if it exists.
    if os.path.exists("/local_disk0"):
        local_training_root = os.path.join("/local_disk0", model_training_dir_name)
    # Otherwise use the home directory.
    else:
        local_training_root = os.path.join(os.path.expanduser('~'), model_training_dir_name)

dbfs_output_root = None #dbutils.widgets.get("dbfs_output_root")
if not dbfs_output_root:
    dbfs_output_root = f"/dbfs/{model_training_dir_name}"

os.makedirs(local_training_root, exist_ok=True)
os.makedirs(dbfs_output_root, exist_ok=True)

local_output_dir = os.path.join(local_training_root, checkpoint_dir_name)
dbfs_output_dir = os.path.join(dbfs_output_root, checkpoint_dir_name)

num_gpus_flag = ""
num_gpus = 1 #dbutils.widgets.get("num_gpus")
if num_gpus:
    num_gpus = int(num_gpus)
    num_gpus_flag = f"--num_gpus={num_gpus}"

tensorboard_display_dir = f"{local_output_dir}/runs"

print(f"Local Output Dir: {local_output_dir}")
print(f"DBFS Output Dir: {dbfs_output_dir}")
print(f"Tensorboard Display Dir: {tensorboard_display_dir}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir '{tensorboard_display_dir}'

# COMMAND ----------

# MAGIC !deepspeed {num_gpus_flag} \
# MAGIC     --module training.trainer \
# MAGIC     --deepspeed {deepspeed_config} \
# MAGIC     --epochs 1 \
# MAGIC     --local-output-dir {local_output_dir} \
# MAGIC     --dbfs-output-dir {dbfs_output_dir} \
# MAGIC     --per-device-train-batch-size 8 \
# MAGIC     --per-device-eval-batch-size 8 \
# MAGIC     --lr 1e-5

# COMMAND ----------

from training.generate import generate_response, load_model_tokenizer_for_generate

model, tokenizer = load_model_tokenizer_for_generate(local_output_dir)

# COMMAND ----------

# Examples from https://huggingface.co/datasets/wikisql
instructions = [
    {'question': "What is terrence ross' nationality",
        'headers': ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team" ]
    }
]

# Use the model to generate responses for each of the instructions above.
for instruction in instructions:
    response = generate_sql_response(instruction, model=model, tokenizer=tokenizer)
    if response:
        print(f"Instruction: {instruction}\n\n{response}\n\n-----------\n")