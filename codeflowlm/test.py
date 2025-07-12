from codeflowlm.data import get_changes_from_features, ord_cross_changes_full
from codeflowlm.data import path
from codeflowlm.train import execute_command
from codeflowlm.global_training_pool import get_global_training_pool
import pandas as pd
import pickle
import os

def prepare_cumulative_test_data(project):
  global_training_pool = get_global_training_pool()
  df = pd.DataFrame(global_training_pool)

  if 'first_fix_date' in df.columns and 'fixes' in df.columns:
    df = df.drop(columns=['first_fix_date', 'fixes'])

  commits = []
  labels = []
  commit_messages = []
  codes = []
  
  for _, row in df.iterrows():
    commit_id = row['commit_hash']
    idx = ord_cross_changes_full[0].index(commit_id)
    label = row['is_buggy_commit']
    commit_message = ord_cross_changes_full[2][idx]
    code = ord_cross_changes_full[3][idx]
    commits.append(commit_id)
    labels.append(label)
    commit_messages.append(commit_message)
    codes.append(code)

  assert len(commits) == len(labels) == len(commit_messages) == len(codes) == df.shape[0]

  changes = (commits, labels, commit_messages, codes)

  with open(f"{path}/changes_online_{project}.pkl", "wb") as f:
    pickle.dump(changes, f)

  with open(f"{path}/features_online_{project}.pkl", "wb") as f:
    pickle.dump(df, f)

  return f"{path}/changes_online_{project}.pkl", f"{path}/features_online_{project}.pkl"


def test(project, features_test, model_path, th, pretrained_model, 
         calculate_metrics=True, peft_alg="lora", eval_metric='f1'):
  changes_test = get_changes_from_features(features_test, do_test=True)
  with open(f"{path}/changes_test_online_{project}.pkl", "wb") as f:
    pickle.dump(changes_test, f)

  with open(f"{path}/features_test_online_{project}.pkl", "wb") as f:
    pickle.dump(features_test, f)

  print(f"Testing with recent data with th = {th}...")
  command = f"""
  python PEFT4CC/just-in-time/run_{peft_alg}.py \
   --test_data_file {path}/changes_test_online_{project}.pkl {path}/features_test_online_{project}.pkl \
   --output_dir {model_path} \
   --pretrained_model {pretrained_model} \
   --batch_size 16 \
   --do_test \
   --threshold {th} \
   --eval_metric {eval_metric} \
   """

  if peft_alg == "lora":
    command += "--use_lora "

  if pretrained_model == 'codet5p-770m':
    command += "--hidden_size 1024 "

  if calculate_metrics:
    command += """--calculate_metrics """

  execute_command(command)

  results = None

  if os.path.exists("results.pkl"):
    with open(f"results.pkl", "rb") as f:
      results = pickle.load(f)

    os.remove("results.pkl")

  with open(f"predictions.pkl", "rb") as f:
    predictions = pickle.load(f)

  os.remove("predictions.pkl")

  return results, predictions