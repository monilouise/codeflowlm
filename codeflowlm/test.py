from codeflowlm.data import get_changes_from_features
from codeflowlm.train import execute_command
import pickle
import os

def test(batch_classifier_dir, path, full_changes_train_file, full_changed_valid_file, full_changes_test_file, project, features_test, model_path, th, pretrained_model, 
         calculate_metrics=True, peft_alg="lora", eval_metric='f1', batch_size=16, stream_changes_file=None, stream_features_file=None, adjust_th=False):
  changes_test = get_changes_from_features(full_changes_train_file, full_changed_valid_file, full_changes_test_file, features_test, do_test=True)
  with open(f"{path}/changes_test_online_{project}.pkl", "wb") as f:
    pickle.dump(changes_test, f)

  with open(f"{path}/features_test_online_{project}.pkl", "wb") as f:
    pickle.dump(features_test, f)

  print(f"Testing with recent data with th = {th}...")

  print("PEFT algorithm: ", peft_alg)
  
  if peft_alg == "lora":
    command = get_lora_command(batch_classifier_dir, path, project, model_path, th, pretrained_model, eval_metric, 
                               batch_size, stream_changes_file=stream_changes_file, stream_features_file=stream_features_file)
    command += "--use_lora "
  else:
    command = get_pret_command(batch_classifier_dir, path, project, model_path, th, pretrained_model, eval_metric, batch_size, 
                               stream_changes_file=stream_changes_file, stream_features_file=stream_features_file)
  if pretrained_model == 'codet5p-770m':
    command += "--hidden_size 1024 "

  if calculate_metrics:
    command += """--calculate_metrics """
    
  if adjust_th:
    command += """--update_threshold """

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

def get_pret_command(batch_classifier_dir, path, project, model_path, th,pretrained_model, eval_metric, batch_size, stream_changes_file=None, stream_features_file=None):
    result = f"""
  python {batch_classifier_dir}PEFT4CC/just-in-time/run_peft.py \
    --pretrained_model {pretrained_model} \
    --method prefix \
    --structure concat \
    --test_data_file {path}/changes_test_online_{project}.pkl {path}/features_test_online_{project}.pkl """

    if stream_changes_file is not None and stream_features_file is not None:
        result += f"""--stream_data_file {stream_changes_file} {stream_features_file} """
    
    result += f"""
    --output_dir {model_path} \
    --batch_size {batch_size} \
    --do_test \
    --threshold {th} \
    --eval_metric {eval_metric} \
    """
    return result

def get_lora_command(batch_classifier_dir, path, project, model_path, th, pretrained_model, eval_metric, batch_size, stream_changes_file=None, stream_features_file=None):
    result = f"""
  python {batch_classifier_dir}PEFT4CC/just-in-time/run_lora.py \
   --test_data_file {path}/changes_test_online_{project}.pkl {path}/features_test_online_{project}.pkl """
    
    if stream_changes_file is not None and stream_features_file is not None:
        result += f"""--stream_data_file {stream_changes_file} {stream_features_file} """

    result += f"""
    --output_dir {model_path} \
    --pretrained_model {pretrained_model} \
    --batch_size {batch_size} \
    --do_test \
    --threshold {th} \
    --eval_metric {eval_metric} \
    """

    return result