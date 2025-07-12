import pickle
import pandas as pd

with open("/content/drive/MyDrive/UFPE/Tese/Datasets/jitfine/ord_cross_features_train.pkl", "rb") as f:
    ord_cross_features_train = pickle.load(f)

with open("/content/drive/MyDrive/UFPE/Tese/Datasets/jitfine/ord_cross_features_valid.pkl", "rb") as f:
    ord_cross_features_valid = pickle.load(f)

with open("/content/drive/MyDrive/UFPE/Tese/Datasets/jitfine/ord_cross_features_test.pkl", "rb") as f:
    ord_cross_features_test = pickle.load(f)

df_features_full = pd.concat([ord_cross_features_train, ord_cross_features_valid, ord_cross_features_test], ignore_index=True)

with open("/content/drive/MyDrive/UFPE/Tese/Datasets/jitfine/ord_cross_changes_train_lst.pkl", "rb") as f:
    ord_cross_changes_train = pickle.load(f)

with open("/content/drive/MyDrive/UFPE/Tese/Datasets/jitfine/ord_cross_changes_valid_lst.pkl", "rb") as f:
    ord_cross_changes_valid = pickle.load(f)

with open("/content/drive/MyDrive/UFPE/Tese/Datasets/jitfine/ord_cross_changes_test_lst.pkl", "rb") as f:
    ord_cross_changes_test = pickle.load(f)
    
ord_cross_changes_full = [ord_cross_changes_train[0] + ord_cross_changes_valid[0] + ord_cross_changes_test[0], ord_cross_changes_train[1] + ord_cross_changes_valid[1] + ord_cross_changes_test[1],
                          ord_cross_changes_train[2] + ord_cross_changes_valid[2] + ord_cross_changes_test[2], ord_cross_changes_train[3] + ord_cross_changes_valid[3] + ord_cross_changes_test[3]]


def get_changes_from_features(df_features, do_test=True):
  commits = []
  labels = []
  commit_messages = []
  codes = []

  for _, row in df_features.iterrows():
      commit_id = row['commit_hash']
      idx = ord_cross_changes_full[0].index(commit_id)
      label = row['is_buggy_commit']
      commit_message = ord_cross_changes_full[2][idx]
      code = ord_cross_changes_full[3][idx]
      commits.append(commit_id)
      labels.append(label)
      commit_messages.append(commit_message)
      codes.append(code)
      if do_test:
        if row['is_buggy_commit'] != ord_cross_changes_full[1][idx]:
          print("row['is_buggy_commit'] = ", row['is_buggy_commit'])
          print("ord_cross_changes_full[1][idx] = ", ord_cross_changes_full[1][idx])
        assert row['is_buggy_commit'] == ord_cross_changes_full[1][idx]

  assert len(commits) == len(labels) == len(commit_messages) == len(codes) == df_features.shape[0]
  changes = (commits, labels, commit_messages, codes)
  return changes


