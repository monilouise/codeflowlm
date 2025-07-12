import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import math


def analyze_results():
  df = pd.read_excel("results.xlsx")

  y_true = list(df["true_label"])
  y_pred = list(df["pred_prob"])
  auc = np.round(roc_auc_score(y_true, y_pred), 3)
  print("Auc for our sample data is {}".format(auc))

  qtd_zeros = len(df.loc[df["true_label"] == 0])
  qtd_ones = len(df.loc[df["true_label"] == 1])
  print("total 0s: ",qtd_zeros," total 1s:",qtd_ones, " imbalance ratio:", qtd_ones/qtd_zeros )

  max_gm = 0
  op_th = 0
  max_rec0 = 0
  max_rec1 = 0

  for idx, row in df.iterrows():
      pred = float(row["pred_prob"])

      rec_0 = len(df.loc[(df["true_label"] == 0) & (df["pred_prob"] <= pred)])
      rec_0 = rec_0/qtd_zeros

      rec_1 = len(df.loc[(df["true_label"] == 1) & (df["pred_prob"] > pred)])
      rec_1 = rec_1/qtd_ones

      gm = math.sqrt(rec_1*rec_0)
      if(gm > max_gm):
          max_gm = gm
          op_th = pred
          max_rec0 = rec_0
          max_rec1 = rec_1

          num = len(df.loc[(df["true_label"] == 1) & (df["pred_prob"] > pred)])
          den = len(df.loc[(df["pred_prob"] > pred)])
          prec1 = num/den

          num = len(df.loc[(df["true_label"] == 0) & (df["pred_prob"] <= pred)])
          den = len(df.loc[(df["pred_prob"] <= pred)])
          prec0 = num/den


  print(max_gm, max_rec0, max_rec1, prec1, op_th)
  f1_1 = 2 * ((prec1 * max_rec1)/(prec1 + max_rec1))
  f1_0 = 2 * ((prec0 * max_rec0)/(prec0 + max_rec1))

  print("f1 class 0: ",f1_0, " f1 class 1:", f1_1)

  return op_th


def calculate_th_from_test(predictions, target_th=0.4):
  probs = predictions['pred_prob']
  quantile = np.quantile(probs, target_th)
  return quantile