from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

#PR, PR_AUCを計算する関数
def plot_prc(y_test, y_prob):

  #PR曲線のprecision,recallの計算
  pr_precision, pr_recall, thresholds = precision_recall_curve(y_test, y_prob)
  auprc = auc(pr_recall, pr_precision)
  #PR曲線のプロットを保存
  plt.figure(figsize=(8, 8))
  plt.axes().set_aspect("equal", "datalim")
  plt.plot(pr_recall, pr_precision, label="PR CURVE (Area = %.3f)"%auprc, lw=4, color="orangered")
  plt.fill_between(pr_recall, pr_precision, color="tab:orange", alpha=0.5)
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title("Precision Recall CURVE")
  plt.legend(loc="lower right")
  plt.grid()
  plt.savefig("pr_curve.png", bbox_inches="tight")
  plt.gca().clear()

  return auprc