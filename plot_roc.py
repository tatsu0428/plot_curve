from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#ROC曲線をプロットし，ROC_AUCを計算する関数
def plot_roc(y_test, y_prob):

  fpr, tpr, thresholds = roc_curve(y_test, y_prob, drop_intermediate=False)
  auroc = auc(fpr, tpr)
  #ROC曲線のプロットを保存
  plt.figure(figsize=(8, 8))
  plt.axes().set_aspect("equal", "datalim")
  plt.plot([0, 1], [0, 1], linestyle="--", lw=3, label="Random ROC CURVE (Area = %.3f)"%0.5, color="k")
  plt.plot(fpr, tpr, label="ROC CURVE (Area = %.3f)"%auroc, lw=4, color="orangered")
  plt.fill_between(fpr, tpr, color="tab:orange", alpha=0.5)
  plt.xlabel("FPR: False Positive Rate")
  plt.ylabel("TPR: True Positive Rate")
  plt.title("ROC CURVE")
  plt.legend(loc="lower right")
  plt.grid()
  plt.savefig("roc_curve.png", bbox_inches="tight")
  plt.gca().clear()

  return auroc