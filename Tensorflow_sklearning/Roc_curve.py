from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 정의
y_prob = np.asarray([
    0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
])
y_real = np.asarray([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 1, 0, 0, 1, 1, 1, 1
])

# ROC AUC 점수 계산
score = roc_auc_score(y_real, y_prob)
print(f"AUROC score: {score:.6f}")

# Confusion Matrix로 TPR, FPR 계산
def get_tpr_fpr(y_real, y_pred):
    cm = confusion_matrix(y_real, y_pred)
    TN, FP, FN, TP = cm.ravel()
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FPR = FP / (TN + FP) if (TN + FP) > 0 else 0.0
    return TPR, FPR

threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)
TPR, FPR = get_tpr_fpr(y_real, y_pred)
print(f"TPR: {TPR * 100:.4f}%")
print(f"FPR: {FPR * 100:.4f}%")

# N개의 포인트로 ROC Curve 생성
def get_n_points(y_real, y_prob, n):
    sorted_pred = np.unique(np.sort(y_prob))
    n = min(len(sorted_pred), n)
    step = len(sorted_pred) / n

    indices = [int(i * step) for i in range(n)]
    points = [(0.0, 0.0)]
    for index in indices:
        threshold = sorted_pred[index]
        y_pred = (y_prob >= threshold).astype(int)
        TPR, FPR = get_tpr_fpr(y_real, y_pred)
        points.append((FPR, TPR))
    if points[-1] != (1.0, 1.0):
        points.append((1.0, 1.0))
    return points

points = get_n_points(y_real, y_prob, 30)
print(points)

# ROC Curve 시각화
def plot_roc_curve(points):
    points = sorted(points)
    plt.plot([point[0] for point in points], [point[1] for point in points], color="red")
    sns.lineplot(x=[0, 1], y=[0, 1], color="green")
    plt.title("ROC Curve")
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.show()

plot_roc_curve(points)

# AUROC 계산
def get_auroc(points):
    points = sorted(points)
    area = 0
    cur_x, cur_y = 0.0, 0.0
    for x, y in points:
        if cur_x != x:
            area += (x - cur_x) * cur_y
            cur_x, cur_y = x, y
    return area

area = get_auroc(points)
print(f"AUROC score: {area:.6f}")
