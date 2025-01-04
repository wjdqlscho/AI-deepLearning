from sklearn.model_selection import train_test_split

X = [
    [0, 1, 2, 3],
    [2, 3, 1, 4],
    [5, 2, 3, 5],
    [3, 5, 2, 1],
    [7, 5, 3, 5]
]
Y = [0, 0, 1, 2, 0]

# 데이터(X)만 입력한 경우
X_train, X_test = train_test_split(X, test_size=0.4, random_state=7777)
print(f"X_train: {X_train}")
print(f"X_test: {X_test}")