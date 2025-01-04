from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, noise=0.05, n_features=1, random_state=1234)
df = pd.DataFrame(dict(x=X[:, 0], y=y))
print(df.head())

print("데이터의 개수 :", len(df))
print(df["x"].head())
print(df["y"].head())

plt.scatter(df["x"], df["y"], s=100, c=y)
plt.xlabel("$X$")
plt.ylabel("$X_2$")
plt.show()