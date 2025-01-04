from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.05, random_state=1234)
df = pd.DataFrame(dict(X=X[:, 0], y=X[:, 1], label=y))
print(df.head())

print("데이터의 개수 :", len(df))
print(df["X"].head())
print(df["y"].head())

plt.scatter(df["X"], df["y"], s=100, c=y)
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.show()