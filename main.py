from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from DecisionTree import DecisionTree
from DecisionTreeWithPruning import DecisionTreeWithPruning

def main():
    print("Decision Tree with Python")

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Without pruning
    tree_no_prune = DecisionTree(max_depth=3)
    tree_no_prune.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, tree_no_prune.predict(X_train))
    test_acc = accuracy_score(y_test, tree_no_prune.predict(X_test))
    print(f"No pruning - Train: {train_acc:.3f}, Test: {test_acc:.3f}")

    # With pruning
    tree_prune = DecisionTreeWithPruning(ccp_alpha=0.01)
    tree_prune.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, tree_prune.predict(X_train))
    test_acc = accuracy_score(y_test, tree_prune.predict(X_test))
    print(f"With pruning - Train: {train_acc:.3f}, Test: {test_acc:.3f}")

    # Compare tree sizes
    def count_nodes(tree):
        if 'value' in tree:
            return 1
        return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])

    print(f"Nodes without pruning: {count_nodes(tree_no_prune.tree)}")
    print(f"Nodes with pruning: {count_nodes(tree_prune.tree)}")

    # Cross-Validation:
    # Example of finding optimal ccp_alpha
    alphas = [0, 0.001, 0.01, 0.1]
    for alpha in alphas:
        tree = DecisionTreeWithPruning(ccp_alpha=alpha)
        tree.fit(X_train, y_train)
        acc = accuracy_score(y_test, tree.predict(X_test))
        print(f"Alpha={alpha:.3f}, Accuracy={acc:.3f}, Nodes={count_nodes(tree.tree)}")

if __name__ == "__main__":
    main()
