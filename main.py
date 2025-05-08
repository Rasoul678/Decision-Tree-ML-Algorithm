from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from DecisionTree import DecisionTree

def main():
    print("Decision Tree with Python")

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the tree
    tree = DecisionTree(max_depth=3)
    tree.fit(X_train, y_train)

    # Make predictions
    predictions = tree.predict(X_test)
    print(predictions)
    print(y_test)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
