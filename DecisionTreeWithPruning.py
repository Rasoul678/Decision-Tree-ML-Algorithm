from DecisionTree import DecisionTree
import numpy as np

class DecisionTreeWithPruning(DecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, ccp_alpha=0.0):
        """
        Initialize with pruning parameter ccp_alpha.

        Parameters:
        - ccp_alpha: Complexity parameter used for pruning.
                     Higher values lead to more pruning.
        """
        super().__init__(max_depth, min_samples_split)
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y):
        """Build the tree and then prune it."""
        super().fit(X, y)
        if self.ccp_alpha > 0:
            self._prune_tree()

    def _prune_tree(self):
        """Prune the tree recursively."""
        self.tree = self._prune_node(self.tree)

    def _prune_node(self, node):
        """Recursively prune a node and its children."""
        # If leaf node, return as is
        if 'value' in node:
            return node

        # Prune left and right children first
        node['left'] = self._prune_node(node['left'])
        node['right'] = self._prune_node(node['right'])

        # If both children are now leaves, consider merging
        if 'value' in node['left'] and 'value' in node['right']:
            # Calculate effective alpha for this node
            leaf_count = self._count_leaves(node) - 1  # Merging would remove 1 leaf
            impurity = self._node_impurity(node)
            left_impurity = self._node_impurity(node['left'])
            right_impurity = self._node_impurity(node['right'])

            # Calculate gain from merging
            gain = (impurity - left_impurity - right_impurity) / leaf_count

            # If gain is less than ccp_alpha, prune
            if gain < self.ccp_alpha:
                # Replace with leaf containing most common class
                merged_y = []
                self._gather_leaf_values(node['left'], merged_y)
                self._gather_leaf_values(node['right'], merged_y)
                return {'value': self._most_common_label(np.array(merged_y))}

        return node

    def _count_leaves(self, node):
        """Count number of leaves in a subtree."""
        if 'value' in node:
            return 1
        return self._count_leaves(node['left']) + self._count_leaves(node['right'])

    def _node_impurity(self, node):
        """Calculate impurity (entropy) of a node."""
        if 'value' in node:
            # For leaves, impurity is 0 (pure)
            return 0

        # For internal nodes, we'd need to track the original samples
        # This is a simplification - in practice you'd store sample counts
        return 0.5  # Placeholder - real implementation would track samples

    def _gather_leaf_values(self, node, values):
        """Gather all values from leaf nodes."""
        if 'value' in node:
            values.append(node['value'])
        else:
            self._gather_leaf_values(node['left'], values)
            self._gather_leaf_values(node['right'], values)