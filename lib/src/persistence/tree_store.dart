import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';

/// Interface for storing and retrieving Decision Tree Classifiers
///
/// This interface provides methods for persisting DecisionTreeClassifier
/// instances to a storage backend (e.g., Sembast database).
abstract class TreeStore {
  /// Saves a DecisionTreeClassifier to the store
  ///
  /// Returns the tree ID that can be used to retrieve the tree later.
  /// If [treeId] is provided, uses that ID; otherwise generates a new one.
  ///
  /// Parameters:
  /// - [tree] The DecisionTreeClassifier to save
  /// - [treeId] Optional custom tree ID. If not provided, generates one.
  ///
  /// Returns the tree ID (String)
  Future<String> saveTree(DecisionTreeClassifier tree, {String? treeId});

  /// Loads a DecisionTreeClassifier from the store
  ///
  /// Parameters:
  /// - [treeId] The ID of the tree to load
  ///
  /// Returns the loaded DecisionTreeClassifier, or null if not found
  Future<DecisionTreeClassifier?> loadTree(String treeId);

  /// Deletes a DecisionTreeClassifier from the store
  ///
  /// Parameters:
  /// - [treeId] The ID of the tree to delete
  Future<void> deleteTree(String treeId);

  /// Lists all tree IDs in the store
  ///
  /// Returns a list of tree IDs
  Future<List<String>> listTrees();

  /// Gets metadata for a tree without loading the full tree
  ///
  /// Parameters:
  /// - [treeId] The ID of the tree
  ///
  /// Returns metadata map, or null if tree not found
  Future<Map<String, dynamic>?> getTreeMetadata(String treeId);
}

