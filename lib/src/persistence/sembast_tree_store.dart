import 'dart:math';

import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/persistence/tree_store.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/helpers/from_tree_assessor_type_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/helpers/to_tree_assessor_type_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/dtype_to_json.dart';
import 'package:ml_linalg/from_dtype_json.dart';
import 'package:sembast/sembast.dart';

/// Sembast implementation of TreeStore
///
/// Stores DecisionTreeClassifier instances in a Sembast database.
class SembastTreeStore implements TreeStore {
  /// The Sembast database instance
  final Database database;

  /// Store name for trees
  static const String treesStoreName = 'decision_trees';

  /// Store reference
  final _treesStore = stringMapStoreFactory.store(treesStoreName);

  /// Creates a SembastTreeStore instance
  ///
  /// Parameters:
  /// - [database] The Sembast database instance to use
  SembastTreeStore({required this.database});

  @override
  Future<String> saveTree(DecisionTreeClassifier tree, {String? treeId}) async {
    final id = treeId ?? _generateTreeId();

    // Cast to implementation to access treeRootNode
    final treeImpl = tree as DecisionTreeClassifierImpl;

    // Extract metadata
    final metadata = _extractMetadata(treeImpl);

    // Serialize tree root node
    final rootNodeJson = treeImpl.treeRootNode.toJson();

    // Store in Sembast
    await _treesStore.record(id).put(database, {
      'id': id,
      'type': 'DecisionTreeClassifier',
      'metadata': metadata,
      'root': rootNodeJson,
    });

    return id;
  }

  @override
  Future<DecisionTreeClassifier?> loadTree(String treeId) async {
    final record = await _treesStore.record(treeId).get(database);

    if (record == null) {
      return null;
    }

    // Deserialize tree structure
    final rootNodeJson = record['root'] as Map<String, dynamic>;
    final rootNode = TreeNode.fromJson(rootNodeJson);

    // Get metadata
    final metadata = record['metadata'] as Map<String, dynamic>;

    // Reconstruct DecisionTreeClassifier
    return DecisionTreeClassifierImpl(
      metadata['min_error'] as num,
      metadata['min_samples'] as int,
      metadata['max_depth'] as int,
      rootNode,
      metadata['target_column'] as String,
      _parseAssessorType(metadata['assessor_type'] as String),
      _parseDType(metadata['dtype'] as String),
    );
  }

  @override
  Future<void> deleteTree(String treeId) async {
    await _treesStore.record(treeId).delete(database);
  }

  @override
  Future<List<String>> listTrees() async {
    final records = await _treesStore.find(database);
    return records.map((record) => record.key).toList();
  }

  @override
  Future<Map<String, dynamic>?> getTreeMetadata(String treeId) async {
    final record = await _treesStore.record(treeId).get(database);

    if (record == null) {
      return null;
    }

    return record['metadata'] as Map<String, dynamic>?;
  }

  /// Generates a unique tree ID
  String _generateTreeId() {
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final random = Random().nextInt(10000);
    return 'tree_${timestamp}_$random';
  }

  /// Extracts metadata from a DecisionTreeClassifier
  Map<String, dynamic> _extractMetadata(DecisionTreeClassifierImpl tree) {
    return {
      'target_column': tree.targetColumnName,
      'min_error': tree.minError,
      'min_samples': tree.minSamplesCount,
      'max_depth': tree.maxDepth,
      'assessor_type': toTreeAssessorTypeJson(tree.assessorType),
      'dtype': dTypeToJson(tree.dtype),
      'created_at': DateTime.now().toIso8601String(),
      'node_count': _countNodes(tree.treeRootNode),
      'depth': _calculateDepth(tree.treeRootNode),
      'schema_version': tree.schemaVersion,
    };
  }

  /// Counts nodes in a tree
  int _countNodes(TreeNode node) {
    int count = 1;
    if (node.children != null) {
      for (final child in node.children!) {
        count += _countNodes(child);
      }
    }
    return count;
  }

  /// Calculates the depth of a tree
  int _calculateDepth(TreeNode node) {
    if (node.isLeaf) {
      return 0;
    }

    if (node.children == null || node.children!.isEmpty) {
      return 0;
    }

    int maxChildDepth = 0;
    for (final child in node.children!) {
      final childDepth = _calculateDepth(child);
      if (childDepth > maxChildDepth) {
        maxChildDepth = childDepth;
      }
    }

    return maxChildDepth + 1;
  }

  /// Parses assessor type from JSON string
  TreeAssessorType _parseAssessorType(String json) {
    return fromTreeAssessorTypeJson(json);
  }

  /// Parses DType from JSON string
  DType _parseDType(String json) {
    final dtype = fromDTypeJson(json);
    if (dtype == null) {
      throw ArgumentError('Unknown dtype: $json');
    }
    return dtype;
  }
}
