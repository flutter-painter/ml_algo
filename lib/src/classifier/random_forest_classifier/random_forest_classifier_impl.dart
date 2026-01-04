import 'dart:math';

import 'package:ml_algo/src/classifier/_mixins/assessable_classifier_mixin.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/decision_tree_classifier_impl.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/voting_strategy.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/persistence/sembast_tree_store.dart';
import 'package:ml_algo/src/persistence/tree_store.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/helpers/from_tree_assessor_type_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/helpers/to_tree_assessor_type_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/voting_strategy_json_converter.dart';
import 'package:ml_linalg/dtype_to_json.dart';
import 'package:ml_linalg/from_dtype_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

// TODO: Add JSON serialization after core functionality works
// part 'random_forest_classifier_impl.g.dart';

// @JsonSerializable()
// @DTypeJsonConverter()

/// Result of bootstrap sampling with tracking
class _BootstrapResult {
  final DataFrame sampledData;
  final Set<int> usedIndices;

  _BootstrapResult(this.sampledData, this.usedIndices);
}

class RandomForestClassifierImpl
    with AssessableClassifierMixin, SerializableMixin
    implements RandomForestClassifier {
  RandomForestClassifierImpl({
    required this.nEstimators,
    required this.bootstrap,
    required this.sampleSize,
    required this.maxFeatures,
    required this.votingStrategy,
    DataFrame? trainData,
    required this.targetName,
    required this.minError,
    required this.minSamplesCount,
    required this.maxDepth,
    required this.assessorType,
    required this.dtype,
    this.treeStore,
    int? seed,
    dynamic classWeight,
    bool balancedBootstrap = false,
    this.schemaVersion = 1,
    bool train = true,
  })  : trainData = trainData ?? DataFrame([]),
        _seed = seed,
        _classWeight = classWeight,
        _balancedBootstrap = balancedBootstrap {
    if (train && trainData != null) {
      _train();
    }
  }

  final int? _seed;

  factory RandomForestClassifierImpl.fromJson(Map<String, dynamic> json) {
    // TODO: Implement JSON deserialization
    throw UnimplementedError('JSON deserialization not yet implemented');
  }

  @override
  Map<String, dynamic> toJson() {
    // TODO: Implement JSON serialization
    throw UnimplementedError('JSON serialization not yet implemented');
  }

  @override
  final int nEstimators;

  @override
  final bool bootstrap;

  @override
  final double sampleSize;

  @override
  final dynamic maxFeatures;

  @override
  final VotingStrategy votingStrategy;

  @override
  List<String>? get treeIds => _treeIds;

  List<String>? _treeIds;

  final DataFrame trainData;

  final String targetName;

  @override
  final num minError;

  @override
  final int minSamplesCount;

  @override
  final int maxDepth;

  @override
  final TreeAssessorType assessorType;

  @override
  final DType dtype;

  final TreeStore? treeStore;

  @override
  final int schemaVersion;

  final dynamic _classWeight;
  final bool _balancedBootstrap;

  // Class weights cache (calculated from _classWeight)
  Map<num, double>? _calculatedClassWeights;

  // In-memory trees (always kept in memory for fast predictions)
  List<DecisionTreeClassifier>? _trees;

  // OOB tracking: for each sample index, track which trees didn't use it
  Map<int, List<int>>? _oobSamples; // sample index -> list of tree indices that didn't use it
  double? _oobScore;

  // Feature importance cache
  Map<String, double>? _featureImportance;

  @override
  Iterable<String> get targetNames => [targetName];

  @override
  num get positiveLabel => double.nan;

  @override
  num get negativeLabel => double.nan;

  @override
  double? get oobScore => _oobScore;

  @override
  Map<String, double> get featureImportance {
    if (_featureImportance == null && _trees != null) {
      _featureImportance = _calculateFeatureImportance();
    }
    return _featureImportance ?? {};
  }

  /// Trains the random forest
  void _train() {
    final random = _seed != null ? Random(_seed) : Random();
    final treeIdsList = <String>[];
    final trees = <DecisionTreeClassifier>[];

    // Calculate class weights if needed
    if (_classWeight != null) {
      _calculatedClassWeights = _calculateClassWeights();
    }

    // Initialize OOB tracking if bootstrap is enabled
    if (bootstrap) {
      _oobSamples = {};
      for (int i = 0; i < trainData.rows.length; i++) {
        _oobSamples![i] = [];
      }
    }

    for (int treeIndex = 0; treeIndex < nEstimators; treeIndex++) {
      // Create bootstrap sample and track which samples were used
      final bootstrapResult = bootstrap
          ? (_balancedBootstrap
              ? _balancedBootstrapSampleWithTracking(trainData, sampleSize, random)
              : _bootstrapSampleWithTracking(trainData, sampleSize, random, _calculatedClassWeights))
          : _BootstrapResult(trainData, <int>{});

      // Train tree
      final tree = DecisionTreeClassifier(
        bootstrapResult.sampledData,
        targetName,
        minError: minError,
        minSamplesCount: minSamplesCount,
        maxDepth: maxDepth,
        assessorType: assessorType,
        dtype: dtype,
      );

      trees.add(tree);

      // Track OOB samples (samples not in this bootstrap)
      if (bootstrap && _oobSamples != null) {
        for (int sampleIndex = 0; sampleIndex < trainData.rows.length; sampleIndex++) {
          if (!bootstrapResult.usedIndices.contains(sampleIndex)) {
            _oobSamples![sampleIndex]!.add(treeIndex);
          }
        }
      }

      // Save to store if provided (synchronously for now)
      if (treeStore != null) {
        // Save tree and collect ID
        // Using synchronous approach: keep trees in memory and save to store
        // This is a limitation - ideally this would be async
        final treeId = _saveTreeSync(tree, treeIndex);
        treeIdsList.add(treeId);
      }
    }

    _trees = trees;
    _treeIds = treeStore != null ? treeIdsList : null;

    // Calculate OOB score if bootstrap is enabled
    if (bootstrap && _oobSamples != null) {
      _oobScore = _calculateOobScore();
    }

    // Calculate feature importance
    _featureImportance = _calculateFeatureImportance();
  }

  /// Calculates class weights based on classWeight parameter
  Map<num, double> _calculateClassWeights() {
    if (_classWeight == null) {
      return {};
    }

    // Get target column index
    final headerList = trainData.header.toList();
    final targetIndex = headerList.indexOf(targetName);
    if (targetIndex == -1) {
      return {};
    }

    // Count class frequencies
    final classCounts = <num, int>{};
    for (final row in trainData.rows) {
      final classLabel = row.elementAt(targetIndex) as num;
      classCounts.update(classLabel, (count) => count + 1, ifAbsent: () => 1);
    }

    if (classCounts.isEmpty) {
      return {};
    }

    // Handle 'balanced' mode
    if (_classWeight == 'balanced') {
      final nSamples = trainData.rows.length;
      final nClasses = classCounts.length;
      final weights = <num, double>{};

      for (final entry in classCounts.entries) {
        final classLabel = entry.key;
        final classCount = entry.value;
        // weight = n_samples / (n_classes * class_frequency)
        weights[classLabel] = nSamples / (nClasses * classCount);
      }

      return weights;
    }

    // Handle manual Map<num, double>
    if (_classWeight is Map) {
      final manualWeights = _classWeight as Map<num, double>;
      // Validate that all classes have weights (or use default 1.0)
      final weights = <num, double>{};
      for (final classLabel in classCounts.keys) {
        weights[classLabel] = manualWeights[classLabel] ?? 1.0;
      }
      return weights;
    }

    return {};
  }

  /// Creates bootstrap sample and tracks which original indices were used
  /// Optionally uses class weights for weighted sampling
  _BootstrapResult _bootstrapSampleWithTracking(
      DataFrame data, double sampleSize, Random random,
      [Map<num, double>? classWeights]) {
    final targetSize = (data.rows.length * sampleSize).round();
    final sampledRows = <Iterable<num>>[];
    final usedIndices = <int>{};

    // Get target column index
    final headerList = data.header.toList();
    final targetIndex = headerList.indexOf(targetName);

    if (classWeights != null && classWeights.isNotEmpty && targetIndex != -1) {
      // Weighted bootstrap sampling
      // Group samples by class
      final samplesByClass = <num, List<int>>{};
      for (int i = 0; i < data.rows.length; i++) {
        final row = data.rows.elementAt(i);
        final classLabel = row.elementAt(targetIndex) as num;
        samplesByClass.putIfAbsent(classLabel, () => []).add(i);
      }

      // Sample with weights
      for (int i = 0; i < targetSize; i++) {
        // Select class based on weights
        final selectedClass = _selectClassByWeight(classWeights, random);
        final classSamples = samplesByClass[selectedClass];
        if (classSamples != null && classSamples.isNotEmpty) {
          final randomIndex = classSamples[random.nextInt(classSamples.length)];
          usedIndices.add(randomIndex);
          final row = data.rows.elementAt(randomIndex);
          sampledRows.add(row.map((v) => v as num).toList());
        }
      }
    } else {
      // Standard bootstrap sampling
      for (int i = 0; i < targetSize; i++) {
        final randomIndex = random.nextInt(data.rows.length);
        usedIndices.add(randomIndex);
        final row = data.rows.elementAt(randomIndex);
        // Convert to List<num> to ensure type safety
        sampledRows.add(row.map((v) => v as num).toList());
      }
    }

    return _BootstrapResult(
      DataFrame(sampledRows, header: data.header),
      usedIndices,
    );
  }

  /// Selects a class based on weights using weighted random selection
  num _selectClassByWeight(Map<num, double> weights, Random random) {
    final totalWeight = weights.values.fold(0.0, (a, b) => a + b);
    final randomValue = random.nextDouble() * totalWeight;
    
    double cumulativeWeight = 0.0;
    for (final entry in weights.entries) {
      cumulativeWeight += entry.value;
      if (randomValue <= cumulativeWeight) {
        return entry.key;
      }
    }
    
    // Fallback to last class (shouldn't happen)
    return weights.keys.last;
  }

  /// Creates balanced bootstrap sample (equal samples per class)
  _BootstrapResult _balancedBootstrapSampleWithTracking(
      DataFrame data, double sampleSize, Random random) {
    // Get target column index
    final headerList = data.header.toList();
    final targetIndex = headerList.indexOf(targetName);
    if (targetIndex == -1) {
      // Fallback to standard bootstrap
      return _bootstrapSampleWithTracking(data, sampleSize, random);
    }

    // Group samples by class
    final samplesByClass = <num, List<int>>{};
    for (int i = 0; i < data.rows.length; i++) {
      final row = data.rows.elementAt(i);
      final classLabel = row.elementAt(targetIndex) as num;
      samplesByClass.putIfAbsent(classLabel, () => []).add(i);
    }

    if (samplesByClass.isEmpty) {
      return _BootstrapResult(data, <int>{});
    }

    // Calculate samples per class
    final nClasses = samplesByClass.length;
    final samplesPerClass = (sampleSize * data.rows.length / nClasses).round();
    final sampledRows = <Iterable<num>>[];
    final usedIndices = <int>{};

    // Sample equal amounts from each class
    for (final classSamples in samplesByClass.values) {
      for (int i = 0; i < samplesPerClass; i++) {
        if (classSamples.isNotEmpty) {
          final randomIndex = classSamples[random.nextInt(classSamples.length)];
          usedIndices.add(randomIndex);
          final row = data.rows.elementAt(randomIndex);
          sampledRows.add(row.map((v) => v as num).toList());
        }
      }
    }

    return _BootstrapResult(
      DataFrame(sampledRows, header: data.header),
      usedIndices,
    );
  }

  /// Calculates Out-Of-Bag score
  double _calculateOobScore() {
    if (_oobSamples == null || _trees == null) {
      return 0.0;
    }

    int correctPredictions = 0;
    int totalOobSamples = 0;

    // Get target column index
    final headerList = trainData.header.toList();
    final targetIndex = headerList.indexOf(targetName);
    if (targetIndex == -1) {
      return 0.0;
    }

    // For each sample that has OOB predictions
    for (final entry in _oobSamples!.entries) {
      final sampleIndex = entry.key;
      final treeIndices = entry.value;

      if (treeIndices.isEmpty) {
        continue; // Sample was in all bootstraps, skip
      }

      totalOobSamples++;

      // Get actual label
      final actualLabel = trainData.rows.elementAt(sampleIndex).elementAt(targetIndex) as num;

      // Get predictions from trees that didn't use this sample
      final oobPredictions = <num>[];
      for (final treeIdx in treeIndices) {
        // Create DataFrame with single sample
        final sampleRow = trainData.rows.elementAt(sampleIndex);
        final sampleData = DataFrame(
          [sampleRow.map((v) => v as num).toList()],
          header: trainData.header,
        );

        // Predict with this tree
        final prediction = _trees![treeIdx].predict(sampleData);
        if (prediction.rows.isNotEmpty) {
          final predictedLabel = prediction.rows.first.first as num;
          oobPredictions.add(predictedLabel);
        }
      }

      // Skip if no OOB predictions available
      if (oobPredictions.isEmpty) {
        continue;
      }

      // Majority vote on OOB predictions
      final predictedLabel = _majorityVote(oobPredictions);

      // Check if correct
      if (predictedLabel == actualLabel) {
        correctPredictions++;
      }
    }

    if (totalOobSamples == 0) {
      return 0.0; // No OOB samples
    }

    return correctPredictions / totalOobSamples;
  }

  /// Calculates feature importance based on how often features are used in splits
  Map<String, double> _calculateFeatureImportance() {
    if (_trees == null || _trees!.isEmpty) {
      return {};
    }

    // Count feature usage across all trees
    final featureUsageCounts = <int, int>{};
    
    // Get feature column indices (excluding target)
    final headerList = trainData.header.toList();
    final targetIndex = headerList.indexOf(targetName);
    final featureIndices = <int>[];
    for (int i = 0; i < headerList.length; i++) {
      if (i != targetIndex) {
        featureIndices.add(i);
      }
    }

    // Traverse each tree and count feature usage
    for (final tree in _trees!) {
      final treeImpl = tree as DecisionTreeClassifierImpl;
      _countFeatureUsage(treeImpl.treeRootNode, featureUsageCounts);
    }

    // Convert to feature names and normalize
    final importance = <String, double>{};
    final totalUsage = featureUsageCounts.values.fold(0, (a, b) => a + b);

    if (totalUsage == 0) {
      // If no features were used, assign equal importance
      for (final idx in featureIndices) {
        if (idx < headerList.length) {
          importance[headerList[idx]] = 1.0 / featureIndices.length;
        }
      }
    } else {
      // Normalize by total usage
      for (final entry in featureUsageCounts.entries) {
        final featureIndex = entry.key;
        final usageCount = entry.value;
        if (featureIndex < headerList.length) {
          final featureName = headerList[featureIndex];
          importance[featureName] = usageCount / totalUsage;
        }
      }

      // Ensure all features have an entry (even if 0 usage)
      for (final idx in featureIndices) {
        if (idx < headerList.length) {
          final featureName = headerList[idx];
          importance.putIfAbsent(featureName, () => 0.0);
        }
      }
    }

    return importance;
  }

  /// Recursively counts feature usage in tree nodes
  void _countFeatureUsage(TreeNode node, Map<int, int> featureUsageCounts) {
    // If this node has a split, count the feature
    if (node.splitIndex != null && !node.isLeaf) {
      featureUsageCounts.update(
        node.splitIndex!,
        (count) => count + 1,
        ifAbsent: () => 1,
      );
    }

    // Recursively process children
    if (node.children != null) {
      for (final child in node.children!) {
        _countFeatureUsage(child, featureUsageCounts);
      }
    }
  }

  /// Synchronously saves a tree to TreeStore (workaround for async limitation)
  String _saveTreeSync(DecisionTreeClassifier tree, int treeIndex) {
    // Since saveToStore is async but we need synchronous training,
    // we'll use a workaround: generate ID first, then save with that ID
    // This ensures treeIds is populated immediately
    // Note: The actual save happens asynchronously, but we return the ID immediately
    final treeId = 'tree_${DateTime.now().millisecondsSinceEpoch}_$treeIndex';
    // Fire and forget - save happens in background
    tree.saveToStore(treeStore!, treeId: treeId);
    return treeId;
  }


  @override
  DataFrame predict(DataFrame features) {
    if (_trees == null) {
      throw StateError('Forest not trained. Trees not available.');
    }

    return _predictSync(features, _trees!);
  }

  DataFrame _predictSync(
      DataFrame features, List<DecisionTreeClassifier> trees) {
    final predictions = <List<num>>[];

    // Get predictions from all trees
    for (final tree in trees) {
      final treePrediction = tree.predict(features);
      final rowValues = <num>[];
      for (final row in treePrediction.rows) {
        rowValues.add(row.first as num);
      }
      predictions.add(rowValues);
    }

    // Aggregate predictions
    return _aggregatePredictions(predictions, votingStrategy);
  }

  @override
  DataFrame predictProbabilities(DataFrame features) {
    if (_trees == null) {
      throw StateError('Forest not trained. Trees not available.');
    }

    return _predictProbabilitiesSync(features, _trees!);
  }

  DataFrame _predictProbabilitiesSync(
      DataFrame features, List<DecisionTreeClassifier> trees) {
    final probabilities = <List<List<num>>>[];

    // Get probability distributions from all trees
    for (final tree in trees) {
      final treeProbs = tree.predictProbabilities(features);
      final probRows = <List<num>>[];
      for (final row in treeProbs.rows) {
        probRows.add(row.map((v) => v as num).toList());
      }
      probabilities.add(probRows);
    }

    // Aggregate probabilities
    return _aggregateProbabilities(probabilities, votingStrategy);
  }

  /// Async version of predict that loads trees from store if needed
  ///
  /// This method will:
  /// 1. Use in-memory trees if available (fast path)
  /// 2. Load trees from treeStore if _trees is null but treeIds exist
  /// 3. Throw error if trees are not available
  Future<DataFrame> predictAsync(DataFrame features) async {
    List<DecisionTreeClassifier> trees;

    // Fast path: use in-memory trees if available
    if (_trees != null) {
      trees = _trees!;
    } else if (treeStore != null && _treeIds != null && _treeIds!.isNotEmpty) {
      // Load trees from store
      trees = await _loadTreesFromStore();
      if (trees.isEmpty) {
        throw StateError(
            'No trees could be loaded from store. Expected ${_treeIds!.length} trees, but loaded ${trees.length}.');
      }
      if (trees.length != _treeIds!.length) {
        throw StateError(
            'Not all trees could be loaded from store. Expected ${_treeIds!.length} trees, but loaded ${trees.length}.');
      }
      // Cache loaded trees for future use
      _trees = trees;
    } else {
      throw StateError(
          'Forest not trained. Trees not available in memory or store.');
    }

    return _predictSync(features, trees);
  }

  /// Async version of predictProbabilities that loads trees from store if needed
  ///
  /// This method will:
  /// 1. Use in-memory trees if available (fast path)
  /// 2. Load trees from treeStore if _trees is null but treeIds exist
  /// 3. Throw error if trees are not available
  Future<DataFrame> predictProbabilitiesAsync(DataFrame features) async {
    List<DecisionTreeClassifier> trees;

    // Fast path: use in-memory trees if available
    if (_trees != null) {
      trees = _trees!;
    } else if (treeStore != null && _treeIds != null && _treeIds!.isNotEmpty) {
      // Load trees from store
      trees = await _loadTreesFromStore();
      if (trees.isEmpty) {
        throw StateError(
            'No trees could be loaded from store. Expected ${_treeIds!.length} trees, but loaded ${trees.length}.');
      }
      if (trees.length != _treeIds!.length) {
        throw StateError(
            'Not all trees could be loaded from store. Expected ${_treeIds!.length} trees, but loaded ${trees.length}.');
      }
      // Cache loaded trees for future use
      _trees = trees;
    } else {
      throw StateError(
          'Forest not trained. Trees not available in memory or store.');
    }

    return _predictProbabilitiesSync(features, trees);
  }

  /// Loads trees from treeStore using stored treeIds
  Future<List<DecisionTreeClassifier>> _loadTreesFromStore() async {
    if (treeStore == null || _treeIds == null || _treeIds!.isEmpty) {
      return [];
    }

    final trees = <DecisionTreeClassifier>[];
    for (final treeId in _treeIds!) {
      final tree = await treeStore!.loadTree(treeId);
      if (tree != null) {
        trees.add(tree);
      }
    }

    return trees;
  }

  DataFrame _aggregatePredictions(
    List<List<num>> predictions,
    VotingStrategy strategy,
  ) {
    if (predictions.isEmpty) {
      return DataFrame([<num>[]], header: targetNames.toList());
    }

    final nSamples = predictions.first.length;
    final aggregated = <num>[];

    for (int i = 0; i < nSamples; i++) {
      final samplePredictions = predictions.map((p) => p[i]).toList();
      aggregated.add(_majorityVote(samplePredictions));
    }

    // Create DataFrame with same structure as DecisionTreeClassifier
    // Each row is one sample, column is the target name
    final outcomeVector = Vector.fromList(aggregated, dtype: dtype);
    return DataFrame.fromMatrix(
      Matrix.fromColumns([outcomeVector], dtype: dtype),
      header: targetNames.toList(),
    );
  }

  num _majorityVote(List<num> votes) {
    final counts = <num, int>{};
    for (final vote in votes) {
      counts[vote] = (counts[vote] ?? 0) + 1;
    }

    num? majorityClass;
    int maxCount = 0;
    for (final entry in counts.entries) {
      if (entry.value > maxCount) {
        maxCount = entry.value;
        majorityClass = entry.key;
      }
    }

    return majorityClass!;
  }

  DataFrame _aggregateProbabilities(
    List<List<List<num>>> probabilities,
    VotingStrategy strategy,
  ) {
    if (probabilities.isEmpty) {
      return DataFrame([<num>[]], header: targetNames.toList());
    }

    final nSamples = probabilities.first.length;
    final aggregated = <List<num>>[];

    for (int i = 0; i < nSamples; i++) {
      final sampleProbs = probabilities.map((p) => p[i]).toList();
      final avgProbs = <num>[];

      // Average probabilities across trees
      final nClasses = sampleProbs.first.length;
      for (int j = 0; j < nClasses; j++) {
        double sum = 0.0;
        for (final probs in sampleProbs) {
          sum += probs[j].toDouble();
        }
        avgProbs.add(sum / sampleProbs.length);
      }

      aggregated.add(avgProbs);
    }

    // Create DataFrame with same structure as DecisionTreeClassifier
    // Each row is one sample, columns are class probabilities
    final probVectors = aggregated
        .map((probs) => Vector.fromList(probs, dtype: dtype))
        .toList();
    return DataFrame.fromMatrix(
      Matrix.fromRows(probVectors, dtype: dtype),
      header: targetNames.toList(),
    );
  }

  @override
  RandomForestClassifier retrain(DataFrame data) {
    // TODO: Implement retraining
    throw UnimplementedError('Retraining not yet implemented');
  }

  @override
  Future<String> saveToStore(TreeStore store, {String? forestId}) async {
    if (store is! SembastTreeStore) {
      throw ArgumentError(
          'saveToStore requires SembastTreeStore, got ${store.runtimeType}');
    }

    final sembastStore = store;
    final id = forestId ?? sembastStore.generateForestId();

    // Prepare forest metadata
    final metadata = <String, dynamic>{
      'n_estimators': nEstimators,
      'bootstrap': bootstrap,
      'sample_size': sampleSize,
      'max_features': maxFeatures?.toString() ?? 'null',
      'voting_strategy': _votingStrategyToJson(votingStrategy),
      'target_name': targetName,
      'min_error': minError,
      'min_samples_count': minSamplesCount,
      'max_depth': maxDepth,
      'assessor_type': toTreeAssessorTypeJson(assessorType),
      'dtype': dTypeToJson(dtype),
      'tree_ids': _treeIds ?? [],
      'schema_version': schemaVersion,
      'created_at': DateTime.now().toIso8601String(),
      'seed': _seed,
      'oob_score': _oobScore,
      'feature_importance': _featureImportance ?? {},
    };

    await sembastStore.saveForest(id, metadata);
    return id;
  }

  static Future<RandomForestClassifier?> loadFromStore(
    TreeStore store,
    String forestId,
  ) async {
    if (store is! SembastTreeStore) {
      throw ArgumentError(
          'loadFromStore requires SembastTreeStore, got ${store.runtimeType}');
    }

    final sembastStore = store;
    final metadata = await sembastStore.loadForest(forestId);

    if (metadata == null) {
      return null;
    }

    // Parse voting strategy
    final votingStrategyStr = metadata['voting_strategy'] as String;
    final votingStrategy = _votingStrategyFromJson(votingStrategyStr);

    // Parse maxFeatures
    dynamic maxFeatures;
    final maxFeaturesStr = metadata['max_features'] as String;
    if (maxFeaturesStr == 'null') {
      maxFeatures = null;
    } else if (maxFeaturesStr == 'sqrt') {
      maxFeatures = 'sqrt';
    } else if (maxFeaturesStr == 'log2') {
      maxFeatures = 'log2';
    } else {
      // Try to parse as number
      final parsed = num.tryParse(maxFeaturesStr);
      maxFeatures = parsed ?? maxFeaturesStr;
    }

    // Parse treeIds
    final treeIdsList = (metadata['tree_ids'] as List<dynamic>?)
            ?.map((e) => e as String)
            .toList() ??
        <String>[];

    // Reconstruct forest
    return RandomForestClassifierImpl(
      nEstimators: metadata['n_estimators'] as int,
      bootstrap: metadata['bootstrap'] as bool,
      sampleSize: (metadata['sample_size'] as num).toDouble(),
      maxFeatures: maxFeatures,
      votingStrategy: votingStrategy,
      targetName: metadata['target_name'] as String,
      minError: metadata['min_error'] as num,
      minSamplesCount: metadata['min_samples_count'] as int,
      maxDepth: metadata['max_depth'] as int,
      assessorType: fromTreeAssessorTypeJson(
          metadata['assessor_type'] as String),
      dtype: _parseDTypeFromJson(metadata['dtype'] as String),
      treeStore: store,
      seed: metadata['seed'] as int?,
      schemaVersion: metadata['schema_version'] as int? ?? 1,
      train: false, // Don't retrain, just load
    ).._restoreFromMetadata(metadata, treeIdsList);
  }

  /// Restores internal state from loaded metadata
  void _restoreFromMetadata(
      Map<String, dynamic> metadata, List<String> treeIdsList) {
    _treeIds = treeIdsList.isNotEmpty ? treeIdsList : null;
    _oobScore = metadata['oob_score'] as double?;
    _featureImportance =
        (metadata['feature_importance'] as Map<String, dynamic>?)
                ?.map((key, value) => MapEntry(key, (value as num).toDouble())) ??
            {};
  }

  /// Converts VotingStrategy to JSON string
  static String _votingStrategyToJson(VotingStrategy strategy) {
    final converter = VotingStrategyJsonConverter();
    return converter.toJson(strategy);
  }

  /// Converts JSON string to VotingStrategy
  static VotingStrategy _votingStrategyFromJson(String json) {
    final converter = VotingStrategyJsonConverter();
    return converter.fromJson(json);
  }

  /// Parses DType from JSON string
  static DType _parseDTypeFromJson(String json) {
    final dtype = fromDTypeJson(json);
    if (dtype == null) {
      throw ArgumentError('Unknown dtype: $json');
    }
    return dtype;
  }
}
