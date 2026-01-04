import 'dart:io';
import 'dart:math';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier.dart';
import 'package:ml_algo/src/persistence/sembast_tree_store.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:sembast/sembast_io.dart';

/// Example demonstrating Random Forest with Sembast persistence
///
/// This example shows:
/// 1. Training a Random Forest with tree persistence
/// 2. Saving the forest to database
/// 3. Loading the forest back
/// 4. Making predictions with loaded trees
Future<void> main() async {
  print('=== Random Forest with Sembast Persistence Example ===\n');

  // Create a temporary database for this example
  final tempDir = Directory.systemTemp;
  final dbPath = '${tempDir.path}/rf_example_${DateTime.now().millisecondsSinceEpoch}.db';
  final database = await databaseFactoryIo.openDatabase(dbPath);
  final store = SembastTreeStore(database: database);

  try {
    // Step 1: Generate synthetic training data
    print('Step 1: Generating training data...');
    final trainData = _generateTrainingData();
    print('   Generated ${trainData.rows.length} training samples');
    print('   Features: ${trainData.header.where((h) => h != 'target').join(', ')}');
    print('   Target: target\n');

    // Step 2: Train Random Forest with treeStore
    print('Step 2: Training Random Forest with tree persistence...');
    final forest = RandomForestClassifier(
      trainData,
      'target',
      nEstimators: 10, // Small number for demo
      maxDepth: 4,
      treeStore: store, // Trees will be saved to database
      dtype: DType.float32,
      seed: 42, // For reproducibility
    );

    print('   Trained ${forest.nEstimators} trees');
    print('   Tree IDs: ${forest.treeIds?.take(3).join(', ')}... (${forest.treeIds?.length} total)');
    
    // Wait for tree saves to complete
    await Future<void>.delayed(Duration(milliseconds: 200));
    print('   ✓ Trees saved to database\n');

    // Step 3: Verify trees are in database
    print('Step 3: Verifying trees in database...');
    int treesFound = 0;
    for (final treeId in forest.treeIds!) {
      final tree = await store.loadTree(treeId);
      if (tree != null) treesFound++;
    }
    print('   Found $treesFound/${forest.treeIds!.length} trees in database\n');

    // Step 4: Save the entire forest
    print('Step 4: Saving forest configuration...');
    final forestId = await forest.saveToStore(store);
    print('   Forest saved with ID: $forestId\n');

    // Step 5: Load the forest back
    print('Step 5: Loading forest from database...');
    final loadedForest = await RandomForestClassifier.loadFromStore(
      store,
      forestId,
    );

    if (loadedForest == null) {
      print('   ✗ Failed to load forest!');
      return;
    }

    print('   ✓ Forest loaded successfully');
    print('   Configuration:');
    print('     - nEstimators: ${loadedForest.nEstimators}');
    print('     - maxDepth: ${loadedForest.maxDepth}');
    print('     - treeIds: ${loadedForest.treeIds?.length} trees\n');

    // Step 6: Make predictions with loaded forest
    print('Step 6: Making predictions with loaded forest...');
    final testData = _generateTestData();
    print('   Test samples: ${testData.rows.length}');

    // Use async predict - trees will be loaded from database automatically
    final predictions = await loadedForest.predictAsync(testData);
    
    print('   Predictions:');
    for (int i = 0; i < testData.rows.length; i++) {
      final features = testData.rows.elementAt(i);
      final prediction = predictions.rows.elementAt(i).first;
      print('     Sample ${i + 1}: [${features.take(3).map((v) => v.toStringAsFixed(1)).join(', ')}...] -> Class $prediction');
    }
    print('');

    // Step 7: Get probability predictions
    print('Step 7: Getting probability predictions...');
    final probabilities = await loadedForest.predictProbabilitiesAsync(testData);
    print('   Probabilities for first sample:');
    final firstProbs = probabilities.rows.first.toList();
    for (int i = 0; i < firstProbs.length; i++) {
      final prob = (firstProbs[i] as num).toDouble() * 100;
      print('     Class $i: ${prob.toStringAsFixed(1)}%');
    }
    print('');

    // Step 8: Show feature importance
    print('Step 8: Feature importance:');
    final importance = loadedForest.featureImportance;
    final sortedFeatures = importance.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    
    for (final entry in sortedFeatures.take(5)) {
      print('   ${entry.key}: ${(entry.value * 100).toStringAsFixed(1)}%');
    }
    print('');

    print('=== Example completed successfully! ===');
    print('\nNote: Database file: $dbPath');
    print('You can inspect it or delete it manually.');

  } finally {
    // Cleanup: close database
    await database.close();
  }
}

/// Generates synthetic training data for classification
DataFrame _generateTrainingData() {
  final random = Random(42); // Fixed seed for reproducibility
  final data = <List<num>>[];

  // Generate 200 samples with 5 features
  for (int i = 0; i < 200; i++) {
    final feature1 = random.nextDouble() * 10;
    final feature2 = random.nextDouble() * 10;
    final feature3 = random.nextDouble() * 10;
    final feature4 = random.nextDouble() * 10;
    final feature5 = random.nextDouble() * 10;

    // Simple classification rule: class based on sum of features
    final sum = feature1 + feature2 + feature3 + feature4 + feature5;
    final target = sum > 25 ? 1 : 0;

    data.add([feature1, feature2, feature3, feature4, feature5, target]);
  }

  return DataFrame(
    data,
    header: ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'target'],
  );
}

/// Generates test data for predictions
DataFrame _generateTestData() {
  final random = Random(123); // Different seed for test data
  final data = <List<num>>[];

  // Generate 5 test samples
  for (int i = 0; i < 5; i++) {
    final feature1 = random.nextDouble() * 10;
    final feature2 = random.nextDouble() * 10;
    final feature3 = random.nextDouble() * 10;
    final feature4 = random.nextDouble() * 10;
    final feature5 = random.nextDouble() * 10;

    data.add([feature1, feature2, feature3, feature4, feature5]);
  }

  return DataFrame(
    data,
    header: ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
  );
}

