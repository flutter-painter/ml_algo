import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/_init_module.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier_factory.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/voting_strategy.dart';
import 'package:ml_algo/src/common/constants/default_parameters/common.dart';
import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/persistence/tree_store.dart';
import 'package:ml_algo/src/predictor/retrainable.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

/// A class that performs classification using Random Forest algorithm
///
/// Random Forest is an ensemble method that trains multiple Decision Trees
/// on bootstrap samples of the training data and aggregates their predictions.
abstract class RandomForestClassifier
    implements
        Assessable,
        Serializable,
        Retrainable<RandomForestClassifier>,
        Classifier {
  /// Factory constructor for creating a Random Forest classifier
  ///
  /// Parameters:
  ///
  /// [trainData] A [DataFrame] with observations for training.
  /// Must contain [targetName] column.
  ///
  /// [targetName] Name of the target column in [trainData].
  ///
  /// [nEstimators] Number of trees in the forest (default: 100).
  ///
  /// [bootstrap] Whether to use bootstrap sampling (default: true).
  ///
  /// [sampleSize] Size of bootstrap sample as fraction of training data
  /// (default: 1.0, meaning same size as training data).
  ///
  /// [maxFeatures] Number of features to consider at each split.
  /// Can be:
  /// - An integer (exact number)
  /// - A double (fraction of features, e.g., 0.5 = 50%)
  /// - "sqrt" (default: sqrt of total features)
  /// - "log2" (log2 of total features)
  /// - null (use all features, no feature sampling)
  ///
  /// [votingStrategy] How to aggregate tree predictions (default: majority).
  ///
  /// [treeStore] Optional TreeStore for persisting trees (default: null,
  /// trees stored in memory only).
  ///
  /// [parallel] Whether to train trees in parallel (default: false).
  ///
  /// [seed] Random seed for reproducibility (default: null).
  ///
  /// [classWeight] Class weights for handling imbalanced datasets.
  /// - `null` (default): No class weighting
  /// - `'balanced'`: Automatically calculate weights inversely proportional to class frequency
  /// - `Map<num, double>`: Manual class weights (class label -> weight)
  ///
  /// [balancedBootstrap] Whether to use balanced bootstrap sampling (equal samples per class).
  /// Only effective when [bootstrap] is true (default: false).
  ///
  /// Decision Tree hyperparameters (passed to each tree):
  /// [minError] Minimal error on a tree node (default: 0.5).
  /// [minSamplesCount] Minimal samples per node (default: 1).
  /// [maxDepth] Maximum tree depth (default: 10).
  /// [assessorType] Tree assessment type (default: gini).
  /// [dtype] Data type (default: float64).
  factory RandomForestClassifier(
    DataFrame trainData,
    String targetName, {
    int nEstimators = 100,
    bool bootstrap = true,
    double sampleSize = 1.0,
    dynamic maxFeatures = 'sqrt',
    VotingStrategy votingStrategy = VotingStrategy.majority,
    TreeStore? treeStore,
    bool parallel = false,
    int? seed,
    dynamic classWeight,
    bool balancedBootstrap = false,
    // Decision Tree hyperparameters
    num minError = 0.5,
    int minSamplesCount = 1,
    int maxDepth = 10,
    TreeAssessorType assessorType = TreeAssessorType.gini,
    DType dtype = dTypeDefaultValue,
  }) =>
      initRandomForestModule().get<RandomForestClassifierFactory>().create(
            trainData: trainData,
            targetName: targetName,
            nEstimators: nEstimators,
            bootstrap: bootstrap,
            sampleSize: sampleSize,
            maxFeatures: maxFeatures,
            votingStrategy: votingStrategy,
            treeStore: treeStore,
            parallel: parallel,
            seed: seed,
            minError: minError,
            minSamplesCount: minSamplesCount,
            maxDepth: maxDepth,
            assessorType: assessorType,
            dtype: dtype,
            classWeight: classWeight,
            balancedBootstrap: balancedBootstrap,
          );

  /// Restores a Random Forest from JSON
  factory RandomForestClassifier.fromJson(String json) =>
      initRandomForestModule()
          .get<RandomForestClassifierFactory>()
          .fromJson(json);

  /// Number of trees in the forest
  int get nEstimators;

  /// Whether bootstrap sampling is used
  bool get bootstrap;

  /// Sample size as fraction of training data
  double get sampleSize;

  /// Maximum features to consider at each split
  dynamic get maxFeatures;

  /// Voting strategy used for aggregation
  VotingStrategy get votingStrategy;

  /// Tree IDs (if using TreeStore) or null (if in-memory)
  List<String>? get treeIds;

  /// Out-Of-Bag score (accuracy on samples not used in bootstrap).
  /// Only available when [bootstrap] is true, otherwise null.
  double? get oobScore;

  /// Feature importance scores (normalized to sum to 1.0).
  /// Higher values indicate more important features.
  Map<String, double> get featureImportance;

  /// Decision Tree hyperparameters (same for all trees)
  num get minError;
  int get minSamplesCount;
  int get maxDepth;
  TreeAssessorType get assessorType;
}
