import 'package:ml_algo/src/classifier/linear/linear_classifier.dart';
import 'package:ml_algo/src/classifier/linear/log_likelihood_optimizer_factory.dart';
import 'package:ml_algo/src/classifier/linear/softmax_regressor/softmax_regressor_impl.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/linear_optimizer/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';

/// A factory that creates different presets of softmax regressors
///
/// Softmax regression is an algorithm that solves a multiclass classification
/// problem. The algorithm uses maximization of the passed
/// data likelihood (as well as
/// [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression).
/// In other words, the regressor iteratively tries to select coefficients,
/// that makes combination of passed features and these coefficients most
/// likely. But, instead of [Logit link function](https://en.wikipedia.org/wiki/Logit)
/// it uses [Softmax link function](https://en.wikipedia.org/wiki/Softmax_function),
/// that's why the algorithm has such a name.
///
/// Also, it is worth to mention, that the algorithm is a generalization of
/// [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression))
abstract class SoftmaxRegressor implements LinearClassifier, Assessable {
  /// Creates a gradient descent based softmax regressor classifier.
  ///
  /// Parameters:
  ///
  /// [fittingData] A [DataFrame] with observations, that will be used by the
  /// classifier to learn coefficients of the hyperplane, which divides the
  /// features space, forming classes of the features. Should contain target
  /// column id (index or name)
  ///
  /// [targetNames] A collection of strings, that serves as names for the
  /// encoded target columns (a column, that contains class labels or outcomes
  /// for the associated features)
  ///
  /// [iterationsLimit] A number of fitting iterations. Default value is 100
  ///
  /// [initialLearningRate] A value, defining velocity of the convergence of the
  /// gradient descent solver. Default value is 1e-3
  ///
  /// [minCoefficientsUpdate] A minimum distance between weights vectors in two
  /// subsequent iterations. Uses as a condition of convergence in the
  /// [solver]. In other words, if difference is small, there is no reason to
  /// continue fitting. Default value is 1e-12
  ///
  /// [lambda] A coefficient of regularization. In gradient version of softmax
  /// regression L2 regularisation is used.
  ///
  /// [randomSeed] A seed, that will be passed to a random value generator,
  /// used by stochastic optimizers. Will be ignored, if the [solver] is not
  /// a stochastic. Remember, each time you run the regressor based on, for
  /// instance, stochastic gradient descent, with the same parameters, you will
  /// receive a different result. To avoid it, define [randomSeed]
  ///
  /// [batchSize] A size of data (in rows), that will be used for fitting per
  /// one iteration. If [batchSize] == `1` when stochastic gradient descent is
  /// used; if `1` < [batchSize] < `total number of rows`, when mini-batch
  /// gradient descent is used; if [batchSize] == `total number of rows`,
  /// when full-batch gradient descent is used
  ///
  /// [fitIntercept] Whether or not to fit intercept term. Default value is
  /// `false`.
  ///
  /// [interceptScale] A value, defining a size of the intercept term
  ///
  /// [learningRateType] A value, defining a strategy for the learning rate
  /// behaviour throughout the whole fitting process.
  ///
  /// [dtype] A data type for all the numeric values, used by the algorithm. Can
  /// affect performance or accuracy of the computations. Default value is
  /// [DType.float32]
  factory SoftmaxRegressor(
      DataFrame fittingData,
      Iterable<String> targetNames, {
        LinearOptimizerType optimizerType = LinearOptimizerType.vanillaGD,
        int iterationsLimit = 100,
        double initialLearningRate = 1e-3,
        double minCoefficientsUpdate = 1e-12,
        double lambda,
        int randomSeed,
        int batchSize = 1,
        bool fitIntercept = false,
        double interceptScale = 1.0,
        LearningRateType learningRateType,
        Matrix initialWeights,
        InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,
        DType dtype = DType.float32,
  }) {
        if (targetNames.isNotEmpty && targetNames.length < 2) {
            throw Exception('The target column should be encoded properly '
                '(e.g., via one-hot encoder)');
        }

        final dependencies = getDependencies();

        final linkFunctionFactory = dependencies
            .getDependency<LinkFunctionFactory>();

        final linkFunction = linkFunctionFactory
            .createByType(LinkFunctionType.softmax, dtype: dtype);

        final splits = featuresTargetSplit(fittingData,
              targetNames: targetNames,
        ).toList();

        final points = splits[0].toMatrix();
        final labels = splits[1].toMatrix();

        final optimizer = createLogLikelihoodOptimizer(
              points,
              labels,
              LinkFunctionType.softmax,
              optimizerType: optimizerType,
              iterationsLimit: iterationsLimit,
              initialLearningRate: initialLearningRate,
              minCoefficientsUpdate: minCoefficientsUpdate,
              lambda: lambda,
              randomSeed: randomSeed,
              batchSize: batchSize,
              learningRateType: learningRateType,
              initialWeightsType: initialWeightsType,
              fitIntercept: fitIntercept,
              interceptScale: interceptScale,
              dtype: dtype,
        );

        return SoftmaxRegressorImpl(
              optimizer,
              labels.uniqueRows(),
              linkFunction,
              batchSize: batchSize,
              fitIntercept: fitIntercept,
              interceptScale: interceptScale,
              initialWeights: initialWeights,
              dtype: dtype,
        );
  }
}
