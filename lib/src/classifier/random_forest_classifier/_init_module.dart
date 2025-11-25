import 'package:injector/injector.dart';
import 'package:ml_algo/src/classifier/decision_tree_classifier/_init_module.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/_injector.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier_factory.dart';
import 'package:ml_algo/src/classifier/random_forest_classifier/random_forest_classifier_factory_impl.dart';
import 'package:ml_algo/src/extensions/injector.dart';

Injector initRandomForestModule() {
  initDecisionTreeModule();

  return randomForestInjector
    ..registerSingletonIf<RandomForestClassifierFactory>(
        () => RandomForestClassifierFactoryImpl());
}
