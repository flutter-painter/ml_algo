import 'dart:typed_data';

import 'package:ml_algo/src/data_preprocessing/ml_data/csv_ml_data.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/ml_data.dart';

/// An abstract factory for instantiating a [Float32x4CsvMLDataInternal] exemplar. The latter is used for reading
/// csv-files and getting distinct data structures for features and labels.
abstract class Float32x4CsvMLData {

  /// Creates a csv-data instance from file. Resulting instance uses [Float32x4] data type for features and labels
  /// representation
  /// [fileName] Target csv-file name
  /// [eol] End of line marker of the csv-file
  /// [labelPos] Position of the label column (by default - the last column)
  /// [headerExists] Indicates, whether the csv-file header exists or not
  static MLData<Float32x4> fromFile(String fileName, {String eol = '\n', int labelPos, bool headerExists = true}) =>
      Float32x4CsvMLDataInternal.fromFile(fileName, eol: eol, labelPos: labelPos, headerExists: headerExists);
}