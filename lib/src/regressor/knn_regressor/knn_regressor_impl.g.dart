// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'knn_regressor_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

KnnRegressorImpl _$KnnRegressorImplFromJson(Map<String, dynamic> json) =>
    $checkedCreate(
      'KnnRegressorImpl',
      json,
      ($checkedConvert) {
        $checkKeys(
          json,
          allowedKeys: const ['D', 'T', 'S', 'K', r'$V'],
        );
        final val = KnnRegressorImpl(
          $checkedConvert('T', (v) => v as String),
          $checkedConvert(
              'S',
              (v) => const KnnSolverJsonConverter()
                  .fromJson(v as Map<String, dynamic>)),
          $checkedConvert(
              'K', (v) => const KernelJsonConverter().fromJson(v as String)),
          $checkedConvert(
              'D', (v) => const DTypeJsonConverter().fromJson(v as String)),
          schemaVersion: $checkedConvert(r'$V',
              (v) => (v as num?)?.toInt() ?? knnRegressorJsonSchemaVersion),
        );
        return val;
      },
      fieldKeyMap: const {
        'targetName': 'T',
        'solver': 'S',
        'kernel': 'K',
        'dtype': 'D',
        'schemaVersion': r'$V'
      },
    );

Map<String, dynamic> _$KnnRegressorImplToJson(KnnRegressorImpl instance) =>
    <String, dynamic>{
      'D': const DTypeJsonConverter().toJson(instance.dtype),
      'T': instance.targetName,
      'S': const KnnSolverJsonConverter().toJson(instance.solver),
      'K': const KernelJsonConverter().toJson(instance.kernel),
      if (instance.schemaVersion case final value?) r'$V': value,
    };
