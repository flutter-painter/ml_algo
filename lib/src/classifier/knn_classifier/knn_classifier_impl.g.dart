// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'knn_classifier_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

KnnClassifierImpl _$KnnClassifierImplFromJson(Map<String, dynamic> json) =>
    $checkedCreate(
      'KnnClassifierImpl',
      json,
      ($checkedConvert) {
        $checkKeys(
          json,
          allowedKeys: const ['T', 'D', 'C', 'K', 'S', 'P', r'$V'],
        );
        final val = KnnClassifierImpl(
          $checkedConvert('T', (v) => v as String),
          $checkedConvert(
              'C', (v) => (v as List<dynamic>).map((e) => e as num).toList()),
          $checkedConvert(
              'K', (v) => const KernelJsonConverter().fromJson(v as String)),
          $checkedConvert(
              'S',
              (v) => const KnnSolverJsonConverter()
                  .fromJson(v as Map<String, dynamic>)),
          $checkedConvert('P', (v) => v as String),
          $checkedConvert(
              'D', (v) => const DTypeJsonConverter().fromJson(v as String)),
          schemaVersion: $checkedConvert(r'$V',
              (v) => (v as num?)?.toInt() ?? knnClassifierJsonSchemaVersion),
        );
        return val;
      },
      fieldKeyMap: const {
        'targetColumnName': 'T',
        'classLabels': 'C',
        'kernel': 'K',
        'solver': 'S',
        'classLabelPrefix': 'P',
        'dtype': 'D',
        'schemaVersion': r'$V'
      },
    );

Map<String, dynamic> _$KnnClassifierImplToJson(KnnClassifierImpl instance) =>
    <String, dynamic>{
      'T': instance.targetColumnName,
      'D': const DTypeJsonConverter().toJson(instance.dtype),
      'C': instance.classLabels,
      'K': const KernelJsonConverter().toJson(instance.kernel),
      'S': const KnnSolverJsonConverter().toJson(instance.solver),
      'P': instance.classLabelPrefix,
      if (instance.schemaVersion case final value?) r'$V': value,
    };
