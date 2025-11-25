// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'random_binary_projection_searcher_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

RandomBinaryProjectionSearcherImpl _$RandomBinaryProjectionSearcherImplFromJson(
        Map<String, dynamic> json) =>
    $checkedCreate(
      'RandomBinaryProjectionSearcherImpl',
      json,
      ($checkedConvert) {
        $checkKeys(
          json,
          allowedKeys: const ['S', 'D', 'H', 'P', 'R', 'B', r'$V'],
        );
        final val = RandomBinaryProjectionSearcherImpl(
          $checkedConvert(
              'H', (v) => (v as List<dynamic>).map((e) => e as String)),
          $checkedConvert(
              'P', (v) => Matrix.fromJson(v as Map<String, dynamic>)),
          $checkedConvert('D', (v) => (v as num).toInt()),
          seed: $checkedConvert('S', (v) => (v as num?)?.toInt()),
          schemaVersion:
              $checkedConvert(r'$V', (v) => (v as num?)?.toInt() ?? 1),
        );
        $checkedConvert(
            'R',
            (v) =>
                val.randomVectors = Matrix.fromJson(v as Map<String, dynamic>));
        $checkedConvert(
            'B',
            (v) => val.bins = (v as Map<String, dynamic>).map(
                  (k, e) => MapEntry(
                      int.parse(k),
                      (e as List<dynamic>)
                          .map((e) => (e as num).toInt())
                          .toList()),
                ));
        return val;
      },
      fieldKeyMap: const {
        'columns': 'H',
        'points': 'P',
        'digitCapacity': 'D',
        'seed': 'S',
        'schemaVersion': r'$V',
        'randomVectors': 'R',
        'bins': 'B'
      },
    );

Map<String, dynamic> _$RandomBinaryProjectionSearcherImplToJson(
        RandomBinaryProjectionSearcherImpl instance) =>
    <String, dynamic>{
      if (instance.seed case final value?) 'S': value,
      'D': instance.digitCapacity,
      'H': instance.columns.toList(),
      'P': instance.points.toJson(),
      'R': instance.randomVectors.toJson(),
      'B': instance.bins.map((k, e) => MapEntry(k.toString(), e)),
      r'$V': instance.schemaVersion,
    };
