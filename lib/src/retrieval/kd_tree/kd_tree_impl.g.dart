// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'kd_tree_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

KDTreeImpl _$KDTreeImplFromJson(Map<String, dynamic> json) => $checkedCreate(
      'KDTreeImpl',
      json,
      ($checkedConvert) {
        $checkKeys(
          json,
          allowedKeys: const ['P', 'L', 'R', 'D', 'S'],
        );
        final val = KDTreeImpl(
          $checkedConvert(
              'P', (v) => Matrix.fromJson(v as Map<String, dynamic>)),
          $checkedConvert('L', (v) => (v as num).toInt()),
          $checkedConvert(
              'R', (v) => KDTreeNode.fromJson(v as Map<String, dynamic>)),
          $checkedConvert('D', (v) => $enumDecode(_$DTypeEnumMap, v)),
          $checkedConvert('S', (v) => (v as num).toInt()),
        );
        return val;
      },
      fieldKeyMap: const {
        'points': 'P',
        'leafSize': 'L',
        'root': 'R',
        'dtype': 'D',
        'schemaVersion': 'S'
      },
    );

Map<String, dynamic> _$KDTreeImplToJson(KDTreeImpl instance) =>
    <String, dynamic>{
      'P': instance.points.toJson(),
      'L': instance.leafSize,
      'R': instance.root.toJson(),
      'D': _$DTypeEnumMap[instance.dtype]!,
      'S': instance.schemaVersion,
    };

const _$DTypeEnumMap = {
  DType.float32: 'float32',
  DType.float64: 'float64',
};
