// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'linear_regressor_impl.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

LinearRegressorImpl _$LinearRegressorImplFromJson(Map<String, dynamic> json) =>
    $checkedCreate(
      'LinearRegressorImpl',
      json,
      ($checkedConvert) {
        $checkKeys(
          json,
          allowedKeys: const [
            'OT',
            'IL',
            'LRT',
            'ICT',
            'ILT',
            'D',
            'DR',
            'MCU',
            'L',
            'RT',
            'RS',
            'BS',
            'IC',
            'FDN',
            'TN',
            'FI',
            'IS',
            'CS',
            'CPI',
            'DT',
            r'$V'
          ],
        );
        final val = LinearRegressorImpl(
          $checkedConvert(
              'CS', (v) => Vector.fromJson(v as Map<String, dynamic>)),
          $checkedConvert('TN', (v) => v as String),
          optimizerType: $checkedConvert(
              'OT',
              (v) => const LinearOptimizerTypeJsonConverter()
                  .fromJson(v as String)),
          iterationsLimit: $checkedConvert('IL', (v) => (v as num).toInt()),
          learningRateType: $checkedConvert(
              'LRT',
              (v) =>
                  const LearningRateTypeJsonConverter().fromJson(v as String)),
          initialCoefficientsType: $checkedConvert(
              'ICT',
              (v) => const InitialCoefficientsTypeJsonConverter()
                  .fromJson(v as String)),
          initialLearningRate: $checkedConvert('ILT', (v) => v as num),
          decay: $checkedConvert('D', (v) => v as num),
          dropRate: $checkedConvert('DR', (v) => (v as num).toInt()),
          minCoefficientsUpdate: $checkedConvert('MCU', (v) => v as num),
          lambda: $checkedConvert('L', (v) => v as num),
          batchSize: $checkedConvert('BS', (v) => (v as num).toInt()),
          isFittingDataNormalized: $checkedConvert('FDN', (v) => v as bool),
          fitIntercept: $checkedConvert(
              'FI', (v) => v as bool? ?? fitInterceptDefaultValue),
          interceptScale: $checkedConvert('IS',
              (v) => (v as num?)?.toDouble() ?? interceptScaleDefaultValue),
          dtype: $checkedConvert(
              'DT',
              (v) => v == null
                  ? dTypeDefaultValue
                  : const DTypeJsonConverter().fromJson(v as String)),
          schemaVersion: $checkedConvert(r'$V',
              (v) => (v as num?)?.toInt() ?? linearRegressorJsonSchemaVersion),
          regularizationType: $checkedConvert(
              'RT',
              (v) => const RegularizationTypeJsonConverterNullable()
                  .fromJson(v as String?)),
          randomSeed: $checkedConvert('RS', (v) => (v as num?)?.toInt()),
          initialCoefficients: $checkedConvert(
              'IC',
              (v) => v == null
                  ? null
                  : Matrix.fromJson(v as Map<String, dynamic>)),
          costPerIteration: $checkedConvert('CPI',
              (v) => (v as List<dynamic>?)?.map((e) => e as num).toList()),
        );
        return val;
      },
      fieldKeyMap: const {
        'coefficients': 'CS',
        'targetName': 'TN',
        'optimizerType': 'OT',
        'iterationsLimit': 'IL',
        'learningRateType': 'LRT',
        'initialCoefficientsType': 'ICT',
        'initialLearningRate': 'ILT',
        'decay': 'D',
        'dropRate': 'DR',
        'minCoefficientsUpdate': 'MCU',
        'lambda': 'L',
        'batchSize': 'BS',
        'isFittingDataNormalized': 'FDN',
        'fitIntercept': 'FI',
        'interceptScale': 'IS',
        'dtype': 'DT',
        'schemaVersion': r'$V',
        'regularizationType': 'RT',
        'randomSeed': 'RS',
        'initialCoefficients': 'IC',
        'costPerIteration': 'CPI'
      },
    );

Map<String, dynamic> _$LinearRegressorImplToJson(
        LinearRegressorImpl instance) =>
    <String, dynamic>{
      'OT': const LinearOptimizerTypeJsonConverter()
          .toJson(instance.optimizerType),
      'IL': instance.iterationsLimit,
      'LRT': const LearningRateTypeJsonConverter()
          .toJson(instance.learningRateType),
      'ICT': const InitialCoefficientsTypeJsonConverter()
          .toJson(instance.initialCoefficientsType),
      'ILT': instance.initialLearningRate,
      'D': instance.decay,
      'DR': instance.dropRate,
      'MCU': instance.minCoefficientsUpdate,
      'L': instance.lambda,
      if (const RegularizationTypeJsonConverterNullable()
              .toJson(instance.regularizationType)
          case final value?)
        'RT': value,
      if (instance.randomSeed case final value?) 'RS': value,
      'BS': instance.batchSize,
      if (instance.initialCoefficients?.toJson() case final value?) 'IC': value,
      'FDN': instance.isFittingDataNormalized,
      'TN': instance.targetName,
      'FI': instance.fitIntercept,
      'IS': instance.interceptScale,
      'CS': instance.coefficients.toJson(),
      if (instance.costPerIteration case final value?) 'CPI': value,
      'DT': const DTypeJsonConverter().toJson(instance.dtype),
      if (instance.schemaVersion case final value?) r'$V': value,
    };
