// On-device risk classifier for the STSMIRS Flutter app.
//
// The deployed model is a multinomial logistic regression: 4 classes x 5 features
// (= 20 weights) + 4 biases, plus 5 means and 5 scales for standardisation.
// Copy models/deployed_model.json into the app's assets and load it with
// RiskClassifier.fromJson(...). No inference runtime is needed.
import 'dart:convert';
import 'dart:math' as math;

class RiskResult {
  final String label;          // LOW, MODERATE, HIGH, CRITICAL
  final double confidence;     // probability of the predicted class
  final List<double> probabilities;
  final bool escalate;         // send to the cloud LLM?
  RiskResult(this.label, this.confidence, this.probabilities, this.escalate);
}

class RiskClassifier {
  final List<String> classes;
  final List<double> mean, scale, intercept;
  final List<List<double>> coef;
  final double theta;          // confidence threshold, tunable per zone
  final bool safetyAware;      // also escalate every HIGH / CRITICAL prediction

  RiskClassifier(this.classes, this.mean, this.scale, this.coef, this.intercept,
      {this.theta = 0.7, this.safetyAware = true});

  factory RiskClassifier.fromJson(String jsonText, {double theta = 0.7, bool safetyAware = true}) {
    final j = jsonDecode(jsonText) as Map<String, dynamic>;
    List<double> v(String k) => (j[k] as List).map((e) => (e as num).toDouble()).toList();
    final coef = (j['coef'] as List)
        .map((row) => (row as List).map((e) => (e as num).toDouble()).toList())
        .toList();
    return RiskClassifier(List<String>.from(j['classes']), v('scaler_mean'), v('scaler_scale'),
        coef, v('intercept'), theta: theta, safetyAware: safetyAware);
  }

  /// displacementM: distance from planned route (m); idleMin: idle minutes (>= 15);
  /// hour: 0-24 local time; inHazardZone: cached geofence flag.
  RiskResult classify(double displacementM, double idleMin, double hour, bool inHazardZone) {
    final x = [
      math.log(math.max(displacementM, 20.0)),
      math.log(idleMin),
      math.sin(2 * math.pi * hour / 24),
      math.cos(2 * math.pi * hour / 24),
      inHazardZone ? 1.0 : 0.0,
    ];
    final z = List<double>.generate(classes.length, (k) {
      var s = intercept[k];
      for (var i = 0; i < x.length; i++) {
        s += coef[k][i] * (x[i] - mean[i]) / scale[i];
      }
      return s;
    });
    final m = z.reduce(math.max);
    final e = z.map((v) => math.exp(v - m)).toList();
    final sum = e.reduce((a, b) => a + b);
    final p = e.map((v) => v / sum).toList();
    var best = 0;
    for (var k = 1; k < p.length; k++) {
      if (p[k] > p[best]) best = k;
    }
    final escalate = p[best] < theta || (safetyAware && best >= 2);
    return RiskResult(classes[best], p[best], p, escalate);
  }
}
