from flask import Flask, render_template, request, jsonify
import pickle, os, numpy as np

app = Flask(__name__)


class HypertensionModel:
    def __init__(self, model, scaler, features):
        self.model   = model
        self.scaler  = scaler
        self.features = features
        self.stage_labels = {
            0: 'Normal',
            1: 'Stage 1 Hypertension',
            2: 'Stage 2 Hypertension',
            3: 'Hypertensive Crisis',
        }
        self.encode_maps = {
            'Gender':        {'Male': 0, 'Female': 1},
            'Age':           {'18-34': 0, '35-50': 1, '51-64': 2, '65+': 3},
            'Severity':      {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3},
            'Whendiagnoused':{'<1 Year': 0, '1 - 5 Years': 1, '>5 Years': 2},
            'Systolic':      {'100+': 0, '111 - 120': 1, '121- 130': 2, '121 - 130': 2, '130+': 3},
            'Diastolic':     {'70 - 80': 0, '81 - 90': 1, '91 - 100': 2, '100+': 3, '130+': 4},
        }
        self.binary_cols = [
            'History', 'Patient', 'TakeMedication',
            'BreathShortness', 'VisualChanges', 'NoseBleeding', 'ControlledDiet',
        ]

    def predict_patient(self, data):
        row = []
        for feat in self.features:
            val = str(data.get(feat, '')).strip()
            if feat in self.encode_maps:
                val = self.encode_maps[feat].get(val, 0)
            elif feat in self.binary_cols:
                val = 1 if val in ['Yes', 'yes', '1'] else 0
            else:
                val = 0
            row.append(float(val))
        X_sc  = self.scaler.transform(np.array(row).reshape(1, -1))
        stage = int(self.model.predict(X_sc)[0])
        proba = self.model.predict_proba(X_sc)[0]
        return {
            'stage':         stage,
            'label':         self.stage_labels[stage],
            'confidence':    round(float(max(proba)) * 100, 1),
            'probabilities': {
                self.stage_labels[i]: round(float(p) * 100, 1)
                for i, p in enumerate(proba)
            },
        }

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))



MODEL_PATH = os.path.join(os.path.dirname(__file__), 'logreg_model.pkl')
model = None

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded. Features:", getattr(model, 'features', 'N/A'))
except FileNotFoundError:
    print("logreg_model.pkl not found - using rule-based mode.")

STAGES = {
    -1: {"label": "Hypotension",          "color": "#2979ff", "urgency": "warning",
         "summary": "Blood pressure is below normal range. Monitor closely and seek evaluation."},
    0:  {"label": "Normal",               "color": "#4CAF50", "urgency": "low",
         "summary": "Blood pressure is within a healthy range."},
    1:  {"label": "Elevated",             "color": "#FFD700", "urgency": "low",
         "summary": "Slightly above normal — lifestyle modification advised."},
    2:  {"label": "Stage 1 Hypertension", "color": "#FFB347", "urgency": "medium",
         "summary": "Elevated pressure — lifestyle changes and possible medication recommended."},
    3:  {"label": "Stage 2 Hypertension", "color": "#FF6B35", "urgency": "high",
         "summary": "High BP — medical evaluation and likely medication needed."},
    4:  {"label": "Hypertensive Crisis",  "color": "#FF2D2D", "urgency": "critical",
         "summary": "URGENT: Seek emergency medical care immediately."},
}


def numeric_stage(sys: float, dia: float) -> int:
    if sys < 90 or dia < 60:   return -1   # Hypotension (FIX: was missing)
    if sys > 180 or dia > 120: return 4    # Crisis
    if sys >= 140 or dia >= 90: return 3   # Stage 2
    if sys >= 130 or dia >= 80: return 2   # Stage 1
    if sys >= 120 and dia < 80: return 1   # Elevated
    return 0                               # Normal

def sys_bucket(v: float) -> str:
    if v >= 130: return '130+'
    if v >= 121: return '121 - 130'
    if v >= 111: return '111 - 120'
    return '100+'

def dia_bucket(v: float) -> str:
    if v >= 130: return '130+'
    if v >= 100: return '100+'
    if v >= 91:  return '91 - 100'
    if v >= 81:  return '81 - 90'
    return '70 - 80'

def age_bucket(v: float) -> str:
    if v >= 65: return '65+'
    if v >= 51: return '51-64'
    if v >= 35: return '35-50'
    return '18-34'


def get_clinical_alerts(sys: float, dia: float, pp: float,
                        map_val: float, hr: float, stage: int) -> list:
    alerts = []

    if stage == -1:
        alerts.append({
            "icon": "🔻", "type": "danger",
            "title": "Hypotension Detected",
            "text": (f"BP of {sys:.0f}/{dia:.0f} mmHg is below normal (90/60). "
                     "Possible causes: dehydration, medications, or cardiac conditions. "
                     "Seek medical evaluation.")
        })

    if pp > 60:
        alerts.append({
            "icon": "⚡", "type": "danger",
            "title": "Wide Pulse Pressure",
            "text": (f"Pulse pressure of {pp:.0f} mmHg (>60) can indicate aortic regurgitation, "
                     "arterial stiffness, or thyrotoxicosis. Cardiovascular evaluation advised.")
        })
    elif pp < 25 and sys > 80:
        alerts.append({
            "icon": "↔", "type": "warning",
            "title": "Narrow Pulse Pressure",
            "text": (f"Pulse pressure of {pp:.0f} mmHg (<25) may indicate cardiac tamponade, "
                     "aortic stenosis, or severe heart failure.")
        })

    if map_val < 65:
        alerts.append({
            "icon": "⚠", "type": "danger",
            "title": "Low Mean Arterial Pressure",
            "text": (f"MAP of {map_val:.0f} mmHg is critically low (normal 70-100 mmHg). "
                     "Organ perfusion may be compromised.")
        })

    if hr > 0:
        if hr > 100:
            alerts.append({
                "icon": "💓", "type": "warning",
                "title": "Tachycardia",
                "text": f"Heart rate of {hr:.0f} bpm exceeds normal range (60-100 bpm). Monitor for underlying cause."
            })
        elif hr < 50:
            alerts.append({
                "icon": "💔", "type": "warning",
                "title": "Bradycardia",
                "text": f"Heart rate of {hr:.0f} bpm is below normal range. May be athletic or a conduction issue."
            })

    return alerts


def get_recommendations(stage: int, data: dict) -> list:
    recs = []

    if stage == 4:
        recs.append({"icon": "🚨", "text": "URGENT: Go to the emergency room or call emergency services now.", "priority": "critical"})

    if stage == -1:
        recs.append({"icon": "💧", "text": "Increase fluid and salt intake (under medical guidance) to help raise blood pressure.", "priority": "high"})
        recs.append({"icon": "🛋️", "text": "Avoid prolonged standing. Rise slowly from sitting/lying positions to prevent dizziness.", "priority": "high"})
        recs.append({"icon": "👨‍⚕️", "text": "Consult a physician to investigate the underlying cause of low blood pressure.", "priority": "high"})

    if stage >= 3:
        recs.append({"icon": "💊", "text": "Discuss antihypertensive medication with your physician immediately.", "priority": "high"})

    if stage >= 2:
        recs.append({"icon": "🥗", "text": "Follow the DASH diet: limit sodium to under 2,300 mg/day.", "priority": "high"})
        recs.append({"icon": "🏃", "text": "Target 150 minutes of moderate aerobic exercise per week.", "priority": "medium"})
        recs.append({"icon": "📊", "text": "Monitor blood pressure daily and keep a written log.", "priority": "medium"})

    if data.get('smoking'):
        recs.append({"icon": "🚬", "text": "Smoking raises BP and cardiovascular risk — cessation support is available.", "priority": "high"})

    if data.get('diabetes'):
        recs.append({"icon": "🩺", "text": "Diabetes + hypertension significantly raise cardiac risk — target BP below 130/80.", "priority": "high"})

    if data.get('family_hx'):
        recs.append({"icon": "🧬", "text": "Family history raises your risk — annual cardiovascular screening is advised.", "priority": "medium"})

    bmi = data.get('bmi', 22)
    if bmi and float(bmi) >= 30:
        recs.append({"icon": "⚖️", "text": "Losing 5-10% body weight can reduce systolic BP by 5-20 mmHg.", "priority": "medium"})

    alcohol = data.get('alcohol', 0)
    if alcohol and float(alcohol) > 14:
        recs.append({"icon": "🍷", "text": f"Reducing alcohol from {int(alcohol)} to under 14 drinks/week can lower BP by 3-4 mmHg.", "priority": "medium"})

    activity = data.get('activity', 1)
    if activity is not None and int(activity) == 0:
        recs.append({"icon": "🚶", "text": "Even light daily walking (30 min) can lower BP by 4-9 mmHg.", "priority": "medium"})

    age = data.get('age', 40)
    if age and float(age) >= 65:
        recs.append({"icon": "👴", "text": "Older adults: rise slowly from seated positions to avoid dizziness.", "priority": "low"})

    if stage == 0:
        recs.append({"icon": "✅", "text": "Great work! Maintain your healthy lifestyle and schedule annual BP check-ups.", "priority": "low"})

    return recs


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        sys_val = float(data.get('systolic', 120))
        dia_val = float(data.get('diastolic', 80))
        hr_val  = float(data.get('heart_rate', 0) or 0)

       
        pp_val  = round(sys_val - dia_val, 1)
        map_val = round(dia_val + (pp_val / 3), 1)

        # Rule-based stage (now includes hypotension as -1)
        rule_stage = numeric_stage(sys_val, dia_val)
        stage      = rule_stage

        confidence = None
        proba_map  = None

        
        if model is not None and rule_stage >= 0:
            try:
                model_input = {
                    'Gender':         'Male' if data.get('gender', 'Male') == 'Male' else 'Female',
                    'Age':            age_bucket(float(data.get('age', 40))),
                    'History':        'Yes' if data.get('family_hx') else 'No',
                    'Patient':        'Yes' if data.get('existing_patient') else 'No',
                    'TakeMedication': 'Yes' if data.get('on_medication') else 'No',
                    'Severity':       data.get('symptom_severity', 'None'),
                    'BreathShortness':'Yes' if data.get('shortness_breath') else 'No',
                    'VisualChanges':  'Yes' if data.get('visual_changes') else 'No',
                    'NoseBleeding':   'Yes' if data.get('nosebleeds') else 'No',
                    'Whendiagnoused': '<1 Year',
                    'Systolic':       sys_bucket(sys_val),
                    'Diastolic':      dia_bucket(dia_val),
                    'ControlledDiet': 'Yes' if data.get('controlled_diet') else 'No',
                }
                result = model.predict_patient(model_input)
               
                ml_stage_map = {0: 0, 1: 2, 2: 3, 3: 4}
                ml_stage   = ml_stage_map.get(result['stage'], result['stage'])
                confidence = result['confidence']
                proba_map  = result['probabilities']

                stage = max(rule_stage, ml_stage)
                if ml_stage < rule_stage:
                    confidence = None
                    proba_map  = None

            except Exception as e:
                print(f"[ML error] {e} - keeping rule_stage={stage}")

        stage_info      = STAGES[stage]
        clinical_alerts = get_clinical_alerts(sys_val, dia_val, pp_val, map_val, hr_val, stage)
        recs            = get_recommendations(stage, data)
        risk_map        = {-1: 5, 0: 0, 1: 15, 2: 35, 3: 65, 4: 100}
        risk_score      = risk_map.get(stage, 0)

        return jsonify({
            'success':         True,
            'stage':           stage,
            'label':           stage_info['label'],
            'color':           stage_info['color'],
            'urgency':         stage_info['urgency'],
            'summary':         stage_info['summary'],
            'confidence':      confidence,
            'risk_score':      risk_score,
            'systolic':        sys_val,
            'diastolic':       dia_val,
            'map':             map_val,
            'pulse_pressure':  pp_val,
            'probabilities':   proba_map,
            'clinical_alerts': clinical_alerts,
            'recommendations': recs,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({
        'status':       'ok',
        'model_loaded': model is not None,
        'features':     getattr(model, 'features', None),
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
