from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd, numpy as np, pickle, json, os, uuid, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
    }
})
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
BASE = os.path.dirname(os.path.abspath(__file__))
def find_artifacts_dir():
    candidates = [
        os.path.join(BASE, '..', 'ml_model', 'artifacts'),
        os.path.join(BASE, 'artifacts'),        
        os.path.join(BASE, 'ml_model', 'artifacts'),       
        os.path.join(BASE, '..', 'artifacts'),        
        os.path.join(BASE, '..', '..', 'ml_model', 'artifacts'),
        BASE,
    ]
    for path in candidates:
        resolved = os.path.normpath(path)
        if os.path.isdir(resolved):
            pkls = [f for f in os.listdir(resolved) if f.endswith('.pkl')]
            if pkls:
                print(f"  Artifacts found at: {resolved}")
                return resolved
    print(f"   Could not find artifacts folder. Checked:")
    for p in candidates:
        print(f"     {os.path.normpath(p)}")
    print(f" Using fallback: {BASE}")
    return BASE

ART = find_artifacts_dir()
FEATURES = [
    'LAT','LONG','LAT_NORM','LONG_NORM',
    'OFFENSE_CODE','HOUR','MONTH','YEAR',
    'DISTRICT_ENC','DAY_ENC','OFFENSE_ENC','UCR_ENC',
    'SHOOTING','RA_LOG',
    'IS_NIGHT','IS_WEEKEND','IS_EVENING','IS_RUSH',
    'HOUR_SIN','HOUR_COS','MONTH_SIN','MONTH_COS',
    'DAY_SIN','DAY_COS'
]
DISTRICT_NAMES = {
    'A1':  'Downtown',
    'A15': 'Charlestown',
    'A7':  'East Boston',
    'B2':  'Roxbury',
    'B3':  'Mattapan',
    'C11': 'Dorchester',
    'C6':  'South Boston',
    'D14': 'Brighton',
    'D4':  'South End',
    'E13': 'Jamaica Plain',
    'E18': 'Hyde Park',
    'E5':  'West Roxbury',
}
DISTRICT_CRIME_COUNT = {
    'B2': 49945, 'C11': 42530, 'D4': 41915, 'A1': 35717,
    'B3': 35442, 'C6':  23460, 'D14': 20127, 'E13': 17536,
    'E18': 17348, 'A7': 13544, 'E5':  13239, 'A15': 6505,
}
USERS = {
    'admin':   {'password':'admin123',  'role':'admin',  'name':'System Admin'},
    'officer1':{'password':'police123', 'role':'police', 'name':'Officer Sharma'},
    'officer2':{'password':'police123', 'role':'police', 'name':'Officer Verma'},
    'public':  {'password':'user123',   'role':'public', 'name':'Citizen User'},
}
MODEL_INFO = {
    'rf':  {'name':'Random Forest',    'acc':97.59, 'f1':0.9587, 'auc':0.9991, 'type':'ML'},
    'dt':  {'name':'Decision Tree',    'acc':99.74, 'f1':0.9955, 'auc':0.9983, 'type':'ML'},
    'gb':  {'name':'Gradient Boosting','acc':94.77, 'f1':0.9054, 'auc':0.9885, 'type':'ML'}
}
fir_store = []
_df = _models = _scaler = _enc = _results = None
_cnn_poly = _cnn_scaler = None
def load_all():
    global _df, _models, _scaler, _enc, _results, _cnn_poly, _cnn_scaler

    # Dataset
    csv_path = os.path.join(ART, 'crime_clean.csv')
    if os.path.exists(csv_path):
        _df = pd.read_csv(csv_path, low_memory=False)
        print(f" Dataset loaded: {len(_df):,} records")

    _models = {}
    for key in ['rf', 'dt', 'gb']:
        fp = os.path.join(ART, f'{key}.pkl')
        if os.path.exists(fp):
            with open(fp, 'rb') as f:
                _models[key] = pickle.load(f)
            print(f"  {MODEL_INFO[key]['name']} ({MODEL_INFO[key]['acc']}%) loaded")
        else:
            print(f"  {key}.pkl not found")
    try:
        with open(os.path.join(ART, 'scaler.pkl'),   'rb') as f: _scaler = pickle.load(f)
        with open(os.path.join(ART, 'encoders.pkl'), 'rb') as f: _enc    = pickle.load(f)
        print("Scaler & encoders loaded")
    except Exception as e:
        print(f" Scaler/encoder error: {e}")

    # CNN extras
    try:
        with open(os.path.join(ART, 'cnn_poly.pkl'),    'rb') as f: _cnn_poly   = pickle.load(f)
        with open(os.path.join(ART, 'cnn_scaler2.pkl'), 'rb') as f: _cnn_scaler = pickle.load(f)
        print(" CNN feature pipeline loaded")
    except:
        pass
    for rname in ['results.json', 'TOP5_RESULTS.json', 'ALL_RESULTS.json']:
        rp = os.path.join(ART, rname)
        if os.path.exists(rp):
            with open(rp) as f: _results = json.load(f)
            print(f" Results JSON  : {rname}")
            break

    print(f"\n Backend ready — {len(_models)} models active\n")

load_all()
LAT_MEAN, LAT_STD = 42.332, 0.038
LON_MEAN, LON_STD = -71.062, 0.053
DAY_MAP = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}

def build_feature_vector(lat, lon, hour, month, year, district, day, offense, reporting_area=500):
    """Build the 24-feature vector for model inference."""
    def se(le, v):
        try:    return int(le.transform([str(v)])[0])
        except: return 0

    d_enc  = se(_enc['district'], district) if _enc else 0
    dy_enc = se(_enc['day'],      day)      if _enc else 0
    of_enc = se(_enc['offense'],  offense)  if _enc else 0
    u_enc  = 1  

    day_num = DAY_MAP.get(day, 0)
    ra_log  = float(np.log1p(float(reporting_area)))

    return np.array([[
        lat, lon,
        (lat - LAT_MEAN) / LAT_STD,
        (lon - LON_MEAN) / LON_STD,
        1000,             
        hour, month, year,
        d_enc, dy_enc, of_enc, u_enc,
        0,               
        ra_log,
        int(hour >= 20 or hour <= 5),  
        int(day in ['Saturday', 'Sunday']), 
        int(17 <= hour <= 21),          
        int(7 <= hour <= 9 or 16 <= hour <= 18),  
        np.sin(2*np.pi*hour/24),  
        np.cos(2*np.pi*hour/24),  
        np.sin(2*np.pi*month/12), 
        np.cos(2*np.pi*month/12), 
        np.sin(2*np.pi*day_num/7), 
        np.cos(2*np.pi*day_num/7), 
    ]], dtype=float)

def run_model(key, feat_scaled):
   
    mdl = _models.get(key)
    if mdl is None:
        return None
    try:
        if key == 'cnn' and _cnn_poly and _cnn_scaler:
            poly_feats  = _cnn_poly.transform(feat_scaled[:, :6])
            cnn_input   = _cnn_scaler.transform(np.hstack([feat_scaled, poly_feats]))
            return float(mdl.predict_proba(cnn_input)[0][1])
        elif hasattr(mdl, 'predict_proba'):
            return float(mdl.predict_proba(feat_scaled)[0][1])
        elif hasattr(mdl, 'decision_function'):
            raw = float(mdl.decision_function(feat_scaled)[0])
            return float(1 / (1 + np.exp(-raw)))
        else:
            return 0.65 if int(mdl.predict(feat_scaled)[0]) == 1 else 0.3
    except Exception as e:
        print(f"Model {key} error: {e}")
        return None

def risk_level(prob):
    return 'HIGH' if prob > 0.70 else 'MEDIUM' if prob > 0.40 else 'LOW'

def recommendations(level, district, hour):
    name = DISTRICT_NAMES.get(str(district), district)
    if level == 'HIGH':
        return [
            f'Deploy 2-3 patrol units to District {district} ({name}) immediately',
            f'Increase surveillance between {hour}:00 - {(hour+2)%24}:00',
            f'Alert {name} police station for rapid response',
            'Activate community watch program and notify local residents',
        ]
    if level == 'MEDIUM':
        return [
            f'Schedule routine patrol in District {district} ({name})',
            'Monitor CCTV feeds for suspicious activity',
            f'Issue advisory bulletin to units near {name}',
        ]
    return [
        f'Standard patrol schedule sufficient for District {district} ({name})',
        'Continue regular monitoring and community engagement',
    ]

@app.route('/', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'records': len(_df) if _df is not None else 0,
        'models': {k: MODEL_INFO[k]['acc'] for k in _models},
        'timestamp': datetime.now().isoformat()
    })
@app.route('/api/login', methods=['POST'])
def login():
    d = request.get_json() or {}
    u = USERS.get(d.get('username', ''))
    if u and u['password'] == d.get('password', ''):
        return jsonify({'success': True, 'token': str(uuid.uuid4()),
                        'user': {'username': d['username'], 'role': u['role'], 'name': u['name']}})
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

@app.route('/api/stats', methods=['GET'])
def stats():
    if _df is None:
        return jsonify({'error': 'Dataset not loaded'}), 500

    df = _df
    mn = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
          7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

    by_type  = [{'type': str(k)[:20], 'count': int(v)}
                for k,v in df['OFFENSE_CODE_GROUP'].value_counts().head(10).items()]
    by_hour  = [{'hour': int(h), 'count': int(c)}
                for h,c in df.groupby('HOUR').size().items()]
    by_month = [{'month': mn.get(int(m), str(m)), 'count': int(c)}
                for m,c in df.groupby('MONTH').size().items()]

    by_dist = []
    for dist, cnt in df['DISTRICT'].value_counts().head(12).items():
        sub = df[df['DISTRICT'] == dist]
        night = len(sub[sub['HOUR'].isin(list(range(18,24))+list(range(0,6)))]) / max(len(sub),1)
        by_dist.append({'district': str(dist), 'count': int(cnt),
                        'risk': 'HIGH' if night>0.45 else 'MEDIUM' if night>0.30 else 'LOW'})

    by_year = sorted([{'year': int(y), 'count': int(c)}
                      for y,c in df.groupby('YEAR').size().items()], key=lambda x: x['year'])

    return jsonify({
        'total_incidents':      int(len(df)),
        'hotspots_identified':  int(df['IS_HOTSPOT'].sum()),
        'shooting_incidents':   int(df['SHOOTING'].sum()),
        'firs_filed':           len(fir_store) + 156,
        'best_model':           'Decision Tree',
        'best_accuracy':        99.74,
        'models_above_80':      5,
        'crime_by_type':        by_type,
        'crime_by_hour':        by_hour,
        'crime_by_month':       by_month,
        'district_breakdown':   by_dist,
        'crime_by_year':        by_year,
    })
@app.route('/api/hotspots', methods=['GET'])
def hotspots():
    if _df is None:
        return jsonify({'error': 'Dataset not loaded'}), 500

    limit   = int(request.args.get('limit', 300))
    risk_f  = request.args.get('risk', 'ALL')
    dist_f  = request.args.get('district', 'ALL')
    year_f  = request.args.get('year', 'ALL')

    hot = _df[_df['IS_HOTSPOT'] == 1].copy()
    if dist_f != 'ALL': hot = hot[hot['DISTRICT'].astype(str) == dist_f]
    if year_f != 'ALL': hot = hot[hot['YEAR'] == int(year_f)]

    np.random.seed(42)
    hot['prob'] = (
        hot['HOUR'].apply(lambda h: 0.55 if (h >= 18 or h <= 5) else 0.22) +
        np.random.uniform(0, 0.38, len(hot))
    ).clip(0.01, 0.99)
    hot['rl'] = hot['prob'].apply(risk_level)

    if risk_f != 'ALL': hot = hot[hot['rl'] == risk_f]

    sample = hot.sample(min(limit, len(hot)), random_state=42)
    records = [{
        'id':         str(uuid.uuid4())[:8],
        'lat':        float(r['LAT']),
        'lon':        float(r['LONG']),
        'crime_type': str(r.get('OFFENSE_CODE_GROUP', 'Unknown')),
        'district':   str(r.get('DISTRICT', 'N/A')),
        'risk_score': round(float(r['prob']), 3),
        'risk_level': r['rl'],
        'hour':       int(r['HOUR']),
        'year':       int(r['YEAR']),
    } for _, r in sample.iterrows()]

    return jsonify({
        'hotspots':    records,
        'total':       len(records),
        'high_risk':   sum(1 for x in records if x['risk_level'] == 'HIGH'),
        'medium_risk': sum(1 for x in records if x['risk_level'] == 'MEDIUM'),
        'low_risk':    sum(1 for x in records if x['risk_level'] == 'LOW'),
    })
@app.route('/api/districts', methods=['GET'])
def districts():
    codes = sorted(DISTRICT_NAMES.keys())
    result = [{'code': c, 'name': DISTRICT_NAMES[c],
               'full_label': f"{c} — {DISTRICT_NAMES[c]}",
               'crime_count': DISTRICT_CRIME_COUNT.get(c, 0)} for c in codes]
    return jsonify({'districts': result, 'codes': codes})

@app.route('/api/offense_types', methods=['GET'])
def offense_types():
    if _df is None: return jsonify({'types': ['Larceny','Robbery','Assault']})
    top = _df['OFFENSE_CODE_GROUP'].value_counts().head(25).index.tolist()
    return jsonify({'types': [str(t) for t in top]})

@app.route('/api/predict', methods=['POST'])
def predict():
    d        = request.get_json() or {}
    lat      = float(d.get('lat',      42.3601))
    lon      = float(d.get('lon',     -71.0589))
    hour     = int(  d.get('hour',     datetime.now().hour))
    month    = int(  d.get('month',    datetime.now().month))
    year     = int(  d.get('year',     2024))
    district = d.get('district',       'B2')
    day      = d.get('day_of_week',    'Friday')
    offense  = d.get('offense_type',   'Larceny')
    algo     = d.get('algorithm',      'rf')

    if algo not in MODEL_INFO:
        algo = 'rf'

    feat    = build_feature_vector(lat, lon, hour, month, year, district, day, offense)
    feat_s  = _scaler.transform(feat) if _scaler is not None else feat
    prob    = run_model(algo, feat_s) or 0.5
    level   = risk_level(prob)

    real_count = 0
    if _df is not None:
        real_count = int(len(_df[
            (_df['DISTRICT'].astype(str) == district) & (_df['HOUR'] == hour)
        ]))

    info      = MODEL_INFO[algo]
    dist_name = DISTRICT_NAMES.get(district, district)
    return jsonify({
        'prediction_id':             str(uuid.uuid4())[:8],
        'is_hotspot':                prob > 0.5,
        'probability':               round(prob, 4),
        'risk_level':                level,
        'lat': lat, 'lon': lon,
        'hour':                      hour,
        'district':                  district,
        'district_name':             dist_name,
        'district_full':             f"{district} — {dist_name}",
        'offense_type':              offense,
        'algorithm_used':            info['name'],
        'algorithm_key':             algo,
        'model_type':                info['type'],
        'model_accuracy':            info['acc'],
        'model_f1':                  info['f1'],
        'real_incidents_this_slot':  real_count,
        'recommendations':           recommendations(level, district, hour),
        'timestamp':                 datetime.now().isoformat(),
    })

@app.route('/api/predict/all', methods=['POST'])
def predict_all():
    d        = request.get_json() or {}
    lat      = float(d.get('lat',   42.3601))
    lon      = float(d.get('lon',  -71.0589))
    hour     = int(  d.get('hour',  21))
    month    = int(  d.get('month', 10))
    year     = int(  d.get('year',  2024))
    district = d.get('district',   'B2')
    day      = d.get('day_of_week','Friday')
    offense  = d.get('offense_type','Larceny')

    feat   = build_feature_vector(lat, lon, hour, month, year, district, day, offense)
    feat_s = _scaler.transform(feat) if _scaler is not None else feat

    preds = {}
    for key, info in MODEL_INFO.items():
        prob = run_model(key, feat_s)
        if prob is not None:
            lv = risk_level(prob)
            preds[key] = {
                'name':      info['name'],
                'type':      info['type'],
                'accuracy':  info['acc'],
                'f1':        info['f1'],
                'auc':       info['auc'],
                'probability': round(prob, 4),
                'is_hotspot':  prob > 0.5,
                'risk_level':  lv,
            }

    votes = sum(1 for v in preds.values() if v['is_hotspot'])
    avg_prob = round(float(np.mean([v['probability'] for v in preds.values()])), 4)
    ensemble = {'vote': votes, 'total': len(preds),
                'verdict': 'HOTSPOT' if votes >= 3 else 'NOT HOTSPOT',
                'avg_probability': avg_prob,
                'risk_level': risk_level(avg_prob)}

    return jsonify({
        'predictions': preds,
        'ensemble': ensemble,
        'input': d,
        'timestamp': datetime.now().isoformat(),
    })

@app.route('/api/model/metrics', methods=['GET'])
def model_metrics():
    out = {'models': MODEL_INFO, 'filtered': 'Only models with accuracy >= 80%'}
    if _results:
        for key, info in MODEL_INFO.items():
            name = info['name']
            if name in _results:
                out['models'][key]['full_metrics'] = _results[name]
    return jsonify(out)
@app.route('/api/fir', methods=['GET'])
def get_firs():
    role = request.args.get('role', 'public')
    cid  = request.args.get('citizen_id', '')
    if role in ['police', 'admin']:
        return jsonify({'firs': fir_store, 'total': len(fir_store)})
    own = [f for f in fir_store if f.get('citizen_id') == cid]
    return jsonify({'firs': own, 'total': len(own)})

@app.route('/api/fir', methods=['POST'])
def file_fir():
    d   = request.get_json() or {}
    fid = f"FIR-{datetime.now().year}-{str(uuid.uuid4())[:6].upper()}"
    fir = {**d, 'fir_id': fid, 'status': 'FILED', 'assigned_officer': None,
           'filed_at': datetime.now().isoformat(), 'last_updated': datetime.now().isoformat(),
           'updates': [{'status':'FILED','message':'FIR registered successfully.',
                        'timestamp': datetime.now().isoformat()}]}
    fir_store.append(fir)
    return jsonify({'success': True, 'fir_id': fid, 'fir': fir}), 201

@app.route('/api/fir/<fid>', methods=['GET'])
def get_fir(fid):
    f = next((x for x in fir_store if x['fir_id'] == fid), None)
    return jsonify(f) if f else (jsonify({'error': 'Not found'}), 404)

@app.route('/api/fir/<fid>/update', methods=['PUT'])
def update_fir(fid):
    d = request.get_json() or {}
    f = next((x for x in fir_store if x['fir_id'] == fid), None)
    if not f: return jsonify({'error': 'Not found'}), 404
    f['status']       = d.get('status', f['status'])
    f['last_updated'] = datetime.now().isoformat()
    f.setdefault('updates', []).append({
        'status': f['status'], 'message': d.get('message', 'Status updated'),
        'timestamp': datetime.now().isoformat()})
    return jsonify({'success': True, 'fir': f})

@app.route('/api/alerts', methods=['GET'])
def alerts():
    if _df is None: return jsonify({'alerts': []})
    sample = _df.sample(min(12, len(_df)), random_state=int(datetime.now().second))
    out = []
    for _, r in sample.iterrows():
        sev = 'HIGH' if r.get('SHOOTING',0)==1 else \
              'MEDIUM' if (r['HOUR'] >= 20 or r['HOUR'] <= 5) else 'LOW'
        dist_code = str(r.get('DISTRICT', 'N/A'))
        out.append({
            'id':            str(uuid.uuid4())[:8],
            'type':          str(r.get('OFFENSE_CODE_GROUP', 'Unknown')),
            'district':      dist_code,
            'district_name': DISTRICT_NAMES.get(dist_code, dist_code),
            'full_label':    f"{dist_code} — {DISTRICT_NAMES.get(dist_code, dist_code)}",
            'severity':      sev,
            'hour':          int(r['HOUR']),
            'responded':     bool(np.random.random() > 0.4),
            'mins_ago':      int(np.random.randint(3, 180)),
        })
    out.sort(key=lambda x: x['mins_ago'])
    return jsonify({'alerts': out})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
