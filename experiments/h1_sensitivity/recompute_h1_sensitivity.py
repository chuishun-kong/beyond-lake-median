"""Audit-only reaggregation of the public k=5 per-target median RER values.

Source: the public capacity_boundary_summary_nested.csv, viewed 2026-09-15.
The 72 k=5 RER values were transcribed from the rendered raw CSV, not inferred
from published confidence limits. Order follows the source file. This is not
model retraining or an independent recomputation of query-level predictions.
Run: python recompute_h1_sensitivity.py
Requires numpy. Results and the transcribed inputs are written beside this file.
"""
from pathlib import Path
import json
import numpy as np

SOURCE_URL = ('https://raw.githubusercontent.com/chuishun-kong/beyond-lake-median/'
 'main/experiments/capacity_boundary/results/capacity_boundary_summary_nested.csv')
# Each source site occupies 15 rows; k=5 MLP/PLSR/XGBoost are rows 4/9/14
# within each block (header at row 0 in the web rendering).
NAMES = ['Branched Oak Lake','Chaohu','Dianchi','Eagle Creek Reservoir','Erhai',
 'Garda','Geist Reservoir','High Rock Lake','Ibitinga Reservoir','Lake Chotkowskie',
 'Lake Erie','Lake Geneva','Lake Hume','Lake Jasie\u0144 Po\u0142udniowy',
 'Lake Jele\u0144','Lake Kasumigaura','Lake Kummerow','Lake Ob\u0142\u0119\u017ce',
 'Lake Peipsi','Lake Winnipeg','Lake \u0141ebsko','Mantova','Morse Reservoir','Taihu']
RER = {
'PLSR': [0.03014834870149115,0.2877843355846974,-0.8430363705896833,
0.11619345826313726,0.24344863028756494,-0.06353002233926469,
-0.1519155565228844,0.1727396482493191,0.3023436429108909,
0.3470963438992965,0.1227449722703373,0.20862379038778173,
0.26603763277996484,0.0552436528272207,0.18364298747283003,
-0.18139733471952418,-0.0550025905690853,0.22136234411306727,
0.07435299551360139,0.09790469465357715,-0.00950036411166915,
-0.010368238851781949,0.3656928137868951,0.2069009741722157],
'XGBoost': [0.25490958981356204,0.26988727755483055,-0.930019462222733,
-0.0460479078045938,0.10751075229124245,-0.14996225926718976,
-0.5099871624162464,0.17830960307131605,0.39396014921202954,
0.5857427806157649,0.3289160865783786,0.18277862319135485,
0.217481992600079,0.33228825745697405,0.14491994689477933,
0.18086437202229494,0.11876210218265351,0.3674291402356248,
0.2728182226308653,0.38051220585430134,0.296805724724423,
-0.1315689937966375,0.36962963891097694,0.3139102691010818],
'MLP': [0.417484097006778,0.3270445211809357,-0.684483525974247,
0.17911292123925684,0.12943592056623845,0.0272612549086137,
-0.29894955026594483,0.08574392472718664,0.4351382283207383,
0.5157733605004814,0.30274012716558857,0.284587360629859,
0.371610285223297,0.3655410140523472,0.26572561123814553,
0.12980194278845092,-0.0026545751847756,0.3147613373505517,
0.1672668405442497,0.31310285527356163,0.12946880509250455,
-0.1839373297644214,0.2488394048379462,0.3817055246579347]
}
EXPECTED = {'PLSR':[8.28,-2.47,16.93], 'XGBoost':[14.71,0.78,25.85],
            'MLP':[17.59,6.09,27.05]}

def intervals(x: np.ndarray, n_boot: int, seed: int=0) -> dict:
    rng = np.random.default_rng(seed)
    # Same target indices across learners preserve the paired target structure.
    means = x[rng.integers(0,len(x),size=(n_boot,len(x)))].mean(axis=1)
    return {'nominal_95':np.quantile(means,[.025,.975]).tolist(),
            'bonferroni_98_333333':np.quantile(means,[.05/(2*3),1-.05/(2*3)]).tolist()}

def main() -> None:
    out=Path(__file__).resolve().parent
    results={}
    for name, vals in RER.items():
        x=np.array(vals,dtype=float)*100
        assert len(x)==24 and np.isfinite(x).all()
        primary=intervals(x,10_000)
        check=[round(float(x.mean()),2), *[round(v,2) for v in primary['nominal_95']]]
        if check != EXPECTED[name]:
            raise ValueError(f'Manuscript check failed: {name}: {check}')
        results[name]={'mean_RER_percent':float(x.mean()),
            'median_RER_percent':float(np.median(x)),
            'primary_10000_seed0':primary,
            'post_hoc_100000_seed0':intervals(x,100_000),
            'MC_check_seed1':intervals(x,100_000,1),
            'MC_check_seed2':intervals(x,100_000,2)}
    payload={'source_url':SOURCE_URL,'source_view_date':'2026-09-15',
             'inputs_transcribed_from_rendered_raw_csv':True,
             'inferential_unit':'24 exact-site target units',
             'unchanged_primary':'per-target median RER, then equal-weight mean',
             'new_analysis_status':'post hoc sensitivity, not a replacement primary',
             'numpy_version':np.__version__,'results':results}
    (out/'h1_sensitivity_results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    inputs={'source_url':SOURCE_URL,'sites_in_source_order':NAMES,
            'source_lines_for_k5':{name:[i*15+d for i in range(24)] for name,d in [('MLP',4),('PLSR',9),('XGBoost',14)]},
            'median_RER_fraction':RER}
    (out/'h1_k5_public_input_values.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
