"""Check the saved records and print the capability appendix table values.

Ported from the ICLR supplement's ``verify_records.py``. The only changes are
the paths: the records now live in ``records/pipeline/`` at the repository root
and the checksum manifest is ``pipeline/SHA256SUMS`` (which covers the record
files only, not the whole archive). Run it from anywhere:

    python pipeline/verify_records.py

It performs no downloads and no model inference, and needs only the standard
library on Python 3.11 or later.
"""
from pathlib import Path
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECORDS = ROOT / 'records' / 'pipeline'
SUMS = HERE / 'SHA256SUMS'

def read(name):
    return json.loads((RECORDS / name).read_text())

def main():
    for line in SUMS.read_text().splitlines():
        expected, path = line.split('  ', 1)
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert actual == expected, f'Checksum mismatch: {path}'
    panel = read('capability_panel.json')['rows']
    maps = {x['key']: x for x in read('capability.json')['maps']}
    assert len(panel) == 10
    print('Model\tLayer\tLambda\tCounts\tRidge R2\tRidge top1\tIdentity+bias R2\tIdentity+bias top1')
    for row in panel:
        mapping = maps[row['key']]
        fit = read('fits/' + mapping['cell'] + '.json')
        layer = str(fit['layer_star'])
        result = fit['layers'][layer]
        assert fit['input_position'] == 'prompt_last'
        assert int(layer) == mapping['layer_star'] == row['layer']
        for value in (mapping['mapping_performance']['test_r2'], row['test_r2']):
            assert math.isclose(result['test_r2'], value, rel_tol=0, abs_tol=1e-12)
        ridge = result['knn_test']['ridge']['cosine']
        bias = result['knn_test']['identity_bias']['cosine']
        assert ridge['n_pool'] == row['n_test']
        assert math.isclose(ridge['chance_at_k']['1'], 1 / row['n_test'], abs_tol=1e-15)
        print(row['label'], layer, mapping['selected_lambda'], result['n'],
              f"{result['test_r2']:.3f}", f"{100*ridge['acc_at_k']['1']:.1f}",
              f"{result['floors_test_r2']['identity_bias']:.3f}",
              f"{100*bias['acc_at_k']['1']:.1f}", sep='\t')
    retrieval = read('retrieval.json')
    composition = retrieval['pool_composition']
    assert composition['n_targets'] + composition['n_distractors'] == retrieval['retrieval']['n_pool']
    assert math.isclose(retrieval['retrieval']['chance_at_1']['10000'], 1 / retrieval['retrieval']['n_pool'])
    print('PASS: checksums, all ten capability records, and retrieval-pool consistency.')

if __name__ == '__main__':
    main()
