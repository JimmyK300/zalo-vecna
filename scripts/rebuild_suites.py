"""Recreate benchmark ID manifests locally from the original train release.

Only seeds and selection procedures are distributed in Git; question and
business identifiers stay in ignored local JSON manifests.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datamate_bench.suite import make_suite


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def combine(output, inputs, purpose):
    first = read(inputs[0])
    items = [{'id': item['id']} for path in inputs for item in read(path)['items']]
    if len(items) != len({item['id'] for item in items}):
        raise ValueError('Overlapping selection stages')
    value = {'version': 1, 'purpose': purpose,
             **{k: v for k, v in first.items() if k.endswith('_sha256')}, 'items': items}
    output.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=ROOT.parent / 'train')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'benchmarks')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='datamate-suites-') as temp:
        base = Path(temp)
        first, second = base/'train10.json', base/'unseen10.json'
        make_suite(args.train, first, 20261102, 10)
        make_suite(args.train, second, 20261103, 10, first, True)
        dev20 = base/'development20.json'
        combine(dev20, [first, second], 'Development regression')
        third = base/'business20.json'
        make_suite(args.train, third, 20261104, 20, dev20, True)
        dev40 = base/'development40.json'
        combine(dev40, [dev20, third], 'Development regression')
        fourth = base/'question30.json'
        make_suite(args.train, fourth, 20261105, 30, dev40)
        dev70 = base/'development70.json'
        combine(dev70, [dev40, fourth], 'Development regression')
        fifth = base/'question20.json'
        make_suite(args.train, fifth, 20261106, 20, dev70)
        dev90 = base/'development90.json'
        combine(dev90, [dev70, fifth], 'Development regression; all 90 inspected')
        validation = base/'validation200_random.json'
        subprocess.run([sys.executable, str(ROOT/'scripts/select_random_suite.py'),
                        '--train', str(args.train.resolve()), '--exclude-suite', str(dev90),
                        '--output', str(validation), '--size', '200', '--seed', '20261107'], check=True)
        value = read(validation)
        value['excluded_suite'] = 'benchmarks/development90.json'
        validation.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
        combine(base/'seen290.json', [dev90, validation], 'Exclusion only; not a held-out test')
        def uniform(name, excluded, size, seed):
            subprocess.run([sys.executable,str(ROOT/'scripts/select_random_suite.py'),
                            '--train',str(args.train.resolve()),'--exclude-suite',str(base/excluded),
                            '--output',str(base/name),'--size',str(size),'--seed',str(seed)],check=True)
        uniform('normalization12_random.json','seen290.json',12,2026110301)
        combine(base/'seen302.json',[base/'seen290.json',base/'normalization12_random.json'],'Exclusion only; inspected questions')
        value=read(base/'seen302.json')
        value['items'].sort(key=lambda x:x['id'])
        (base/'seen302.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        uniform('sql100_random.json','seen302.json',100,2026110302)
        value=read(base/'sql100_random.json');value['items']=value['items'][:20]
        value['purpose']='Prespecified first 20 of random 100: generated SQL vs hybrid'
        (base/'sql20_paired.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        combine(base/'seen402.json',[base/'seen302.json',base/'sql100_random.json'],'Exclusion only; inspected questions')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ['train10.json', 'development90.json', 'validation200_random.json', 'seen290.json',
                     'normalization12_random.json','seen302.json','sql100_random.json','sql20_paired.json','seen402.json']:
            source, target = base/name, args.output_dir/name
            if target.exists():
                old, new = read(target), read(source)
                if ([x['id'] for x in old['items']] != [x['id'] for x in new['items']] or
                        any(old.get(k) != new.get(k) for k in ['questions_sha256','answers_sha256','rulebook_sha256'])):
                    raise ValueError(f'Existing selection differs: {target}; use another output directory')
            else:
                target.write_bytes(source.read_bytes())
            print(f'{name}: {len(read(source)["items"])} local IDs reproduced', flush=True)


if __name__ == '__main__':
    main()
