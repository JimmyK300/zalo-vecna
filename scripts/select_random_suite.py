"""Select fresh questions uniformly, without using reference types or answers."""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datamate_bench.__main__ import source_fingerprints
from datamate_bench.suite import digest, load_suite, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=ROOT.parent / 'train')
    parser.add_argument('--exclude-suite', type=Path, default=ROOT / 'benchmarks/development90.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; use a new filename to preserve the selected sample.')
    _, previous = load_suite(args.train, args.exclude_suite)
    excluded = {q['id'] for q in previous}
    pool = sorted((q for q in read_jsonl(args.train / 'questions.jsonl') if q['id'] not in excluded), key=lambda q: q['id'])
    if not 1 <= args.size <= len(pool):
        parser.error('Size must be positive and no larger than the eligible pool.')
    chosen = random.Random(args.seed).sample(pool, args.size)
    manifest = {
        'version': 1, 'purpose': 'uniform random unseen-question validation; businesses may overlap development',
        'seed': args.seed, 'sampling': f'random.Random(seed).sample(sorted(eligible, key=id), {args.size})',
        'eligible_questions': len(pool), 'excluded_suite': str(args.exclude_suite),
        'excluded_questions': len(excluded), 'source_sha256': source_fingerprints(),
        **{f'{name}_sha256': digest(args.train / f'{name}.jsonl') for name in ['questions', 'answers']},
        'rulebook_sha256': digest(args.train / 'rulebook.md'),
        'items': [{'id': q['id']} for q in chosen],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'questions': len(chosen),
                      'businesses': len({q['business_id'] for q in chosen}), 'labels_used_for_selection': False}))


if __name__ == '__main__':
    main()
