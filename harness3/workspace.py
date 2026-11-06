"""Build one business's DuckDB workspace from a DataMate data folder, using the repository's datamate loader.

The loader's cleaned/derived tables (e.g. orders, lines, calendar) are exported; nothing else (no recipes,
profiles or helpers) reaches the model. Cached as <cache>/<business_id>.duckdb + .docs.txt.
"""
from pathlib import Path


def build(data_dir, business_id, cache_dir):
    import duckdb
    import pandas as pd
    from datamate.io_contract import find_dataset_dir, load_rulebook
    from datamate.rules import active_rules
    from datamate.workspace import load_workspace

    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    db = cache_dir / f'{business_id}.duckdb'
    docs_path = cache_dir / f'{business_id}.docs.txt'
    if db.exists() and docs_path.exists():
        return db, docs_path.read_text(encoding='utf8')
    shared = {}
    if (data_dir / 'calendar.csv').exists():
        shared['calendar'] = pd.read_csv(data_dir / 'calendar.csv')
    ws = load_workspace(find_dataset_dir(data_dir, business_id), shared=shared,
                        rules=active_rules(load_rulebook(data_dir)))
    tmp = db.with_suffix('.building.duckdb')
    if tmp.exists():
        tmp.unlink()
    con = duckdb.connect(str(tmp), config={'threads': 1, 'memory_limit': '512MB'})
    try:
        for name, table in ws.tables.items():
            con.execute('SET pandas_analyze_sample = ' + str(max(1000, len(table.df))))
            con.register('_source', table.df)
            con.execute('CREATE TABLE "' + name.replace('"', '""') + '" AS SELECT * FROM _source')
            con.unregister('_source')
    finally:
        con.close()
    tmp.replace(db)
    docs = '\n\n'.join(f'{name}\n{text}' for name, text in ws.docs.items())
    docs_path.write_text(docs, encoding='utf8')
    return db, docs
