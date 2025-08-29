from __future__ import annotations
import json
from typing import List, Dict
from ..common.config import CONFIG

def build_mv_v2(merged_candidates_file: str, out_path: str) -> None:
    with open(merged_candidates_file, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            mc = json.loads(line)
            top = mc.get('candidates', [])[:1]
            mv = {
                'mention_id': mc.get('mention_id'),
                'surface': mc.get('surface'),
                'span': mc.get('span'),
                'selected': top[0] if top else None,
                'confident': mc.get('confident_singleton', False)
            }
            fout.write(json.dumps(mv, ensure_ascii=False)+'\n')
