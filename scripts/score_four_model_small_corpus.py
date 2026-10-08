"""Official NAVSIM v1 scoring on the fixed independent development scenes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import numpy as np
from score_lpwm_drivor_navtest import score_scene

ROOT=Path(__file__).resolve().parents[1]


def main(arguments):
    import pandas as pd
    manifest=json.loads((ROOT/'outputs/four_model_small_corpus_v1/corpus/manifest.json').read_text())
    records={r['token']:r for r in manifest['records'] if r['study_split']=='dev'}
    predictions=np.load(arguments.predictions)
    tokens=predictions['tokens'].tolist()
    assert set(tokens)==set(records) and len(tokens)==1024
    jobs=[(token,poses,records[token]['metric_cache_file'])
          for token,poses in zip(tokens,predictions['trajectories'])]
    with ProcessPoolExecutor(max_workers=arguments.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(score_scene,jobs,chunksize=8))
    ego=np.load(ROOT/'outputs/four_model_small_corpus_v1/corpus/ego.npy',mmap_mode='r')
    for row in rows:
        record=records[row['token']]
        row['recording_group']=record['recording_group']
        row['command']=int(ego[record['cache_row'],7:11].argmax())
    frame=pd.DataFrame(rows)
    frame.to_csv(arguments.predictions.with_suffix('.scores.csv'),index=False)
    summary=dict(count=len(rows),benchmark='NAVSIM v1 independent development subset',full_navtest=False,
        pdms=float(frame['score'].mean()*100),failed=0,
        by_command=frame.groupby('command')['score'].agg(['count','mean']).to_dict('index'),
        subscores=frame.select_dtypes('number').drop(columns=['command']).mean().to_dict(),
        train_dev_recording_overlap=0)
    arguments.predictions.with_suffix('.pdms.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8); main(parser.parse_args())
