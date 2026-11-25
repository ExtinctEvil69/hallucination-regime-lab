"""Six-turn, source-grounded QA with known answers in independent fictional dossiers."""
import argparse
from collections import Counter
import json
import re
from pathlib import Path
import numpy as np

MODEL = 'mlx-community/Qwen3.8-27B-4bit'
FIELDS = ('depot','port','sector','relay')
def cosine(a,b):
    a,b = [Counter(re.findall(r'\w+', s.lower())) for s in (a,b)]
    return sum(v*b[k] for k,v in a.items()) / max(1,(sum(v*v for v in a.values())*sum(v*v for v in b.values()))**.5)

def dossier(i):
    rng = np.random.default_rng(41000+i)
    codes = [f'{a}-{b:02d}' for a in ('RED','TEAL','GOLD','BLUE') for b in range(20)]
    records = [dict(zip(FIELDS,rng.choice(codes,4,replace=False))) for _ in range(4)]
    names = [f'Unit {i:04d}-{s}' for s in 'ABCD']
    lines = [n+' | '+' | '.join(f'{f} {r[f]}' for f in FIELDS) for n,r in zip(names,records)]
    questions = []
    for t in range(6):
        j = int(rng.integers(4))
        f = str(rng.choice(FIELDS))
        if rng.random()<.22:
            f = 'archive'
        if t and rng.random()<.45:
            j = last
            subject = 'the unit in my previous question'
        else:
            subject = names[j]
        questions.append((f'What is the {f} code for {subject}?',records[j].get(f),names[j]))
        last = j
    return lines, questions

def grade(answer, expected):
    s = answer.strip().rstrip('.').upper()
    if s == 'UNKNOWN':
        return 'Grounded' if expected is None else 'Uncertain'
    if re.fullmatch(r'(RED|TEAL|GOLD|BLUE)-\d{2}',s):
        return 'Grounded' if s == expected else 'Hallucinating'
    return None  # review free-form replies against evidence

def main():
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    p = argparse.ArgumentParser()
    p.add_argument('--count',type=int,default=500)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--dir',type=Path,default=Path('rollouts'))
    p.add_argument('--model',default=MODEL)
    p.add_argument('--guard',type=Path,help='trained detector .npz for an intervention demo')
    args = p.parse_args()
    args.dir.mkdir(parents=True,exist_ok=True)
    guard = None
    if args.guard:
        from evaluate import load_guard
        guard = load_guard(args.guard)
    model,tok = load(args.model)
    sampler = make_sampler(temp=.7)
    for i in range(args.start,args.start+args.count):
        path = args.dir/f'{i:05d}.json'
        if path.exists():
            old = json.loads(path.read_text())
            if old['model'] != args.model or guard or old.get('guarded'):
                raise ValueError('Use a fresh directory for a different model or intervention.')
            continue
        mx.random.seed(73000+i)
        lines,questions = dossier(i)
        messages = [dict(role='system',content='Use only these records. Reply with one code or UNKNOWN if missing.\n'+'\n'.join(lines))]
        turns = []
        previous = None
        for q,expected,unit in questions:
            messages.append(dict(role='user',content=q))
            prompt = tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
            ids,features = [],[]
            eos = getattr(tok,'eos_token_ids', {tok.eos_token_id})
            for token,lp in generate_step(mx.array(prompt),model,max_tokens=32,sampler=sampler):
                token = int(token)
                if token in eos:
                    break
                ids.append(token)
                lp = lp.reshape(-1).astype(mx.float32)
                entropy = -mx.sum(mx.where(mx.isfinite(lp),mx.exp(lp)*lp,0))
                features.append([lp[token].item(),entropy.item()])
            answer = tok.decode(ids)
            if not ids:
                raise RuntimeError(f'Empty response at dossier {i}, turn {len(turns)+1}; inspect before continuing.')
            x = [*np.mean(features,axis=0).tolist(),max(cosine(answer,line) for line in lines)]
            row = dict(question=q,unit=unit,expected=expected,answer=answer,x=x,label=grade(answer,expected))
            turns.append(row)
            messages.append(dict(role='assistant',content=answer))
            if guard:
                h,center,scale,k,threshold = guard
                previous = h.step((np.array(x)-center)/scale,previous)
                row['risk'] = float(previous[k])
                if previous[k] >= threshold:
                    row['halted'] = True
                    break
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(id=i,model=args.model,source=lines,guarded=bool(guard),turns=turns),indent=2))
        tmp.replace(path)

if __name__ == '__main__':
    main()
