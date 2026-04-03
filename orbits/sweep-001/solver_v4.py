"""
Circle packing solver v4: lightweight basin hopping.

Key change: basin hops use only repair_and_grow + local_search (no L-BFGS-B).
L-BFGS-B only used for initial multi-start and final polish.
"""

import numpy as np
from scipy.optimize import minimize
import json
import os
import time
import sys


def joint_penalty_and_grad(vec, n, mu):
    xs = vec[:n]; ys = vec[n:2*n]; rs = vec[2*n:]
    obj = -np.sum(rs)
    gx = np.zeros(n); gy = np.zeros(n); gr = -np.ones(n)

    v = np.maximum(0, rs - xs); obj += mu*np.sum(v**2); gx -= 2*mu*v; gr += 2*mu*v
    v = np.maximum(0, xs+rs-1); obj += mu*np.sum(v**2); gx += 2*mu*v; gr += 2*mu*v
    v = np.maximum(0, rs - ys); obj += mu*np.sum(v**2); gy -= 2*mu*v; gr += 2*mu*v
    v = np.maximum(0, ys+rs-1); obj += mu*np.sum(v**2); gy += 2*mu*v; gr += 2*mu*v

    dx = xs[:,None]-xs[None,:]; dy = ys[:,None]-ys[None,:]
    dist = np.sqrt(np.maximum(dx**2+dy**2, 1e-30))
    rsum = rs[:,None]+rs[None,:]
    mask = np.triu(np.ones((n,n),dtype=bool),k=1)
    olap = np.maximum(0, rsum-dist)*mask
    obj += mu*np.sum(olap**2)

    inv_d = np.where(dist>1e-15, 1.0/dist, 0.0)
    of = 2*mu*olap*inv_d
    gx += np.sum(of*dx,axis=1)-np.sum(of*dx,axis=0)
    gy += np.sum(of*dy,axis=1)-np.sum(of*dy,axis=0)
    or_ = 2*mu*olap
    gr -= np.sum(or_,axis=1)+np.sum(or_,axis=0)

    nv = np.maximum(0,-rs); obj += 100*mu*np.sum(nv**2); gr -= 200*mu*nv
    return obj, np.concatenate([gx,gy,gr])


def lbfgsb_optimize(xs, ys, rs, n, max_iter=300):
    vec = np.concatenate([xs,ys,rs])
    bds = [(1e-4,1-1e-4)]*n + [(1e-4,1-1e-4)]*n + [(1e-6,0.5)]*n
    for mu in [10, 100, 1000, 10000, 100000]:
        r = minimize(lambda v: joint_penalty_and_grad(v,n,mu), vec, jac=True,
                     method='L-BFGS-B', bounds=bds,
                     options={'maxiter':max_iter,'ftol':1e-15,'gtol':1e-12})
        vec = r.x
    return vec[:n], vec[n:2*n], vec[2*n:]


def repair_and_grow(xs, ys, rs, n):
    for i in range(n):
        rs[i] = min(rs[i], xs[i], 1-xs[i], ys[i], 1-ys[i])
    for _ in range(200):
        ok = True
        for i in range(n):
            for j in range(i+1,n):
                d = np.sqrt((xs[i]-xs[j])**2+(ys[i]-ys[j])**2)
                o = rs[i]+rs[j]-d
                if o > 1e-13:
                    t = rs[i]+rs[j]
                    if t > 0:
                        s = o+1e-14
                        rs[i] -= s*rs[i]/t; rs[j] -= s*rs[j]/t
                    ok = False
        if ok: break
    rs = np.maximum(rs, 1e-15)
    for _ in range(10):
        ch = False
        for i in range(n):
            mr = min(xs[i],1-xs[i],ys[i],1-ys[i])
            for j in range(n):
                if j!=i:
                    d = np.sqrt((xs[i]-xs[j])**2+(ys[i]-ys[j])**2)
                    mr = min(mr, d-rs[j])
            if mr > rs[i]+1e-15:
                rs[i] = mr; ch = True
        if not ch: break
    return rs


def local_search(xs, ys, rs, n, step=0.01, gp=9, iters=3):
    for it in range(iters):
        imp = False
        order = np.random.permutation(n)
        offs = np.linspace(-step, step, gp)
        for i in order:
            bx, by, bg = xs[i], ys[i], 0.0
            old_r = rs[i]
            for dx in offs:
                for dy in offs:
                    if dx==0 and dy==0: continue
                    nx, ny = xs[i]+dx, ys[i]+dy
                    if nx<0.002 or nx>0.998 or ny<0.002 or ny>0.998: continue
                    mr = min(nx, 1-nx, ny, 1-ny)
                    for j in range(n):
                        if j==i: continue
                        d = np.sqrt((nx-xs[j])**2+(ny-ys[j])**2)
                        mr = min(mr, d-rs[j])
                    if mr <= 0: continue
                    g = mr - old_r
                    if g > bg + 1e-15:
                        bx, by, bg = nx, ny, g
                        br = mr
            if bg > 1e-15:
                xs[i], ys[i], rs[i] = bx, by, br
                imp = True
        if not imp: break
        step *= 0.6
    return xs, ys, rs


def hex_init(n, noise=0.0):
    side = int(np.ceil(np.sqrt(n*2/np.sqrt(3))))+1
    pts = []
    for row in range(side+3):
        for col in range(side+3):
            x = (col+0.5*(row%2)+0.5)/(side+2)
            y = (row*np.sqrt(3)/2+0.5)/(side+2)
            if 0.02<x<0.98 and 0.02<y<0.98: pts.append((x,y))
    pts = np.array(pts)
    if len(pts)>=n:
        sel = [len(pts)//2]
        for _ in range(n-1):
            md = np.min([np.sum((pts-pts[s])**2,axis=1) for s in sel],axis=0)
            md[sel]=-1; sel.append(np.argmax(md))
        pts = pts[sel]
    else:
        pts = np.vstack([pts, np.random.uniform(0.05,0.95,(n-len(pts),2))])
    xs, ys = pts[:n,0].copy(), pts[:n,1].copy()
    if noise>0:
        xs += np.random.randn(n)*noise; ys += np.random.randn(n)*noise
        xs = np.clip(xs,0.02,0.98); ys = np.clip(ys,0.02,0.98)
    return xs, ys


def grid_init(n, noise=0.0):
    side = int(np.ceil(np.sqrt(n)))
    pts = [((i+0.5)/side,(j+0.5)/side) for i in range(side) for j in range(side)]
    pts = np.array(pts[:n])
    xs, ys = pts[:,0].copy(), pts[:,1].copy()
    if noise>0:
        xs += np.random.randn(n)*noise; ys += np.random.randn(n)*noise
        xs = np.clip(xs,0.02,0.98); ys = np.clip(ys,0.02,0.98)
    return xs, ys


def random_init(n):
    m = 0.4/np.sqrt(n)
    return np.random.uniform(m,1-m,n), np.random.uniform(m,1-m,n)


def validate(xs, ys, rs, n, tol=1e-10):
    mv = 0.0
    for i in range(n):
        mv = max(mv, rs[i]-xs[i], xs[i]+rs[i]-1, rs[i]-ys[i], ys[i]+rs[i]-1)
    for i in range(n):
        for j in range(i+1,n):
            d = np.sqrt((xs[i]-xs[j])**2+(ys[i]-ys[j])**2)
            mv = max(mv, rs[i]+rs[j]-d)
    return mv<=tol, np.sum(rs), mv


def solve_n(n, time_budget=120, verbose=True):
    if verbose:
        print(f"\n{'='*60}\nSolving n={n} (budget={time_budget}s)\n{'='*60}")

    bxs=bys=brs=None; bm=-1; t0=time.time()
    inits = ['hex','grid','random']

    # Phase 1: Multi-start with L-BFGS-B (20% budget)
    if verbose: print(f"\nPhase 1: Multi-start")
    s = 0
    while time.time()-t0 < time_budget*0.20:
        init = inits[s%3]
        noise = 0.003*(s//3)
        try:
            if init=='hex': xs,ys = hex_init(n,noise)
            elif init=='grid': xs,ys = grid_init(n,noise)
            else: xs,ys = random_init(n)
            rs = np.full(n, 0.35/np.sqrt(n))
            xs,ys,rs = lbfgsb_optimize(xs,ys,rs,n,max_iter=300)
            rs = repair_and_grow(xs,ys,rs,n)
            xs,ys,rs = local_search(xs,ys,rs,n,step=0.02,gp=7,iters=3)
            rs = repair_and_grow(xs,ys,rs,n)
            v,m,_ = validate(xs,ys,rs,n)
            if v and m>bm:
                bm=m; bxs,bys,brs=xs.copy(),ys.copy(),rs.copy()
                if verbose: print(f"  Start {s:3d}: {m:.10f} BEST")
        except: pass
        s += 1
    if verbose: print(f"Phase 1: {s} starts, best={bm:.10f} ({time.time()-t0:.1f}s)")
    if bxs is None: return None

    # Phase 2: Lightweight basin hopping (65% budget)
    # Only repair+local_search per hop - NO L-BFGS-B
    if verbose: print(f"\nPhase 2: Lightweight basin hopping")
    cxs,cys,crs = bxs.copy(),bys.copy(),brs.copy()
    cm = bm; stale = 0; hop = 0
    while time.time()-t0 < time_budget*0.85:
        txs,tys = cxs.copy(),cys.copy()
        trs_base = crs.copy()

        if stale > 50:
            txs,tys = bxs.copy(),bys.copy()
            txs += np.random.randn(n)*0.06; tys += np.random.randn(n)*0.06
            stale = 0
        elif stale > 20:
            k = np.random.randint(n//3, n//2+1)
            idx = np.random.choice(n,k,replace=False)
            txs[idx] += np.random.randn(k)*0.04
            tys[idx] += np.random.randn(k)*0.04
        else:
            st = np.random.randint(7)
            if st==0:
                k = np.random.randint(1,max(2,n//5))
                idx = np.random.choice(n,k,replace=False)
                s2 = 0.01+0.03*np.random.rand()
                txs[idx] += np.random.randn(k)*s2; tys[idx] += np.random.randn(k)*s2
            elif st==1:
                i,j = np.random.choice(n,2,replace=False)
                txs[i],txs[j] = txs[j],txs[i]; tys[i],tys[j] = tys[j],tys[i]
            elif st==2:
                s2 = 0.005+0.015*np.random.rand()
                txs += np.random.randn(n)*s2; tys += np.random.randn(n)*s2
            elif st==3:
                w = np.argmin(crs)
                txs[w] = np.random.uniform(0.05,0.95)
                tys[w] = np.random.uniform(0.05,0.95)
            elif st==4:
                w2 = np.argsort(crs)[:min(3,n)]
                for w in w2:
                    txs[w] = np.random.uniform(0.05,0.95)
                    tys[w] = np.random.uniform(0.05,0.95)
            elif st==5:
                a = np.random.uniform(-0.3,0.3)
                k = np.random.randint(2,max(3,n//3))
                idx = np.random.choice(n,k,replace=False)
                cx2,cy2 = np.mean(txs[idx]),np.mean(tys[idx])
                ca,sa = np.cos(a),np.sin(a)
                for ii in idx:
                    ddx,ddy = txs[ii]-cx2,tys[ii]-cy2
                    txs[ii] = cx2+ca*ddx-sa*ddy; tys[ii] = cy2+sa*ddx+ca*ddy
            else:
                k = np.random.randint(1,max(2,n//4))
                idx = np.random.choice(n,k,replace=False)
                if np.random.rand()<0.5: txs[idx]=1.0-txs[idx]
                else: tys[idx]=1.0-tys[idx]

        txs = np.clip(txs,0.02,0.98); tys = np.clip(tys,0.02,0.98)

        # Lightweight: just repair + local search (fast!)
        trs = np.full(n, 0.35/np.sqrt(n))
        trs = repair_and_grow(txs,tys,trs,n)
        txs,tys,trs = local_search(txs,tys,trs,n,step=0.025,gp=7,iters=3)
        trs = repair_and_grow(txs,tys,trs,n)

        v,met,_ = validate(txs,tys,trs,n)
        if v:
            if met > bm:
                bxs,bys,brs = txs.copy(),tys.copy(),trs.copy()
                bm = met; cxs,cys,crs = txs.copy(),tys.copy(),trs.copy()
                cm = met; stale = 0
                if verbose: print(f"    BH {hop:4d}: {met:.10f} BEST ({time.time()-t0:.0f}s)")
            else:
                d = met-cm
                if d>0 or np.random.rand()<np.exp(d/0.003):
                    cxs,cys,crs = txs.copy(),tys.copy(),trs.copy(); cm=met
                stale += 1
        else: stale += 1
        hop += 1

    if verbose: print(f"Phase 2: {hop} hops, best={bm:.10f} ({time.time()-t0:.1f}s)")

    # Phase 3: Fine local search only (no L-BFGS-B to avoid slowness)
    if verbose: print(f"\nPhase 3: Fine refinement")
    xs,ys,rs = bxs.copy(),bys.copy(),brs.copy()
    for step in [0.005,0.002,0.001,0.0005]:
        xs,ys,rs = local_search(xs,ys,rs,n,step=step,gp=7,iters=2)
        rs = repair_and_grow(xs,ys,rs,n)
        v,m,_ = validate(xs,ys,rs,n)
        if v and m>bm: bm=m; bxs,bys,brs=xs.copy(),ys.copy(),rs.copy()
        if verbose: print(f"    step={step:.4f}: {m:.10f}")

    elapsed = time.time()-t0
    if verbose: print(f"\nFinal n={n}: {bm:.10f} ({elapsed:.1f}s)")
    return bxs,bys,brs,bm


def save_solution(xs,ys,rs,n,filepath):
    circles = [[float(xs[i]),float(ys[i]),float(rs[i])] for i in range(n)]
    with open(filepath,'w') as f:
        json.dump({"circles":circles,"n":n,"metric":float(np.sum(rs))},f,indent=2)
    print(f"Saved {filepath}")


if __name__ == "__main__":
    targets = {24:2.530, 25:2.587, 27:2.685, 29:2.790, 31:2.889}
    if len(sys.argv)>1:
        n_values = [int(sys.argv[1])]
        budget = int(sys.argv[2]) if len(sys.argv)>2 else 120
    else:
        n_values = [29,31,24,25,27]
        budget = 120

    out_dir = os.path.dirname(os.path.abspath(__file__))
    results = {}
    for nv in n_values:
        np.random.seed(42+nv)
        sota = targets.get(nv,0)
        result = solve_n(nv, time_budget=budget)
        if result:
            xs,ys,rs,met = result
            fp = os.path.join(out_dir, f"solution_n{nv}.json")
            save_solution(xs,ys,rs,nv,fp)
            results[nv] = met
            print(f"\n>>> n={nv}: {met:.10f}  SOTA={sota:.3f}  ({met/sota*100:.1f}%)")
        else: results[nv]=0

    print("\n"+"="*60+"\nSUMMARY\n"+"="*60)
    for nv in n_values:
        s=targets.get(nv,0); m=results.get(nv,0)
        print(f"  n={nv}: {m:.10f}  SOTA={s:.3f}  ({m/s*100:.1f}%)" if s else f"  n={nv}: {m:.10f}")
