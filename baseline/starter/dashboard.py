import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Hallucination Regime Lab",layout="wide")
root=Path(__file__).parent/'research_results'
st.title("Bayesian Hallucination Regime Lab")
st.caption("Synthetic research experiment · not measured LLM performance")
if not (root/'report.json').exists():
    st.error("Run research.py first.");st.stop()
r=json.loads((root/'report.json').read_text());df=pd.read_csv(root/'predictions.csv')
a,b,c=st.columns(3)
for col,key in zip([a,b,c],['bayes','em','static']):col.metric(key.upper()+" onset AUROC",f"{r['metrics'][key]['auc']:.3f}")
st.info(r['limitations'])
tabs=st.tabs(["Trajectory","Transitions","Feature attribution","Evaluation","MCMC"])
with tabs[0]:
    d=df[df.dossier==st.selectbox("Held-out dossier",sorted(df.dossier.unique()))]
    fig=go.Figure()
    fig.add_scatter(x=d.turn,y=d.upper,line=dict(width=0),showlegend=False)
    fig.add_scatter(x=d.turn,y=d.lower,fill='tonexty',line=dict(width=0),name="95% posterior interval")
    for k in ['bayes','em','static','next_risk']:fig.add_scatter(x=d.turn,y=d[k],name=k)
    for t in d.loc[d.target==1,'turn']:fig.add_vline(x=t,line_color='red',line_dash='dot')
    if r['threshold'] is not None:fig.add_hline(y=r['threshold'],line_dash='dash')
    fig.update_layout(yaxis_range=[0,1],xaxis_title="Turn",yaxis_title="Probability")
    st.plotly_chart(fig,use_container_width=True);st.dataframe(d)
with tabs[1]:
    A=np.load(root/'transition.npy');names=['Grounded','Uncertain','Hallucinating']
    st.plotly_chart(px.imshow(A.mean(0),x=names,y=names,text_auto='.3f',zmin=0,zmax=1,labels={'x':'Destination','y':'Origin'}),use_container_width=True)
    st.dataframe(pd.DataFrame([{'origin':names[i],'destination':names[j],'mean':A[:,i,j].mean(),'lower':np.quantile(A[:,i,j],.025),'upper':np.quantile(A[:,i,j],.975)} for i in range(3) for j in range(3)]))
with tabs[2]:
    st.caption("Risk difference when omitting one feature across the entire observed prefix; not causal.")
    st.plotly_chart(px.bar(d,x='turn',y=['logp_contribution','entropy_contribution','cosine_contribution'],barmode='group'),use_container_width=True)
with tabs[3]:
    from sklearn.metrics import roc_curve,precision_recall_curve
    e=df[df.at_risk];roc=go.Figure();pr=go.Figure()
    for k in ['bayes','em','static']:
        f,t,_=roc_curve(e.target,e[k]);p,rec,_=precision_recall_curve(e.target,e[k])
        roc.add_scatter(x=f,y=t,name=k);pr.add_scatter(x=rec,y=p,name=k)
    roc.update_layout(xaxis_title='False positive rate',yaxis_title='True positive rate');pr.update_layout(xaxis_title='Recall',yaxis_title='Precision')
    l,rr=st.columns(2);l.plotly_chart(roc,use_container_width=True);rr.plotly_chart(pr,use_container_width=True)
    st.json(r['metrics'])
with tabs[4]:
    st.json({k:v for k,v in r.items() if 'rhat' in k or 'ess' in k or 'chain' in k})
    p=np.load(root/'posterior.npz');k=int(p['mapping'][2]);tr=p['A'][:,:,k,k]
    st.plotly_chart(px.line(pd.DataFrame(tr.T)),use_container_width=True)
