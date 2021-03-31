from dicedps.plot.common import *

prop_list[0]
df=pd.read_csv('cdice-sa-params.csv',sep='\t',index_col=0)
df = df[df.Include==1]

emi_setup = {
'low': {
    'Low':['pop0','popadj','popasym','q0','ga0','gsigma1','dsig','e0'],
    'High':['dela',]
},

    'high': {
        'High': ['pop0', 'popadj', 'popasym', 'q0', 'ga0', 'gsigma1', 'dsig', 'e0'],
        'Low': ['dela', ]
    },

    'nominal': {}
}

a = {}
for emi in ['low','nominal','high']:
    a[emi] = {}
    for k, vlist in emi_setup[emi].items():
        for v in vlist:
            a[emi][v] = df.loc[v,k]

0.134*1.2
a['low']
pd.DataFrame({x:df[x]/df['Nominal']-1. for x in ['Low','High']}).plot(kind='bar')

fig, ax = plt.subplots(1,1,figsize=(6,4))
for x,y in a.items():
    dc = h.args2dice(f'-m time -o greg4 -u 1 -w 1 -e 2100 -C ka18', setup=y)
    dc0 = dc.run(np.zeros(17))
    dc0.EIND.iloc[:,0].plot(label=x,legend=False,ax=ax)
