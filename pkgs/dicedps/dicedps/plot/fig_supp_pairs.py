from dicedps.plot.common import *
import string

df = load_merged(orient='min')
o2decoffset = {o.o_min_mean2degyears_lab:1}
dfround = (df
           .round(
    {olab: ndecs-o2decoffset.get(olab, 0) for olab, ndecs in
         zip(o.oset2labs[last_oset], o.oset2decs(last_oset))
     }))
df2use = df  # or dfround

olabs = o.oset2labs[last_oset]
n = len(olabs)

ssplit = lambda s: s.replace('(','\n(')
sb.set_context('paper')
fig, axs = plt.subplots(3, 3, figsize=(7,6), dpi=dpi4supp)

for miu, p in zip([mdps, mtime], prop_list):
    dfcurr = df2use.loc[miu]
    counter = 0
    for i, row_obj in enumerate([o.o_min_cbgemitcost_lab, o.o_min_cbgedamcost_lab, o.o_min_loss_util_bge_lab]):
        for j, col_obj in enumerate([o.o_min_mean2degyears_lab, o.o_min_cbgemitcost_lab, o.o_min_cbgedamcost_lab]):
            curr_col_obj, curr_row_obj = col_obj, row_obj
            if i == 0 and j == 1:
                handles = [axs[i, j].scatter([], [], **p) for p in prop_list[:2]] + [
                    axs[i, j].scatter([], [], color='k', marker='D')]
                labels = [miu2lab[miu] for miu in [mdps, mtime]] + ['Ideal']
                axs[i, j].legend(handles, labels)
            if j > i:
                axs[i, j].axis('off')
                continue
            if i == 2:
                axs[i, j].set_xlabel(ssplit(obj2lab2[curr_col_obj]))
            if j == 0:
                axs[i,j].set_ylabel(ssplit(obj2lab2[curr_row_obj]))
            dfthinned = dfcurr[[curr_row_obj,curr_col_obj]].drop_duplicates()
            print(f'Len reduced from {dfcurr.shape[0]} to {dfthinned.shape[0]}')
            axs[i, j].scatter(dfthinned[curr_col_obj], dfthinned[curr_row_obj], s=1, label=miu2lab[miu], rasterized=True, **p)
            if miu == mtime:
                fbest = getattr(np, df.name)
                hdiam = axs[i, j].scatter(
                    fbest(dfthinned[curr_col_obj]),  # o.obj2fbest[col_obj](dfcurr[col_obj]),
                    fbest(dfthinned[curr_row_obj]),  # o.obj2fbest[row_obj](dfcurr[row_obj]),
                    marker='D', label=f'ideal', color='k')
            xbnds = np.array([dfthinned[curr_col_obj].min(),
                              dfthinned[curr_col_obj].max()])  # np.array([obj2bounds[col_obj][x] for x in ['min','max']])
            xoff = max(abs(xbnds)) / 40
            ybnds = np.array([dfthinned[curr_row_obj].min(),
                              dfthinned[curr_row_obj].max()])  # np.array([obj2bounds[row_obj][x] for x in ['min', 'max']])
            yoff = max(abs(ybnds)) / 40
            axs[i, j].set_xlim([min(xbnds) - xoff, max(xbnds) + xoff])
            axs[i, j].set_ylim([min(ybnds) - yoff, max(ybnds) + yoff])
            if miu == mdps:
                axs[i, j].text(1, 1, string.ascii_lowercase[counter], transform=axs[i, j].transAxes, weight='bold')
            counter += 1
sb.despine(fig)

axs[0,0].set_xlim([0,100])
fig.tight_layout()
savefig4paper(fig, 'supp_pairs')


df[olabs].groupby('miulab').describe().T