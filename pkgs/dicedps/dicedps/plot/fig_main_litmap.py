from dicedps.plot.common import *

fig, ax =plt.subplots(1, 1, figsize=(w1col, hhalf))

xs = [1,1,1,
      #1,
      4,4]
ys = [1,2,3,
      #4,
      1,3]
ls = ['Nordhaus 2017',
      'Keller et al. 2004',
      'Shayhegh and Thomas 2015',
#      'Lemoine and Traeger 2016',
      'Garner et al. 2016',
      'This study']
ps = [0,0,0,
      #0,
      1,1]

has = ['center', 'center']
xo = [0.05, -0.05]
yo = [-0.1, 0.1]
vas = ['top', 'bottom']
ax.scatter(
    xs, ys, s=30
)

for x, y, l, p in zip(xs, ys, ls, ps):
    ax.annotate(f'(e.g {l})'.replace('and','\nand').replace('2','\n2'), xy=(x,y), xytext=(x+xo[p],y+yo[p]), ha=has[p], va=vas[p])

ax.set_xlim([-1, 6])
ax.set_ylim([0.5, max(ys)+0.5])

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()


# remove stuff

sb.despine(fig)

# removing the axis ticks
ax.set_xticks([1, 4]) # labels
ax.set_yticks(range(1, max(ys)+1))
ax.set_xticklabels(['One', 'Four'])
ax.set_yticklabels(['No\nlearning','One\nfor\nwhole\ntime\nhorizon',
                    'At each\ntime\nperiod', #\n(discrete)',
                    #'Each\ntime\nperiod\n(continuous)'
                    ])
ax.set_xlabel('Number of objectives')
ax.set_ylabel('Frequency of learn-then-act points')

ax.annotate("", xy=(xmax, ymin), xytext=(xmin, ymin),
            arrowprops=dict(arrowstyle="->"))
ax.annotate("", xy=(xmin, ymax), xytext=(xmin, ymin),
            arrowprops=dict(arrowstyle="->"))
fig.tight_layout()

savefig4paper(fig, 'main_litmap')

clp()