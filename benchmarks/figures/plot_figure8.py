"""Render Figure 8: FLOPs panel and FP32 eager latency panel.

Reads figure8_fp32.json. Circle diameters encode parameter count in the FLOPs panel; areas encode peak allocated
memory in the latency panel.
"""
from pathlib import Path
import argparse, itertools, json, math, os
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.patches import FancyArrowPatch
import numpy as np

GREEN='#AAFF33'; GRAY='#DEDEDE'; AREA_MEMORY=1.2; AREA_PARAMS=450.24/13**2
META=dict(Title=None,Author=None,Subject=None,Creator=None,Producer=None,CreationDate=None,ModDate=None)

def content(row, mode):
    texts=[TextArea(row['method'],textprops=dict(fontsize=18,fontweight='bold' if row['method']=='Graphene' else 'normal'))]
    if mode=='bf16_compiled' and row[mode].get('source')!='original_submission':
        change=row['bf16_compiled']['psnr']-row['fp32_quality']['psnr']
        texts.append(TextArea(f'{change:+.3f} dB'.replace('-', '−'),textprops=dict(fontsize=12)))
    elif row.get('year'):
        texts.append(TextArea(f"({row['year']})",textprops=dict(fontsize=12)))
    return VPacker(children=texts,align='center',pad=0,sep=.5)

def panel(ax, rows, mode):
    if mode=='flops':
        x=lambda r:r['flops_G']; y=lambda r:r['published_psnr']; area=lambda r:r['params_M']**2*AREA_PARAMS
        title=''; xlabel='FLOPs (G)'; legend_values=[10,15,20,23]
        legend_title='Params'; legend_label=lambda n:f'{n}M'; legend_area=lambda n:n**2*AREA_PARAMS
        limits=(92,282); preferred_left={'CAT-A','HAT','ATD','IPG','MambaIR','PFT','IET'}
    else:
        rows=[r for r in rows if r.get(mode,{}).get('status')=='complete']
        if mode=='bf16_compiled':
            rows=[r for r in rows if 'psnr' in r[mode] and (r[mode].get('source')=='original_submission' or r.get('fp32_quality',{}).get('status')=='complete')]
        x=lambda r:r[mode]['latency_ms']
        y=lambda r:r['published_psnr'] if mode=='fp32_eager' else r[mode]['psnr']
        area=lambda r:r[mode]['peak_memory_mib']*AREA_MEMORY
        title=''
        xlabel='Latency (ms), FP32 eager' if mode=='fp32_eager' else 'Latency (ms), BF16 compiled'
        legend_values=[200,800,1600] if mode=='fp32_eager' else [100,200,400]
        legend_title='Memory'
        legend_label=lambda n:f'{n:,}MiB';legend_area=lambda n:n*AREA_MEMORY
        xmax=max(x(r) for r in rows)
        limits=(0,math.ceil((xmax*1.12)/25)*25)
        preferred_left={'CAT-A','ART','HAT','IPG','PFT','IET'}
    ymin=min(27.35,min(y(r) for r in rows)-.12)
    if mode=='bf16_compiled':
        ymin-=.35
    ymax=max(28.80,max(y(r) for r in rows)+.13)
    ax.set_xlim(*limits);ax.set_ylim(ymin,ymax)
    ax.set_xlabel(xlabel);ax.set_ylabel(r'Urban100 $\times4$ PSNR (dB)')
    ax.set_title(title,fontsize=18,pad=12)
    ax.grid(True,ls='--',lw=.55,color='black',alpha=.2);ax.set_axisbelow(True)
    for r in sorted(rows,key=lambda r:-area(r)):
        ax.scatter(x(r),y(r),s=area(r),facecolor=GREEN if r['method']=='Graphene' else GRAY,
                   edgecolor='black',linewidth=1,zorder=4 if r['method']=='Graphene' else 3)
    handles=[Line2D([0],[0],marker='o',linestyle='',markerfacecolor=GRAY,markeredgecolor='black',
                    markersize=math.sqrt(legend_area(v)),label=legend_label(v)) for v in legend_values]
    legend_location='upper left' if mode=='flops' else ('upper right' if mode=='bf16_compiled' else 'lower right')
    legend=ax.legend(handles=handles,title=legend_title,loc=legend_location,
                     borderpad=.55,labelspacing=.1,handletextpad=.4,handlelength=1,fontsize=18,title_fontsize=18)
    marker_width=max(h.get_markersize() for h in handles)+1
    for box,handle in zip(legend._legend_handle_box.get_children()[0].get_children(),handles):
        drawing=box.get_children()[0];diameter=handle.get_markersize()
        drawing.width=marker_width;drawing.height=diameter+1
        drawing.xdescent=drawing.ydescent=0;box.align='center'
        marker=drawing.get_children()[0]
        marker.set_xdata([marker_width/2]);marker.set_ydata([(diameter+1)/2]);marker.set_markevery(None)
    legend.get_frame().set_edgecolor('black');legend.get_frame().set_facecolor('white');legend.get_frame().set_alpha(1)
    annotations=[]
    for r in rows:
        radius=math.sqrt(area(r))/2
        left=r['method'] in preferred_left
        arrowprops=None
        a=AnnotationBbox(content(r,mode),(x(r),y(r)),xybox=((-1 if left else 1)*(radius+6),0),
                         xycoords='data',boxcoords='offset points',box_alignment=(1 if left else 0,.5),
                         frameon=False,pad=0,annotation_clip=False,zorder=6,arrowprops=arrowprops)
        ax.add_artist(a);annotations.append((r,a,radius,left))
    ax.figure.canvas.draw();rend=ax.figure.canvas.get_renderer();bounds=ax.get_window_extent(rend)
    legendbox=legend.get_window_extent(rend)
    centers=[(ax.transData.transform((x(r),y(r))),rend.points_to_pixels(math.sqrt(area(r))/2+.7)) for r in rows]
    # Place labels near their own circles while minimizing overlaps with circles,
    # other labels, and the legend. The plotted values never move.
    placed=[legendbox]
    placements={}
    for r,a,radius,left in sorted(annotations,key=lambda t:(t[0]['method']!='Graphene',-y(t[0]))):
        candidates=[]
        for side in ([True,False] if left else [False,True]):
            for dy in [0,-12,12,-24,24,-36,36,-48,48]:
                dx=(-1 if side else 1)*(radius+6)
                candidates.append((dx,dy,1 if side else 0))
        candidates += [(0,-radius-22,.5),(0,radius+22,.5)]
        best=None
        for dx,dy,align in candidates:
            a.xybox=(dx,dy);a._box_alignment=(align,.5)
            a.update_positions(rend);box=a.offsetbox.get_window_extent(rend)
            penalty=abs(dy)*.5+(0 if (dx<0)==left else 5)
            if not(bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1)):penalty+=1e7
            for b in placed:
                inter=box.intersection(box.padded(2),b.padded(2))
                if inter is not None:penalty+=inter.width*inter.height*100
            for center,rad in centers:
                nearest=np.clip(center,[box.x0,box.y0],[box.x1,box.y1])
                dist=np.linalg.norm(center-nearest)
                if dist<rad:penalty+=(rad-dist)*1000
            if best is None or penalty<best[0]:best=(penalty,dx,dy,align)
        _,dx,dy,align=best;a.xybox=(dx,dy);a._box_alignment=(align,.5);a.update_positions(rend)
        placed.append(a.offsetbox.get_window_extent(rend))
        placements[r['method']]=dict(dx=dx,dy=dy,alignment=align,penalty=best[0])
    if mode=='fp32_eager':
        for r,a,radius,left in annotations:
            name=r['method']
            on_left=name in {'CAT-A','ART','HAT','IPG','PFT','IET'}
            dx=(-1 if on_left else 1)*(radius+6);dy=0;align=1 if on_left else 0
            if name=='CAT-A':dy=-8
            if name=='SRFormer':dx=0;dy=-27;align=.5
            if name=='MambaIR':dx+=3
            if name=='SwinIR':dy=-6
            a.xybox=(dx,dy);a._box_alignment=(align,.5);a.update_positions(rend)
            placements[name]=dict(dx=dx,dy=dy,alignment=align)
    ax.annotate('Better',xy=(.47,.91),xytext=(.61,.78),xycoords='axes fraction',textcoords='axes fraction',
                fontsize=18,fontstyle='italic',color='#8C8C8C',ha='center',va='top',
                arrowprops=dict(arrowstyle='simple,head_length=1,head_width=1.3,tail_width=.65',
                                facecolor='#C5C5C5',edgecolor='none',mutation_scale=18),zorder=2)
    if mode=='bf16_compiled':
        ax.text(.025,.035,
                'BF16 compiled unavailable: IET / PFT\n'
                '(unsupported CUDA operator); CAT-A (compilation error).',
                transform=ax.transAxes,fontsize=14,linespacing=1.15,ha='left',va='bottom',
                bbox={'facecolor':'white','edgecolor':'none','pad':2},zorder=8)
    return placements

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=json.loads(a.data.read_text())['classical_x4']
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,
                        'font.size':18,'axes.labelsize':18,'xtick.labelsize':18,'ytick.labelsize':18})
    a.output.mkdir(parents=True,exist_ok=True)
    sets=[('figure8_fp32',['flops','fp32_eager'])]
    qa={}
    for name,modes in sets:
        n=len(modes)
        width=8.4*n
        left,right,spacing=.075/n,.99,.22
        if n==3:
            panel_width=width*(right-left)/(n+(n-1)*spacing)
            left_margin,right_margin=width*left,width*(1-right)
            spacing=.14
            width=left_margin+right_margin+panel_width*(n+(n-1)*spacing)
            left,right=left_margin/width,1-right_margin/width
        fig,axes=plt.subplots(1,n,figsize=(width,6.0),squeeze=False,dpi=120)
        fig.subplots_adjust(left=left,right=right,bottom=.14,top=.90,wspace=spacing)
        qa[name]={mode:panel(ax,rows,mode) for ax,mode in zip(axes[0],modes)}
        if n==3:
            fig.canvas.draw()
            renderer=fig.canvas.get_renderer()
            boxes=[ax.get_window_extent(renderer) for ax in axes[0]]
            label_overhang=max(box.x0-ax.yaxis.label.get_window_extent(renderer).x0
                               for ax,box in zip(axes[0],boxes))
            common_x=min((ax.yaxis.label.get_transform().transform(ax.yaxis.label.get_position())[0]-box.x0)/box.width
                         for ax,box in zip(axes[0],boxes))
            for ax in axes[0]:
                ax.yaxis.set_label_coords(common_x,.5)
            gap_inches=label_overhang/fig.dpi+10/72
            width=left_margin+right_margin+n*panel_width+(n-1)*gap_inches
            fig.set_size_inches(width,6.0)
            fig.subplots_adjust(left=left_margin/width,right=1-right_margin/width,
                                bottom=.14,top=.90,wspace=gap_inches/panel_width)
        fig.savefig(a.output/f'{name}.pdf',metadata=META,bbox_inches='tight',pad_inches=.06)
        fig.savefig(a.output/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.06)
        plt.close(fig)
    (a.output/'layout.json').write_text(json.dumps(qa,indent=2)+'\n')

if __name__=='__main__':main()
