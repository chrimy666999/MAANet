import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import to_rgba 
from matplotlib.lines import Line2D 

try:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'], 
        'mathtext.fontset': 'dejavuserif', 
    })
    print("已应用'B计划'字体逻辑：优先尝试 Times New Roman，失败则自动降级。")
except Exception as e:
    print(f"警告：设置字体时出错: {e}。将使用 Matplotlib 默认字体。")

FLOP_RANGES = [
    {'label': '< 35G',     'max_flops': 35, 'color': '#e41a1c'},  # Red
    {'label': '35G - 40G', 'max_flops': 40, 'color': '#ff7f00'},  # Orange
    {'label': '40G - 70G', 'max_flops': 70, 'color': '#4daf4a'},  # Green
    {'label': '> 70G',     'max_flops': float('inf'), 'color': '#377eb8'} # Blue
]

def get_color_from_flops(flops):
    """根据 FLOPs 返回对应的颜色"""
    for range_info in FLOP_RANGES:
        if flops < range_info['max_flops']:
            return range_info['color']
    return FLOP_RANGES[-1]['color'] 

legend_elements = []
for range_info in FLOP_RANGES:
    legend_elements.append(
        Line2D([0], [0], 
               marker='o', 
               color='w', 
               linestyle='None', 
               label=range_info['label'], 
               markerfacecolor=range_info['color'], 
               markersize=15, 
               alpha=0.8) 
    )

plt.rcParams['font.size'] = 15

fig, ax = plt.subplots(figsize=(15, 10))

data = [
    {'method': 'MAANet-S',       'x': 401,  'y': 31.12, 'flops': 24.8}, # n=6
    {'method': 'MAANet',         'x': 531,  'y': 31.24, 'flops': 31.5}, # n=8
    # {'method': 'MAANet-L',       'x': 661,  'y': 31.27, 'flops': 38.2}, # n=10
    {'method': 'MAANet-L',       'x': 791,  'y': 31.34, 'flops': 44.9}, # n=12
    {'method': 'EDSR-baseline',  'x': 1518, 'y': 30.35, 'flops': 114},
    {'method': 'CARN',           'x': 1592, 'y': 30.42, 'flops': 90.9},
    {'method': 'IMDN',           'x': 715,  'y': 30.45, 'flops': 40.9},
    {'method': 'LAPAR-A',        'x': 659,  'y': 30.42, 'flops': 94},
    {'method': 'PFFN',           'x': 569,  'y': 30.50, 'flops': 45.1},
    {'method': 'SwinIR-light',   'x': 897,  'y': 30.92, 'flops': 49.6},
    {'method': 'SMSR',           'x': 1006, 'y': 30.54, 'flops': 41.6},
    {'method': 'ESRT',           'x': 751,  'y': 30.75, 'flops': 96},
    {'method': 'ELAN-light',     'x': 601,  'y': 30.92, 'flops': 43.2},
    {'method': 'FMEN',           'x': 769,  'y': 30.70, 'flops': 44.2},
    {'method': 'HPUN-L',         'x': 734,  'y': 30.83, 'flops': 39.7},
    {'method': 'GASSL-B',        'x': 694,  'y': 30.70, 'flops': 39.9},
    {'method': 'NGswin',         'x': 1019, 'y': 30.80, 'flops': 36.4},
    {'method': 'SRFormer-light', 'x': 873,  'y': 31.17, 'flops': 62.8},
    {'method': 'CFGN',           'x': 621,  'y': 30.63, 'flops': 59},
    {'method': 'EConvMixN',      'x': 983,  'y': 30.76, 'flops': 44.5},
    {'method': 'KRGN',           'x': 612,  'y': 30.80, 'flops': 31.6},
    {'method': 'SRConvNet-L',    'x': 902,  'y': 30.96, 'flops': 45}
]



def main(save_dir = "."):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for d in data:
        area = 3 * (d['flops'] ** 2)
        
        color = get_color_from_flops(d['flops'])
        
        ax.scatter(d['x'], d['y'], s=area, alpha=0.8, marker='.', c=[color], edgecolors='white', linewidths=2.0)
        
        method_name = d['method']
        
        if method_name in ['MAANet-S', 'MAANet', 'MAANet-L']:
            weight = 'bold' 
            method_name = f"{method_name}\n(Ours)" 
            horizontal_align = 'center' 
        else:
            weight = 'normal' 
            horizontal_align = 'center' 

        ax.annotate(method_name, (d['x'], d['y']), 
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=24, 
                    ha=horizontal_align, 
                    va='bottom', 
                    fontweight=weight)

    ax.minorticks_on() 
    
    ax.set_axisbelow(True) 

    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray') 
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray') 

    ax.plot([data[0]['x'], data[1]['x'], data[2]['x']], [data[0]['y'], data[1]['y'], data[2]['y']], 
            color='red', linestyle='--', linewidth=2.5)
    
    star_coords_x = [data[0]['x'], data[1]['x'], data[2]['x']]
    star_coords_y = [data[0]['y'], data[1]['y'], data[2]['y']]

    star_sizes = [150, 250, 500]  

    ax.scatter(star_coords_x, star_coords_y,
            marker='*',                
            s=star_sizes,              
            c='white',                 
            edgecolors='black',       
            linewidths=1.5,           
            zorder=10                
            )

    ax.set_xlim(300, 1700)
    ax.set_ylim(30.20, 31.40)

    ax.set_xticks([400, 600, 800, 1000, 1200, 1400, 1600])
    ax.set_xticklabels([400, 600, 800, 1000, 1200, 1400, 1600], size=30)
    ax.set_yticks([30.20, 30.40, 30.60, 30.80, 31.0, 31.20, 31.40])
    ax.set_yticklabels([30.20, 30.40, 30.60, 30.80, 31.0, 31.20, 31.40], size=30)

    ax.set_ylabel('PSNR (dB)', fontsize=35, labelpad=10) 
    ax.set_xlabel('Parameters (K)', fontsize=35, labelpad=10) 
    plt.suptitle('PSNR vs. Parameters vs. FLOPs', fontsize=35)

    plt.tight_layout(rect=[0, 0.03, 1, 1.05])
    
    ax.legend(handles=legend_elements,       
              title='FLOPs',             
              loc='upper right',             
              handletextpad=0,            
              borderpad=0.5,                
              fontsize=24,                  
              title_fontsize=26,            
              ncol=1,                        
              facecolor='white',             
              edgecolor='black',             
              framealpha=0.8                 
              )

    output_filename = os.path.join(save_dir, 'model_complexity_robust_font.pdf')
    plt.savefig(output_filename, bbox_inches='tight')
    plt.savefig(output_filename.replace('.pdf', '.png'), bbox_inches='tight')
    
    print(f"图像已保存到 {output_filename.replace('.pdf', '.png')}")

if __name__ == "__main__":
    
    main()