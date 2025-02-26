from nestmodel.unified_functions import to_fast_graph
import numpy as np


from collections import defaultdict, Counter
def relative_colors(partitions, start, stop, shorten=True):
    """Computes relative colors instead of the global colors
    for the following input colors per round
    0,1
    0,2
    1,0
    1,0
    we obtain instead
    0,0  <- first color with prefix 0
    0,1  <- second color with prefix 0
    1,0
    1,0

    if we have shorten = True we also obtain
    0,0
    0,1
    1,
    1,
    i.e. the class 1 at depth 0 is not splitted up further!
    """
    i_tpl = defaultdict(tuple)
    parent_counter = defaultdict(list)
    num_nodes = Counter()
    num_nodes[tuple()]= len(partitions[0])
    for depth in range(start, stop):
        for i in range(len(partitions[depth])):
            if depth == start:
                i_tpl[i] += (partitions[depth][i],)
                num_nodes[i_tpl[i]]+=1
            else:

                new_tpl =i_tpl[i]+(partitions[depth][i],)

                try:
                    val = parent_counter[i_tpl[i]].index(new_tpl)
                except ValueError:
                    val = len(parent_counter[i_tpl[i]])
                    parent_counter[i_tpl[i]].append(new_tpl)
                i_tpl[i] += (val,)
                num_nodes[i_tpl[i]]+=1
    if shorten:
        while True:
            changes = False
            new_i_tpl = i_tpl.copy()
            for i, tpl in i_tpl.items():
                if num_nodes[tpl]==num_nodes[tpl[:-1]]:
                    changes=True
                    new_i_tpl[i]= tpl[:-1]
            if changes:
                i_tpl =new_i_tpl
            else:
                break
    return i_tpl


def to_base_10_tpl(h):
    return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))

def to_base_10_arr(h):
    return np.array(tuple(int(h[i:i+2], 16) for i in (1, 3, 5)))

def to_base_16(tpl):
    out = "#"
    for i in range(3):
        tmp = hex(tpl[i])[2:]
        if len(tmp)==1:
            out+="0"+tmp
        else:
            out+=tmp
    assert len(out)==7, (out, tpl)
    return out

def modify_color(start_hex, vals=(40,40)):
    if isinstance(start_hex, str):
        in_colors = {}
        in_colors[(0,)]=start_hex
    elif isinstance(start_hex, dict):
        in_colors=start_hex

    colors=in_colors.copy()
    for name, h in in_colors.items():
        base_color = to_base_10_arr(h)
        for i, m in enumerate( [
            [0,0,0],
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [1,1,0],
            [0,1,1],
            [1,0,1],
            [1,1,1]
        ]):
            mod =np.array(m, dtype=float)
            #mod/=mod.sum()
            # blue 15, 40
            # red  15,  30

            #vals = (15,40)
            if len(name)!= 1:
                mod*=vals[0]
            else:
                mod*=vals[1]
            mod=mod.astype(int)
            tmp = base_color.copy()
            tmp -= np.array(mod, dtype=int)
            tmp = np.maximum(tmp, 0)
            hex_col = to_base_16(tmp)
            colors[name+(i,)]=hex_col
    return colors


def get_family_node_colors(G_nx, depth, strength):
    """ Obtain colors that represent the wl colors of the graph G_nx


    """

    G = to_fast_graph(G_nx)
    G.ensure_edges_prepared()
    return get_familiy_node_colors_for_partition(G.base_partitions, depth, strength)



def get_familiy_node_colors_for_partition(partitions, depth, strength):
    """Obtain colors that represent the families present in partitions
    returns a list of hex color codes that should resemble the family tree of the partitions
    """
    max_depth = len(partitions)
    depth = min(max_depth-1, depth)
    color_arr = partitions[depth]
    colors = partitions[depth]
    num_nodes = len(partitions[0])
    if depth==0 and np.all(color_arr == color_arr[0]):
        colors=["gray" for _ in range(num_nodes)]
    else:
        colors_map = {
                (0,): '#a2cffe',  #  turquoise   '#AFEEEE',
                 (2,): '#aaff32', #  green   #'#90EE90',
                 (1,): '#fe7b7c', #  red
                 (3,): '#ffff14', #  yellow  # '#fffe71',
                 (4,): '#DDA0DD', #  purple
                 (5,): '#8cffdb', #  green/blue
                 (6,): '#ffa756', #  orange
                 (7,): '#ffb2d0', #  pink
            }
        for _ in range(depth):
            colors_map = modify_color(colors_map, vals=(strength,strength))
        depth_zero_is_uniform = np.all( partitions[0] == color_arr[0])
        node_tpls = relative_colors(partitions, int(depth_zero_is_uniform), depth+depth_zero_is_uniform)
        #print(node_tpls)
        def get_color(i):
            tpl = node_tpls[i]
            return colors_map[tpl]
        colors = [get_color(i) for i in range(num_nodes)]
    return colors



def draw_network_with_colors(G_nx, depth=0, pos=None, strength = 40, **kwargs):
    import networkx as nx
    colors = get_family_node_colors(G_nx, depth, strength=strength)
    #print(colors)
    nx.draw_networkx(G_nx, pos=pos, node_color = colors, **kwargs)