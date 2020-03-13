import os
import graphviz as gv
import pygraphviz as pgv
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import _pickle as pickle

np.random.seed(seed=1607)
import datetime
import matplotlib as mpl

mpl.rc('font', family='Arial')

# import functions for gene network analysis
import Generate_gene_int_dict_UDN_excel_file_new_Feb2020 as gen_dict_file
from Load_gene_HGNC_symbols_alias_info_local import *

now = datetime.datetime.now()
month = str(now.strftime("%b"))
day = str(now.strftime("%d"))
year = str(now.strftime("%y"))

# define python dictionary for 3 letter - 1 letter AA code for exhaustive search

three_aa_to_one_aa = {'ALA': 'A',
                      'ARG': 'R',
                      'ASN': 'N',
                      'ASP': 'D',
                      'CYS': 'C',
                      'GLN': 'Q',
                      'GLU': 'E',
                      'GLY': 'G',
                      'HIS': 'H',
                      'ILE': 'I',
                      'LEU': 'L',
                      'LYS': 'K',
                      'MET': 'M',
                      'PHE': 'F',
                      'PRO': 'P',
                      'SER': 'S',
                      'THR': 'T',
                      'TRP': 'W',
                      'TYR': 'Y',
                      'VAL': 'V',
                      }

# define python dictionaries for color and values for individual systems biology features

cmaps = dict()
cmaps['ppi dist'] = 'YlOrRd_r'
cmaps['pwy dist'] = 'YlOrRd_r'
cmaps['txt dist'] = 'YlOrRd_r'
cmaps['pwy similarity'] = 'YlOrRd'
cmaps['pheno similarity'] = 'YlOrRd'
cmaps['coex rank'] = 'YlOrRd_r'

vmins = dict()
vmins['ppi dist'] = 1
vmins['pwy dist'] = 1
vmins['txt dist'] = 1
vmins['pwy similarity'] = 1
vmins['pheno similarity'] = 1
vmins['coex rank'] = 1

vmaxs = dict()
vmaxs['ppi dist'] = 3
vmaxs['pwy dist'] = 3
vmaxs['txt dist'] = 3
vmaxs['pwy similarity'] = 100
vmaxs['pheno similarity'] = 100
vmaxs['coex rank'] = 1000

ticks = dict()
ticks['ppi dist'] = [1, 2, 3]
ticks['pwy dist'] = [1, 2, 3]
ticks['txt dist'] = [1, 2, 3]
ticks['pwy similarity'] = [30, 60, 90]
ticks['pheno similarity'] = [30, 60, 90]
ticks['coex rank'] = [300, 600, 900]

labels = dict()
labels['ppi dist'] = 'distance on PPI network'
labels['pwy dist'] = 'distance on pathways interaction network'
labels['txt dist'] = 'distance on literature-mined interaction network'
labels['pwy similarity'] = ' % pathways similarity'
labels['pheno similarity'] = '% phenotype similarity'
labels['coex rank'] = 'mutual co-expression rank'

# define python dictionaries for network edge colors and labels

edge_colors = {'ppi': '#08306b',
               'pwy': '#a50f15',
               'text': '#74c476',
               'pathway': '#f768a1',
               'phenotype': '#9e9ac8',
               'coexpression': '#fec44f',
               'digenic': 'yellow1'}

edge_label_dict = {'ppi': 'protein-protein interaction',
                   'pwy': 'interaction between biochemical pathways',
                   'text': 'literature-mined interaction',
                   'pathway': 'pathway overlap',
                   'phenotype': 'phenotype overlap',
                   'coexpression': 'co-expression',
                   'digenic': 'predicted to cause digenic disease'}

mut_color = 'black'
wt_color = 'white'
sib_color = '#d9d9d9'
not_seq_color = '#c6dbef'

# import digenic prediction pandas dataframes

all_digenic_gene_pairs_pred_value_dict = pd.read_pickle(
    '/Users/souhrid/DATA_generated/Digenic_classifier_predictions/pred_digenic_human_gene_meeting_threshold_dfs/all_digenic_gene_pairs_pred_value_dict.pkl')

dig_gp_preds_unaffected_clf_df = pd.read_pickle(
    '/Users/souhrid/DATA_generated/Digenic_classifier_predictions/pred_digenic_human_gene_meeting_threshold_dfs/unaffected_preds_df_Feb12_20.pkl')


# function to plot network image
def draw_network_image(id,  # patient ID
                       gene_int_dict,  # python dict. with interaction details for gene pairs in patient
                       family_seq_result,   # which other family members were sequenced
                       cutoff,  # Distance cutoff on network to consider
                       digenic_pairs,  # predicted digenic interactions in patient
                       all_genes_dict,  # python dict. for all genes with rare variants in patient and inheritance
                       long_excel_file_genes,  # list of genes with rare variants in patient
                       short_excel_file_genes,  # list of genes with rare variants in patient, relevant to patient phenotype
                       sel_feats,  # Systems biology features to plot
                       ):

    udn_id = id

    d_edges = {'ppi': [], 'pwy': [], 'txt': []}

    o_edges = {'ppi': [], 'pwy': [], 'txt': []}

    a_nodes = []
    c_nodes = []
    coex_edges = []
    pathway_edges = []
    phenotype_edges = []
    pred_digenic_edges = []

    # looping through all genes with any interactions
    for g1 in gene_int_dict:
        # lloping through all genes interacting with gene1
        for g2 in gene_int_dict[g1]:

            gene_pair = tuple(sorted((g1, g2)))

        # PPI
            if gene_int_dict[g1][g2]['ppi network distance'] == 1.0:

                c_nodes = list(set(c_nodes).union(list(gene_pair)))

                d_edges['ppi'].append(gene_pair)

            elif 1 < gene_int_dict[g1][g2]['ppi network distance'] <= cutoff:

                c_nodes = list(set(c_nodes).union(list(gene_pair)))

                network_path = gene_int_dict[g1][g2]['ppi shortest path']
                a_nodes.extend(network_path[1:-1])
                for i in range(len(network_path) - 1):
                    gene_a = network_path[i]
                    gene_b = network_path[i + 1]

                    if i == 0:
                        gene_a = g1
                    if i == len(network_path) - 2:
                        gene_b = g2

                    o_edges['ppi'].append(tuple(sorted((gene_a, gene_b))))

        # Pathway distance

            if gene_int_dict[g1][g2]['pwy network distance'] == 1.0:

                c_nodes = list(set(c_nodes).union(list(gene_pair)))

                d_edges['pwy'].append(gene_pair)

            elif 1 < gene_int_dict[g1][g2]['pwy network distance'] <= cutoff:

                c_nodes = list(set(c_nodes).union(list(gene_pair)))

                network_path = gene_int_dict[g1][g2]['pwy shortest path']
                a_nodes.extend(network_path[1:-1])
                for i in range(len(network_path) - 1):
                    gene_a = network_path[i]
                    gene_b = network_path[i + 1]

                    if i == 0:
                        gene_a = g1
                    if i == len(network_path) - 2:
                        gene_b = g2

                    o_edges['pwy'].append(tuple(sorted((gene_a, gene_b))))

        # Literature-mined interactiom distance

            if gene_int_dict[g1][g2]['txt network distance'] == 1.0:

                c_nodes = list(set(c_nodes).union(list(gene_pair)))

                d_edges['txt'].append(gene_pair)

            elif 1 < gene_int_dict[g1][g2]['txt network distance'] <= cutoff:

                c_nodes = list(set(c_nodes).union(list(gene_pair)))

                network_path = gene_int_dict[g1][g2]['txt shortest path']
                a_nodes.extend(network_path[1:-1])
                for i in range(len(network_path) - 1):
                    gene_a = network_path[i]
                    gene_b = network_path[i + 1]

                    if i == 0:
                        gene_a = g1
                    if i == len(network_path) - 2:
                        gene_b = g2

                    o_edges['txt'].append(tuple(sorted((gene_a, gene_b))))

            # co-expression

            if 0 < float(gene_int_dict[g1][g2]['coex rank']) <= 1000:
                c_nodes = list(set(c_nodes).union(list(gene_pair)))

                coex_value = gene_int_dict[g1][g2]['coex rank']
                # edge width on network adjusted such that diff. of width ebetwene rank 1 and 1000 is maximized.
                width_value = np.mean([20.078 * coex_value ** -0.344, np.abs(-1.918 * np.log(coex_value) + 14.216)])
                coex_edges.append((gene_pair, coex_value, width_value))

            # pathways similarity

            if float(gene_int_dict[g1][g2]['pwy similarity']) > 0.1:
                c_nodes = list(set(c_nodes).union(list(gene_pair)))
                path_sim_value = gene_int_dict[g1][g2]['pwy similarity']
                width_value = path_sim_value * 10
                pathway_edges.append((gene_pair, path_sim_value, width_value))

            # phenotypes similarity

            if float(gene_int_dict[g1][g2]['pheno similarity']) > 0.1:
                c_nodes = list(set(c_nodes).union(list(gene_pair)))
                pheno_sim_value = gene_int_dict[g1][g2]['pheno similarity']
                width_value = pheno_sim_value * 10
                phenotype_edges.append((gene_pair, pheno_sim_value, width_value))

    # predicted digenic pairs

    for gene_pair, pred in digenic_pairs:
        list(set(c_nodes).union(list(gene_pair)))
        pred_dig_value = pred
        width_value = pred_dig_value * 10
        pred_digenic_edges.append((gene_pair, pred_dig_value, width_value))

    display_colors = {}
    con_nodes = []

    H = pgv.AGraph(strict=False,
                   ranksep='1.4',
                   fixedsize=False,
                   splines='line',
                   size=50, )

    display_edge_colors = {}

    individual_edge_colors = {}

    # add indirect edges

    for int_type in o_edges:
        if int_type in sel_feats or 'all' in sel_feats:
            for ee in list(set(o_edges[int_type])):
                con_nodes = list(set(con_nodes).union(list(ee)))
                H.add_edge(ee, style='solid',
                           color=edge_colors[int_type],
                           len=10 * cutoff,
                           penwidth=2,
                           alpha=0.5,
                           weight=10)

                display_edge_colors[edge_label_dict[int_type]] = edge_colors[int_type]

    # add direct PPI, pathways, literature-mined interaction edges

    for int_type in d_edges:
        if int_type in sel_feats or 'all' in sel_feats:
            for ee in list(set(d_edges[int_type])):
                con_nodes = list(set(con_nodes).union(list(ee)))
                H.add_edge(ee, style='solid',
                           color=edge_colors[int_type],
                           len=10 * cutoff,
                           penwidth=4,
                           alpha=1,
                           weight=1)

                display_edge_colors[edge_label_dict[int_type]] = edge_colors[int_type]

    # add pred. digenic edges

    if 'digenic' in sel_feats or 'all' in sel_feats:
        for ee in set(pred_digenic_edges):
            con_nodes = list(set(con_nodes).union(list(ee[0])))
            H.add_edge(ee[0], style='solid',
                       color=edge_colors['digenic'],
                       len=10 * cutoff,
                       penwidth=2 * ee[2],
                       alpha=0.5,
                       fontsize=20,
                       xlabel='dig={v}'.format(v=round(ee[1], 2)),
                       weight=1)

            display_edge_colors[edge_label_dict['digenic']] = edge_colors['digenic']

    # add co-expression edges

    if 'coex' in sel_feats or 'all' in sel_feats:
        for ee in set(coex_edges):
            con_nodes = list(set(con_nodes).union(list(ee[0])))
            H.add_edge(ee[0], style='solid',
                       color=edge_colors['coexpression'],
                       len=10 * cutoff,
                       penwidth=2 * ee[2],
                       alpha=0.5,
                       fontsize=18,
                       xlabel='rank={v}'.format(v=int(round(ee[1]))),
                       weight=1)

            display_edge_colors[edge_label_dict['coexpression']] = edge_colors['coexpression']

    # add pathway similarity edges

    if 'pwy' in sel_feats or 'all' in sel_feats:
        for ee in set(pathway_edges):
            con_nodes = list(set(con_nodes).union(list(ee[0])))
            H.add_edge(ee[0], style='solid',
                       color=edge_colors['pathway'],
                       len=10 * cutoff,
                       penwidth=2 * ee[2],
                       xlabel='path={v}%'.format(v=round(ee[1] * 100., 1)),
                       alpha=0.5,
                       fontsize=18,
                       weight=1)

            display_edge_colors[edge_label_dict['pathway']] = edge_colors['pathway']

    # add phenotype similarity edges

    if 'pheno' in sel_feats or 'all' in sel_feats:
        for ee in set(phenotype_edges):
            con_nodes = list(set(con_nodes).union(list(ee[0])))
            H.add_edge(ee[0], style='solid',
                       color=edge_colors['phenotype'],
                       len=10 * cutoff,
                       penwidth=2 * ee[2],
                       alpha=0.5,
                       fontsize=20,
                       xlabel='pheno={v}'.format(v=round(ee[1], 1)),
                       weight=1)

            display_edge_colors[edge_label_dict['phenotype']] = edge_colors['phenotype']

    list_leg_nodes = []

    # add legends

    H.add_node('legA', label='{m_m}\n\n{f_m}'.format(m_m='variant(s)\nfrom mother',
                                                     f_m='variant(s) from\n father'),
               xlabel='gene name\nde novo variant(s)',
               xlp="-1.0,1.0",
               forcelabels=True,
               shape='circle',
               style='wedged',
               fillcolor=mut_color + ':' + mut_color,
               color='grey37',
               fontcolor='#fb6a4a',
               fontname='arial bold',
               fontsize=28,
               size=30,
               width=5,
               penwidth=3,
               fixedsize=True,
               alpha=1, rank='same')
    list_leg_nodes.append('legA')

    if family_seq_result['sibling']:
        H.add_node('legB', label='variant(s) also\npresent in other\nfamily members',
                   shape='circle',
                   style='filled',
                   fillcolor=sib_color,
                   color='white',
                   fontcolor='black',
                   fontname='arial',
                   fontsize=28,
                   size=15,
                   width=5,
                   penwidth=3,
                   fixedsize=False,
                   forcelabels=True)

        list_leg_nodes.append('legB')

    if not family_seq_result['mom'] or not family_seq_result['dad']:
        H.add_node('legD', label='parent\nnot\nsequenced',
                   shape='circle',
                   style='filled',
                   fillcolor=not_seq_color,
                   color='white',
                   fontcolor='black',
                   fontname='arial',
                   fontsize=28,
                   size=15,
                   width=5,
                   penwidth=3,
                   fixedsize=False,
                   forcelabels=True)

        list_leg_nodes.append('legD')

    for i in range(len(list_leg_nodes) - 1):
        H.add_edge(list_leg_nodes[i], list_leg_nodes[i + 1], style='invis', len=6, rank='same')

    x_nodes = [('a', 'b'), ('c', 'd'), ('e', 'f'), ('g', 'h'), ('i', 'j'), ('k', 'l'), ('m', 'n')]

    for n, edge_label in enumerate(display_edge_colors):
        H.add_node(x_nodes[n][0], fillcolor='white', fontcolor='white', style='invis', shape='rectangle', rank='same')
        H.add_node(x_nodes[n][1], fillcolor='white', fontcolor='white', style='invis', shape='rectangle', rank='same')

        H.add_edge(x_nodes[n], color=display_edge_colors[edge_label],
                   xlabel=edge_label, penwidth=8, len=4, rotation=0,
                   fontcolor='black', rank='same',
                   fontname='arial',
                   fontsize=26
                   )

    for i in range(len(display_edge_colors) - 1):
        H.add_edge(x_nodes[i][0], x_nodes[i + 1][0], style='invis')
        H.add_edge(x_nodes[i][1], x_nodes[i + 1][1], style='invis')

    H.add_edge(list_leg_nodes[-1], 'b', style='invis', len=4)

    print('UDN ID =', udn_id)
    print('all nodes and edges added... saving network...')

    # add gene nodes

    H.add_nodes_from(list(set(a_nodes).difference(con_nodes)),
                     style='filled',
                     fillcolor='white',
                     width=0.5,
                     height=0.2,
                     shape='ellipse',
                     fontcolor='black',
                     fontname='arial',
                     fontsize=18,
                     size=20,
                     penwidth=0.5,
                     fixedsize=False,
                     forcelabels=True,
                     alpha=0.4)

    for g in con_nodes:
        muts_dict = get_classified_muts_for_gene(gene=g, genes_dict=all_genes_dict)

        col = get_color_for_gene_from_muts(gene=g, muts_dict=muts_dict, family_seq_result=family_seq_result)

        if g in short_excel_file_genes:

            H.add_node(g, label='{m_m}\n\n\n\n{f_m}'.format(m_m='\n'.join(muts_dict['mom']),
                                                            f_m='\n'.join(muts_dict['dad'])),
                       xlabel='{n}\n{d_m}'.format(n=g, d_m='\n'.join(muts_dict['de novo'])),
                       xlp="-1.0,1.0('!')",
                       forcelabels=True,
                       shape='circle',
                       style='wedged',
                       fillcolor=col,
                       color='grey37',
                       fontcolor='#fb6a4a',
                       fontname='arial bold',
                       fontsize=36,
                       size=50,
                       width=3,
                       penwidth=4,
                       fixedsize=True,
                       alpha=0.7
                       )
        else:
            H.add_node(g, label='{m_m}\n\n\n\n{f_m}'.format(m_m='\n'.join(muts_dict['mom']),
                                                            f_m='\n'.join(muts_dict['dad'])),
                       xlabel='{n}\n{d_m}'.format(n=g, d_m='\n'.join(muts_dict['de novo'])),
                       xlp="-1.0,1.0('!')",
                       forcelabels=True,
                       shape='doublecircle',
                       style='wedged',
                       fillcolor=col,
                       color='grey37',
                       fontcolor='#fb6a4a',
                       fontname='arial bold',
                       fontsize=30,
                       size=50,
                       width=2.5,
                       penwidth=3,
                       fixedsize=True,
                       alpha=0.7
                       )

    H.layout(prog='dot')
    H.layout(prog='neato')
    return H


def draw_heatmaps(udn_id,  # patient ID
                  gene_int_dict,  # python dict. with interaction details for gene pairs in patient
                  all_genes_dict,  # python dict. with all genes with rare variants in patient and inheritance
                  ):

    # Initialize pandas dataframes for all features

    dfs = dict()
    dfs['ppi dist'] = pd.DataFrame(index=sorted(all_genes_dict.keys()), columns=sorted(all_genes_dict.keys()),
                                   data=np.zeros(shape=(len(all_genes_dict), len(all_genes_dict))))

    dfs['pwy dist'] = pd.DataFrame(index=sorted(all_genes_dict.keys()), columns=sorted(all_genes_dict.keys()),
                                   data=np.zeros(shape=(len(all_genes_dict), len(all_genes_dict))))

    dfs['txt dist'] = pd.DataFrame(index=sorted(all_genes_dict.keys()), columns=sorted(all_genes_dict.keys()),
                                   data=np.zeros(shape=(len(all_genes_dict), len(all_genes_dict))))

    dfs['pwy similarity'] = pd.DataFrame(index=sorted(all_genes_dict.keys()), columns=sorted(all_genes_dict.keys()),
                                         data=np.zeros(shape=(len(all_genes_dict), len(all_genes_dict))))

    dfs['pheno similarity'] = pd.DataFrame(index=sorted(all_genes_dict.keys()), columns=sorted(all_genes_dict.keys()),
                                           data=np.zeros(shape=(len(all_genes_dict), len(all_genes_dict))))

    dfs['coex rank'] = pd.DataFrame(index=sorted(all_genes_dict.keys()), columns=sorted(all_genes_dict.keys()),
                                    data=np.zeros(shape=(len(all_genes_dict), len(all_genes_dict))))

    # loop through all genes with interactions

    for g1 in gene_int_dict:
        # loop through all genes interacting with a given gene
        for g2 in gene_int_dict[g1]:
            # add values to df.
            dfs['ppi dist'][g1][g2] = [gene_int_dict[g1][g2]['ppi network distance']
                                       if gene_int_dict[g1][g2]['ppi network distance'] < 4 else 0][0]
            dfs['ppi dist'][g2][g1] = dfs['ppi dist'][g1][g2]

            dfs['pwy dist'][g1][g2] = [gene_int_dict[g1][g2]['pwy network distance']
                                       if gene_int_dict[g1][g2]['pwy network distance'] < 4 else 0][0]
            dfs['pwy dist'][g2][g1] = dfs['pwy dist'][g1][g2]

            dfs['txt dist'][g1][g2] = [gene_int_dict[g1][g2]['txt network distance']
                                       if gene_int_dict[g1][g2]['txt network distance'] < 4 else 0][0]
            dfs['txt dist'][g2][g1] = dfs['txt dist'][g1][g2]

            dfs['pwy similarity'][g1][g2] = gene_int_dict[g1][g2]['pwy similarity'] * 100
            dfs['pwy similarity'][g2][g1] = dfs['pwy similarity'][g1][g2]

            dfs['pheno similarity'][g1][g2] = gene_int_dict[g1][g2]['pheno similarity'] * 100
            dfs['pheno similarity'][g2][g1] = dfs['pheno similarity'][g1][g2]

            dfs['coex rank'][g1][g2] = gene_int_dict[g1][g2]['coex rank']
            dfs['coex rank'][g2][g1] = dfs['coex rank'][g1][g2]

    # initialize figure
    fig, ax = plt.subplots(2, 3, figsize=(30, 20))
    for i, feat in enumerate(dfs):

        # returns df with only rows and columns for which there is coverage for given feature
        df_clean = get_clean_values_df(dfs, feat)

        col = i % 3
        row = i // 3

        # setting up mask to show only half of heatmap; avoids redundancy.

        mask = np.zeros_like(df_clean)
        mask[np.triu_indices_from(mask)] = True

        # plot heatmap using seaborn; using different axes to plot each feature

        hmap = seaborn.heatmap(df_clean, ax=ax[row][col], vmin=vmins[feat], vmax=vmaxs[feat],
                               square=True, cmap=cmaps[feat], mask=mask, linewidths=0,
                               cbar_kws={"shrink": .4, 'ticks': ticks[feat], 'spacing': 'proportional'})

        ax[row][col].set_xticks([i + 0.5 for i in range(len(df_clean.index) - 1)])
        ax[row][col].set_xticklabels(df_clean.index[:-1], fontsize=12, rotation=70, ha='right')
        ax[row][col].set_yticks([i + 1.5 for i in range(len(df_clean.index) - 1)])
        ax[row][col].set_yticklabels(df_clean.index[1:], fontsize=12, rotation=30)
        ax[row][col].set_xlabel(labels[feat], fontsize=16)

        ax[row][col].axhline(y=len(df_clean), xmin=0, xmax=1 / float(len(df_clean)) * (len(df_clean) - 1), color='k',
                             linewidth=1)
        ax[row][col].axvline(x=0, ymin=0, ymax=1 / float(len(df_clean)) * (len(df_clean) - 1), color='k', linewidth=1)

        for j in range(len(df_clean)):
            ax[row][col].axhline(y=j, xmin=0, xmax=1 / float(len(df_clean)) * j, color='k', linewidth=0.5)
            ax[row][col].axvline(x=len(df_clean) - j, ymin=0, ymax=1 / float(len(df_clean)) * j, color='k',
                                 linewidth=0.5)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)

    return fig, dfs


# returns df after removing all rows and cols with only missing values
def get_clean_values_df(dfs, feat):
    df = dfs[feat].copy()
    df[df > vmaxs[feat]] = np.nan
    df[df < vmins[feat]] = np.nan

    row_mask = df.isnull().all(0)

    clean_rows_df = df[~row_mask].copy()

    col_mask = clean_rows_df.isnull().all(1).reset_index()
    col_names = col_mask.loc[col_mask[0] == False, 'index'].values.tolist()

    clean_rows_cols_df = clean_rows_df.loc[:, col_names].copy()

    return clean_rows_cols_df


# returns python dict. with each rare variant in patient classified by its inheritance;
# ie from mom, da,d ocmp het , in sibling or de novo
def get_classified_muts_for_gene(gene, genes_dict):
    g = gene
    muts = {'all': [], 'mom': [], 'dad': [], 'de novo': [], 'others from mom': [], 'others from dad': []}

    for m in genes_dict[g]['Mutations']:
        mut = genes_dict[g]['Mutations'][m]['mut']

        if mut not in muts['all']:
            muts['all'].append((tuple(genes_dict[g]['Mutations'][m]['Rels']), mut))
        if mut not in muts['mom'] and 'mother' in genes_dict[g]['Mutations'][m]['Rels']:
            muts['mom'].append(mut)
        if mut not in muts['dad'] and 'father' in genes_dict[g]['Mutations'][m]['Rels']:
            muts['dad'].append(mut)
        if mut not in muts['de novo'] and genes_dict[g]['Mutations'][m]['Rels'] == []:
            muts['de novo'].append(mut)

        if set(genes_dict[g]['Mutations'][m]['Rels']).difference(['mother', 'father', 'proband']):

            if 'mother' in genes_dict[g]['Mutations'][m]['Rels'] and mut not in muts['others from mom']:
                muts['others from mom'].append(mut)

            if 'father' in genes_dict[g]['Mutations'][m]['Rels'] and mut not in muts['others from dad']:
                muts['others from dad'].append(mut)

    return muts


# returns color of node for gene, depending on the inheritance of the rare variants in th gene.
def get_color_for_gene_from_muts(gene, muts_dict, family_seq_result):
    g = gene
    muts = muts_dict

    col = ''

    if muts['mom']:
        if muts['others from mom']:
            if muts['dad']:
                if muts['others from dad']:
                    col = sib_color + ':' + sib_color

                else:
                    col = sib_color + ':' + mut_color

            elif family_seq_result['dad']:
                col = sib_color + ':' + wt_color

            else:
                col = sib_color + ':' + not_seq_color

        else:
            if muts['dad']:
                if muts['others from dad']:
                    col = mut_color + ':' + sib_color

                else:
                    col = mut_color + ':' + mut_color

            elif family_seq_result['dad']:
                col = mut_color + ':' + wt_color
            else:
                col = mut_color + ':' + not_seq_color
    elif family_seq_result['mom']:
        if muts['dad']:
            if muts['others from dad']:
                col = wt_color + ':' + sib_color

            else:
                col = wt_color + ':' + mut_color

        elif family_seq_result['dad']:
            col = wt_color + ':' + wt_color
        else:
            col = wt_color + ':' + not_seq_color
    else:
        if muts['dad']:

            if muts['others from dad']:
                col = not_seq_color + ':' + sib_color
            else:
                col = not_seq_color + ':' + mut_color
        elif family_seq_result['dad']:
            col = not_seq_color + ':' + wt_color
        else:

            col = not_seq_color + ':' + not_seq_color

    return col


if __name__ == "__main__":
    print('')
