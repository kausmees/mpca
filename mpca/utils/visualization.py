import numpy as np
import matplotlib.pyplot as plt
from .datahandling import get_pop_superpop_list
from matplotlib.colors import rgb2hex
import seaborn as sns
sns.set()


def plot_scores(scores_by_pop, style_dict, pop_superpop_file, markersize=80, figsize=(8,9)):
	'''
	Plot scores by population according to a given style.


	:param scores_by_pop: dict mapping population ID to list of scores
	:param style_dict: a dict mapping population ID to marker styles (from get_plot_style())
	:param pop_superpop_file: name of file mapping populations to superpopulations
	:param markersize: size of markers in plot
	'''
	sns.set_style(style="whitegrid", rc=None)
	plt.figure(figsize=figsize)
	superpop_dict = get_superpop_dict(pop_superpop_file)
	superpops = list(superpop_dict.keys())

	for spop in range(len(superpops)):
		this_pops = superpop_dict[superpops[spop]]

		for pop in this_pops:
			if pop in scores_by_pop.keys():
				this_fam_coords = np.array(scores_by_pop[pop])
				if len(this_fam_coords) > 0:
					plt.scatter(this_fam_coords[:,0], this_fam_coords[:,1], color=style_dict[pop][1], marker=style_dict[pop][0], s=markersize, edgecolors=style_dict[pop][2])



def get_plot_style(pop_superpop_file, legend_filename, num_colors=6, num_shapes = 16, num_columns = 7.0, width = 4.0, height=4.5, markersize = 300, fontsize=15):
	'''
	Get a dict containing a mapping of populations to marker styles, to plot samles according to population so that all
	populations from the same superpopulation (as defined in pop_superpop_file) have a similar color.

	A marker style is defined by a tuple (marker_shape, marker_color, marker_edge_color).

	Adjusting num_colors and num_shapes will change the combinations, and error will be thrown if there are not
	enough combinations for the number of populations in a superpopulation.

	Saves a separate image with the legend.
	Adjusting width, height, num_columns, marker_size and font_size will affect the layout of the legend image.

	:param pop_superpop_file: name of file mapping populations to superpopulations
	:param legend_filename: full filename to write legend to
	:param num_colors: how many colors to use from each color palette
	:param num_shapes: how many different shapes to use
	:param num_columns: number of columns to use in the legend image
	:param width: adjusting the width of the legend image
	:param height: adjusting the height of the legend image
	:param markersize: size of markers in the legend image
	:param fontsize: size of legend font
	:return: dict mapping population IDs to marker styles
	'''

	superpop_dict = get_superpop_dict(pop_superpop_file)
	pop_superpop_list = get_pop_superpop_list(pop_superpop_file)
	superpops = np.unique(pop_superpop_list[:,1])

	lens = [len(s) for s in superpops]
	max_len = max(lens)

	#################### modify these to create different style combinations ##################

	# one color palette per superpopulation
	palette_list = [sns.color_palette("Greys_r", num_colors),
				  sns.color_palette("Oranges_r", num_colors),
				  sns.color_palette("Blues", num_colors),
				  sns.color_palette("Greens_r", num_colors),
				  sns.color_palette("PiYG", num_colors * 2),
				  sns.color_palette("BrBG", num_colors * 2),
				  sns.color_palette("PRGn", num_colors * 2),
				  sns.color_palette("YlOrRd",2*3),
				  sns.color_palette("Reds", num_colors),
				  sns.cubehelix_palette(num_colors, reverse=True),
				  sns.light_palette("purple", num_colors),
				  sns.light_palette("navy", 5, reverse=True),
				  sns.light_palette("green", num_colors),
				  sns.light_palette((210, 90, 60), num_colors, input="husl")  #blue-ish
				  ]
	# marker shapes
	shape_list = ["o", "v","<", "s", "p","H" ,"p","x", "D","X","*","d",">","h","+"]

	if num_shapes <= len(shape_list):
		shape_list = shape_list[0:num_shapes]

	# marker edges
	edge_list = ["black", None, "red"]
	#############################################################################################

	# seaborn style
	sns.set_style(style="white", rc=None)


	max_num_pos = max([len(superpop_dict[spop]) for spop in superpops])
	pops_per_col = max_num_pos - 20


	style_dict = {}

	fig, axes = plt.subplots(figsize=(num_columns * width, height * (max_num_pos / 10)))
	col = 0.0
	num_legend_entries = 0
	legends = []

	for spop in range(len(superpops)):
		this_pops = superpop_dict[superpops[spop]]

		# counter of how many times same color is used: second time want to flip the shapes
		time_used = spop // len(palette_list)
		this_pop_color_list = list(map(rgb2hex, palette_list[spop % len(palette_list)][0:num_colors]))

		if time_used == 0:
			combos = np.array(np.meshgrid(shape_list,this_pop_color_list,edge_list)).T.reshape(-1,3)
		else:
			combos = np.array(np.meshgrid((shape_list[::-1]),this_pop_color_list,edge_list)).T.reshape(-1,3)

		assert len(combos) >= len(this_pops), "Not enough style combinations for {} populations in superpopulation {}. Add more colours/shapes/edges. ".format(len(this_pops), superpops[spop])

		this_superpop_points = []
		for p in range(len(this_pops)):
			assert not this_pops[p] in style_dict.keys()
			style_dict[this_pops[p]] = combos[p]
			point = plt.scatter([-1], [-1], color=combos[p][1], marker=combos[p][0], s=markersize, edgecolors=combos[p][2], label=this_pops[p])
			this_superpop_points.append(point)

		# if we swith to next column
		if num_legend_entries + len(this_pops) > pops_per_col:
			col +=1
			num_legend_entries = 0
		# left, bottom, right and top
		l = plt.legend(this_superpop_points, [p for p in this_pops],
					   title="  " + r'$\bf{'+superpops[spop]+'}$' +" "*int(max_len-lens[spop])*2+"\n",
					   bbox_to_anchor=(float(col) / (num_columns), 1 - float(num_legend_entries) / pops_per_col, 0, 0),
					   loc='upper left', fontsize=fontsize, markerfirst=True)


		l._legend_box.align = "left"
		l.get_title().set_fontsize(str(fontsize+1))
		num_legend_entries += len(this_pops) + 3
		legends.append(l)

	for l in legends:
		axes.add_artist(l)

	plt.xlim(left=0)
	plt.ylim(bottom=0)
	plt.xticks([])
	plt.yticks([])
	plt.savefig(legend_filename, bbox_inches="tight")
	plt.close()

	return style_dict



def get_scores_by_pop(scores, ind_pop_list):
	'''
	Get a dict mapping population IDs to a list of scores of samples from that population.


	:param scores: scores of samples
	:type scores: array, shape (n_samples x n_dimensions)
	:param ind_pop_list: array mapping individual IDs to populations so that ind_pop_list[i,0] is the individual ID
		   of sample i, and ind_pop_list[i,1] is the population of sample i, in the same order as in scores
	:type ind_pop_list:  array, shape (n_samples x 2)
	:return: a dictionary mapping each population in ind_pop_list to a list of scores of samples from that population
	'''
	pop_list = ind_pop_list[:,1]
	unique_pops = np.unique(pop_list)
	scores_by_pop = {}

	for p in unique_pops:
		scores_by_pop[p] = []

	for s in range(len(scores)):
		this_pop = pop_list[s]
		this_coords = scores[s]
		scores_by_pop[this_pop].append(this_coords)

	return scores_by_pop


def get_superpop_dict(pop_superpop_file):
	'''
	Get a dict mapping superpopulation IDs to a list of populations IDs belonging to that superpopulation.

	Assumes file contains one population and superpopulation per line, separated by ","  e.g.

	Kyrgyz,Central/South Asia

	Khomani,Sub-Saharan Africa

	:param pop_superpop_file: name of file mapping populations to superpopulations
	:return: a dictionary mapping each superpopulation ID in the given file to a list of its subpopulations
	'''
	pop_superpop_list = get_pop_superpop_list(pop_superpop_file)
	superpops = np.unique(pop_superpop_list[:,1])
	superpop_dict = {}

	for spop in superpops:
		superpop_dict[spop] = []

	for i in range(len(pop_superpop_list)):
		superpop_dict[pop_superpop_list[i][1]].append(pop_superpop_list[i][0])

	return superpop_dict

def connectpoints(a, b):
	'''
	Plot a line between two points.

	:param a: a 2-d point (x,y)
	:param b: a 2-d point (x,y)
	'''
	plt.plot([a[0], b[0]], [a[1], b[1]], 'k-', linewidth=0.9, alpha=0.8)
