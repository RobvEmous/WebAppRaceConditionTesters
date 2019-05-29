import copy
import itertools

import time
import warnings

import math
import os
import subprocess
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from fitter import Fitter
import statistics as st

from tqdm import tqdm


def read_item(file):
    item_title = next(file, "").strip().replace(":", "")
    if item_title == "":
        return None, None
    items = {}
    current = next(file, "").strip()
    while current != "":
        parts = current.split(":\t")
        items[parts[0]] = parts[1]
        current = next(file, "").strip()
    return item_title, items


def summarize_all_totals():
    command = "find csvs -name *totals.txt"
    result_file_list = subprocess.check_output(command, encoding='UTF-8', shell=True)

    all_items = defaultdict(dict)
    for result_file in result_file_list.split("\n"):
        if not result_file:
            continue
        tool_name = result_file.split("/")[1]
        test_type = result_file.split("/")[2]
        if test_type.endswith("+"):
            tool_name += "+"
            test_type = test_type[:-1]
        all_items[test_type][tool_name] = dict()
        with open(result_file, 'r') as file:
            next(file)  # skip first four lines
            next(file)
            next(file)
            next(file)
            while True:
                item_title, items = read_item(file)
                if item_title is None:
                    break
                all_items[test_type][tool_name][item_title] = items

    with open("csvs/totals.csv", 'w') as file:
        # print header
        gather_topics = ["Success codes", "Vouchers used", "Ratio", "Local", "Webserver", "Appserver", "Diff-times"]
        data_types = ["min", "max", "avg", "med", "std"]
        header_0 = ";".join(
            ["Test type", "Tool name"] + [topic + ";" * (len(data_types) - 1) for topic in gather_topics])
        header_1 = ""
        i = 0
        for j, header in enumerate(header_0.split(";")):
            if header == "" or (len(header_0.split(";")) > j + 1 and header_0.split(";")[j + 1] == ""):
                header_1 += data_types[i % len(data_types)] + ";"
                i += 1
            else:
                header_1 += ";"

        file.write(header_0 + "\n")
        file.write(header_1 + "\n")

        # print data
        for tool_name in sorted(list(all_items.keys())):
            for test_type in all_items[tool_name]:
                file.write(tool_name + ";" + test_type)
                for item_title in all_items[tool_name][test_type]:
                    items = list(all_items[tool_name][test_type][item_title].values())
                    file.write((";" + ";".join(items)).replace(".", ","))
                file.write("\n")


def read_diff_item(file):
    item_title = next(file, "").strip().replace(":", "").replace("\\", "")
    if item_title == "":
        return None, None
    items = []
    current = next(file, "").strip()
    while current != "":
        items.append(current)
        current = next(file, "").strip()
    return item_title, items


##### time diff analysis withing attack
def time_diff_analysis(all_items, test_type, item_type, log_scale=False):
    plt.plot([st.median([float(p) for p in item if p is not None]) for item in list(itertools.zip_longest(
        *[all_items[item_type][test_type]['CR'][i:i + 24] for i in range(0, len(all_items['Ratio']['r']['CR']), 24)]))],
             label='CR')
    plt.plot([st.median([float(p) for p in item if p is not None]) for item in list(itertools.zip_longest(
        *[all_items[item_type][test_type]['CR+'][i:i + 24] for i in range(0, len(all_items['Ratio']['r']['CR+']), 24)]))],
             label='CR+')
    plt.plot([st.median([float(p) for p in item if p is not None]) for item in list(itertools.zip_longest(
        *[all_items[item_type][test_type]['RTW'][i:i + 24] for i in range(0, len(all_items['Ratio']['r']['RTW']), 24)]))],
             label='RTW')
    plt.plot([st.median([float(p) for p in item if p is not None]) for item in list(itertools.zip_longest(
        *[all_items[item_type][test_type]['SR'][i:i + 24] for i in range(0, len(all_items['Ratio']['r']['SR']), 24)]))],
             label='SR')
    plt.plot([st.median([float(p) for p in item if p is not None]) for item in list(itertools.zip_longest(
        *[all_items[item_type][test_type][' TI'][i:i + 24] for i in range(0, len(all_items['Ratio']['r']['TI']), 24)]))],
             label='TI')
    plt.legend()
    plt.title(f"{test_type}-{item_type}")
    if log_scale:
        plt.yscale("log")
    plt.show()


def summarize_all_diffs():
    command = "find csvs -name *diffs.txt"
    result_file_list = subprocess.check_output(command, encoding='UTF-8', shell=True)

    all_items = defaultdict(dict)
    max_num_items = defaultdict(int)
    for result_file in result_file_list.split("\n"):
        if not result_file:
            continue
        tool_name = result_file.split("/")[1].replace("2", "")
        test_type = result_file.split("/")[2]
        if test_type.endswith("+"):
            tool_name += "+"
            test_type = test_type[:-1]
        with open(result_file, 'r') as file:
            while True:
                item_title, items = read_diff_item(file)
                if item_title is None:
                    break
                if test_type not in all_items[item_title]:
                    all_items[item_title][test_type] = dict()
                all_items[item_title][test_type][tool_name] = items
                if len(items) > max_num_items[item_title]:
                    max_num_items[item_title] = len(items)

    items_titles = list(all_items.keys())
    test_types = sorted(list(all_items[items_titles[0]].keys()))
    tool_names = sorted(list(all_items[items_titles[0]][test_types[0]].keys()))

    for item_title in items_titles:
        with open(f"csvs/diffs_{item_title.replace(' ', '_')}.csv", 'w') as file:
            # print header
            header_0 = ";".join([type + ";" * (len(tool_names) - 1) for type in test_types])
            header_1 = ""
            i = 0
            for j, header in enumerate(header_0.split(";")):
                if header == "" or (len(header_0.split(";")) > j + 1 and header_0.split(";")[j + 1] == ""):
                    header_1 += tool_names[i % len(tool_names)] + ";"
                    i += 1
                else:
                    header_1 += ";"
            file.write(header_0 + "\n")
            file.write(header_1 + "\n")

            # print datamax_num_items
            for print_index in range(max_num_items[item_title]):
                first_in_row = True
                for test_type in test_types:
                    for tool_name in tool_names:
                        if first_in_row:
                            first_in_row = False
                        else:
                            file.write(";")
                        if len(all_items[item_title][test_type][tool_name]) > print_index:
                            file.write(str(all_items[item_title][test_type][tool_name][print_index]).replace(".", ","))
                file.write("\n")

    # calculate advanced statistics
    test_types = ['f', 'r', 'n', 's']
    sign_level = 0.05
    for test_type in test_types:
        time.sleep(0.25)
        print("\n---------------------------")
        print("------------ " + test_type + " ------------")
        print("---------------------------\n")
        time.sleep(0.25)
        for item_type in tqdm(all_items):
            if test_type not in all_items[item_type]:
                continue
            # clean the data
            dataset = []
            part_to_use = all_items[item_type][test_type]
            for item in sorted(part_to_use):
                data = sorted(list(map(float, part_to_use[item])), reverse=True)
                for i in range(len(data)):
                    if data[i] < 0.0005:
                        data[i] = 0.0005
                dataset.append(data)

            the_type = item_type
            log = False
            if item_type in ['Local', 'Web', 'App', 'Diff']:
                try:
                    dataset = [[np.log10(item) for item in data if item] for data in dataset]
                except RuntimeWarning as e:
                    print(e)
                    print(dataset)
                the_type += " time-diff (log10)"
                log = True
            remove_outliers_and_get_best_dist_2(dataset, list(sorted(part_to_use)), the_type, test_type, log)

            plot_log_hist(dataset, list(sorted(part_to_use)), item_type + " time-diff (log10)")


def remove_normal_outliers(data, mu, sigma, num_sigm):
    new_data = []
    for item in data:
        if abs(item - mu) <= num_sigm * sigma:
            new_data.append(item)
    return new_data


def get_statistics_from_diffs(diffs):
    the_mean = st.mean(diffs)
    return {'min': min(diffs), 'max': max(diffs),
            'mean': the_mean, 'median': st.median(diffs), 'stdev': st.stdev(diffs, the_mean),
            'q1': np.percentile(diffs, 25), 'q3': np.percentile(diffs, 75)}


def lines_with_labels(axis, x_poss, labels, colors, extra_space, font_size, logarithmic=False, rotated=True):
    items = sorted(list(zip(labels, colors, x_poss)), key=lambda x: x[2])

    middle_index = int((len(items) - 1) / 2)
    new_positions = {items[middle_index][0]: items[middle_index][2]}

    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()
    min_diff = (xmax - xmin) / 17

    last_pos = items[middle_index][2]
    # left positioning
    for i in range(middle_index - 1, -1, -1):
        new_positions[items[i][0]] = min(last_pos - min_diff, items[i][2])
        last_pos = new_positions[items[i][0]]
    # right positioning
    last_pos = items[middle_index][2]
    for i in range(middle_index + 1, len(items)):
        new_positions[items[i][0]] = max(last_pos + min_diff, items[i][2])
        last_pos = new_positions[items[i][0]]
    correction = min_diff / 1.7
    for item in sorted(items, key=lambda x: x[0]):
        if item[0] in ['Q1', 'Q3']:
            axis.axvline(item[2], color=item[1], linestyle='dashed', linewidth=1, alpha=0.7,
                         label=f"Q1 / Q3" if item[0] != "Q3" else "")
        else:
            axis.axvline(item[2], color=item[1], linewidth=1, alpha=0.7, label=item[0])
        if logarithmic:
            label = 10 ** item[2]
        else:
            label = item[2]
        axis.text(new_positions[item[0]] - correction,
                  -1 * (ymax - ymin) / (10 - 6.5 * extra_space),
                  f"{label:.3f}", fontdict={'fontsize': font_size},
                  color=item[1], rotation=-90 if rotated else 0, zorder=10)


def remove_outliers_and_get_best_dist_2(dataset, labels, item_type, test_type, logarithmic=False):
    font_size = 9
    min_val = min([min(data) for data in dataset])
    max_val = max([max(data) for data in dataset])
    num_bins = 2 * math.ceil(math.sqrt(max([len(data) for data in dataset])))
    xs_hist = np.linspace(min_val, max_val, num_bins)

    num_rows = 2
    num_cols = math.ceil(len(dataset) / 2)
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols,
                            figsize=(3 * num_cols, 3 * num_rows), sharex=True, sharey=True)
    plt.rc('legend', **{'fontsize': font_size})

    curr_row = num_rows - 1
    curr_col = 0
    axs[curr_row][curr_col].set_title("Totals", fontsize=17)
    axs[curr_row][curr_col].tick_params(axis='both', which='both', labelsize=font_size)
    axs[curr_row][curr_col].set_xlabel(item_type)
    axs[curr_row][curr_col].set_ylabel("Percentage of total (%)")
    axs[curr_row][curr_col].yaxis.set_tick_params(which='both', labelbottom=True)
    axs[curr_row][curr_col].margins(x=0.05)
    axs[curr_row][curr_col].spines['right'].set_visible(False)
    axs[curr_row][curr_col].spines['top'].set_visible(False)
    for i, data in enumerate(dataset):
        # plot the histogram
        weights = np.ones_like(data) / float(len(data))
        n, _, _ = axs[curr_row][curr_col].hist(data, weights=weights, alpha=0.7, bins=xs_hist, label=labels[i])

    stats = get_statistics_from_diffs(list(itertools.chain.from_iterable(dataset)))
    lines_with_labels(axs[curr_row][curr_col], [stats['median']],
                      [f"Median"],
                      ['blue'],
                      True, font_size, logarithmic, False)
    axs[curr_row][curr_col].legend(loc='upper right', bbox_to_anchor=(1.17, 1.03), framealpha=0.7)
    axs[curr_row][curr_col].patch.set_visible(False)

    for i, data in enumerate(dataset):
        curr_row = math.floor(i / num_cols)
        curr_col = i % num_cols
        if curr_row == num_rows - 1:
            curr_col += 1
        # adjust plot settings
        axs[curr_row][curr_col].set_title(labels[i], fontsize=17)
        axs[curr_row][curr_col].tick_params(axis='both', which='both', labelsize=font_size)
        axs[curr_row][curr_col].margins(x=0.05)
        axs[curr_row][curr_col].spines['right'].set_visible(False)
        axs[curr_row][curr_col].spines['top'].set_visible(False)
        # plot the histogram
        weights = np.ones_like(data) / float(len(data))
        n, _, _ = axs[curr_row][curr_col].hist(data, weights=weights, color='black', alpha=0.7, bins=xs_hist)

    total_items = 6  # len(dataset)
    for i, data in enumerate(dataset):
        curr_row = math.floor(i / num_cols)
        curr_col = i % num_cols
        if curr_row == num_rows - 1:
            curr_col += 1
        stats = get_statistics_from_diffs(data)
        if logarithmic:
            stats['stdev'] = 10 ** stats['stdev']
        lines_with_labels(axs[curr_row][curr_col], [stats['mean'], stats['median'], stats['q1'], stats['q3']],
                          [f"Mean\n({stats['stdev']:.3f})", "Median", "Q1", "Q3"],
                          ['red', 'blue', 'green', 'green'],
                          i + num_cols >= total_items,  # len(dataset),
                          font_size, logarithmic)
        if curr_col == 0:
            axs[curr_row][curr_col].set_ylabel("Percentage of total (%)")
            axs[curr_row][curr_col].yaxis.set_tick_params(which='both', labelbottom=True)
        if i + num_cols >= total_items:  # len(dataset):
            axs[curr_row][curr_col].set_xlabel(item_type)
            axs[curr_row][curr_col].xaxis.set_tick_params(which='both', labelbottom=True)
            axs[curr_row][curr_col].set_zorder(1)
        axs[curr_row][curr_col].legend(loc='upper right', bbox_to_anchor=(1.17, 1.1), framealpha=0.7)
        axs[curr_row][curr_col].patch.set_visible(False)
        # add more bins
        axs[curr_row][curr_col].locator_params(axis='x', nbins=8)
        axs[curr_row][curr_col].locator_params(axis='y', nbins=8)

    #if len(dataset) % 2 != 0:
    #    axs[num_rows - 1][num_cols - 1].spines['right'].set_visible(False)
    #    axs[num_rows - 1][num_cols - 1].spines['top'].set_visible(False)
    #    axs[num_rows - 1][num_cols - 1].spines['left'].set_visible(False)
    #    axs[num_rows - 1][num_cols - 1].spines['bottom'].set_visible(False)
    #    axs[num_rows - 1][num_cols - 1].xaxis.set_tick_params(which='both', bottom=False, labelbottom=False)
    #    axs[num_rows - 1][num_cols - 1].yaxis.set_tick_params(which='both', bottom=False, labelbottom=False)

    path = f"figures/{test_type}_{item_type}.png".replace(" ", "_")
    plt.subplots_adjust(left=0.08, right=0.94, top=0.93, bottom=0.17, hspace=0.44)
    plt.savefig(path, dpi=200)
    plt.close()


dists = ['foldcauchy', 'cauchy', 't', 'gennorm', 'johnsonsu', 'loglaplace',
         'burr12', 'dweibull', 'fisk', 'burr', 'alpha', 'laplace', 'genextreme',
         'invweibull', 'invgamma', 'betaprime', 'exponweib', 'powerlognorm', 'moyal',
         'johnsonsb', 'lognorm', 'exponnorm', 'invgauss', 'genlogistic', 'weibull_max',
         'frechet_l', 'gumbel_r', 'fatiguelife', 'dgamma', 'hypsecant', 'wald', 'kappa3',
         'gilbrat', 'beta', 'pearson3', 'genhalflogistic', 'halflogistic', 'logistic',
         'norm']


def remove_outliers_and_get_best_dist(dataset, labels, item_type, test_type, always_n=True, a=0.05):
    min_val = min([min(data) for data in dataset])
    max_val = max([max(data) for data in dataset])
    max_val += (max_val - min_val) / 4
    num_bins = 2 * math.ceil(math.sqrt(max([len(data) for data in dataset])))
    xs_hist = np.linspace(min_val, max_val, num_bins)
    min_bins = 100
    if num_bins < min_bins:
        xs_plot = np.linspace(min_val, max_val, min_bins)
    else:
        xs_plot = xs_hist

    num_cols = 1
    num_rows = len(dataset)
    fig, axs = plt.subplots(nrows=num_cols, ncols=num_rows,
                            figsize=(3 * len(dataset), 3), sharex=True, sharey=True)
    plt.rc('legend', **{'fontsize': 7})

    results = {}
    print("\n------- " + item_type + " -------\n")
    for i, data in enumerate(dataset):
        # check normality
        (mu, sigma) = st.norm.fit(data)
        (pvalue, statistics) = st.kstest(data, 'norm', (mu, sigma))
        print(f"{labels[i]}\tnorm\t\t{pvalue:.4f} ({mu:.4f}, {sigma:.4f})", end="")
        if always_n or pvalue >= a:
            print(f"\t-> Distribution assumed normal!", end="")
            if always_n and pvalue < a:
                print(" (Forced)")
            else:
                print("")
            pdf = st.norm.pdf(xs_plot, mu, sigma)
            axs[i].hist(data, color='black', density=True, alpha=0.7, bins=xs_hist)
            axs[i].plot(xs_plot, pdf, 'r', linewidth=1, label='norm')
            title = f"{labels[i]} ({mu:.4f}, {sigma:.4f})"
            if always_n and pvalue < a:
                title += "*"
            axs[i].set_title(title)
            axs[i].set_xlabel(f"{item_type}\n")
            axs[i].legend(loc="upper right")

            # store results
            res = {'norm': {'sqe': None,
                            'pval': pvalue,
                            'params': (mu, sigma),
                            'pdf': pdf,
                            'x': xs_hist}
                   }
            results[labels[i]] = {'data': copy.deepcopy(data),
                                  'results': copy.deepcopy(res)}
        else:
            print(f"\t-> Distribution assumed not-normal! Fitting {len(dists)} distributions..")

            # get best fitting distribution (these 38 dists got less than 5 sum sq error on the first dataset)
            f = Fitter(data, verbose=False, xmin=min_val, xmax=max_val, bins=num_bins, distributions=dists)
            # perform fittings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                time.sleep(0.25)
                f.fit()
                time.sleep(0.25)

            # plot results
            axs[i].hist(data, bins=f.bins, density=True, color='black', alpha=0.7)
            best_ten = f.df_errors.sort_values(by='sumsquare_error')[:10].T.to_dict('list')
            for name in best_ten.keys():
                axs[i].plot(f.x, f.fitted_pdf[name], lw=1, label=name)
            axs[i].set_title(labels[i])
            axs[i].set_xlabel(item_type)
            leg = axs[i].legend(loc="upper right")

            # store results
            for name in best_ten.keys():
                best_ten[name] = {'sqe': best_ten[name],
                                  'pval': st.kstest(data, name, f.fitted_param[name]).pvalue,
                                  'params': f.fitted_param[name],
                                  'pdf': f.fitted_pdf[name],
                                  'x': f.x}
            results[labels[i]] = {'data': copy.deepcopy(data),
                                  'results': copy.deepcopy(best_ten)}

    path = f"figures/{test_type}_{item_type}"
    if always_n:
        path += "_always_n"
    path += ".png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()

    # best fitting distributions for all data
    print("\nBest distributions:")
    total = set(list(results.items())[0][1]['results'])
    for res in results.items():
        total = total.intersection(res[1]['results'])
    total = list(total)
    print(total)

    if len(total) > 1 or (len(total) > 0 and total[0] != 'norm'):
        print("")
        for res in results.items():
            for name in total:
                print(f"{res[0]}\t{name}\t\t{res[1]['results'][name]['pval']} ", end="")
                if res[1]['results'][name]['pval'] < a:
                    print("(False)")
                else:
                    print("")

    # variance equality test (Levene) - not-normal
    print("\nTest mean and variance equality:")
    equal_mean = defaultdict(list)
    equal_var = defaultdict(list)
    printed = False
    for i in range(len(dataset)):
        for j in range(i + 1, len(dataset)):
            p_value_var = st.levene(dataset[i], dataset[j]).pvalue  # uses median-based method
            p_val_tt_mean = st.ttest_ind(dataset[i], dataset[j], equal_var=True).pvalue
            p_val_wt_mean = st.ttest_ind(dataset[i], dataset[j], equal_var=False).pvalue
            if p_value_var > a:
                equal_var[labels[i]].append((labels[j], p_value_var))
                if p_val_tt_mean > a:
                    equal_mean[labels[i]].append((labels[j], p_val_tt_mean, "tt"))
            elif p_val_wt_mean > a:
                equal_mean[labels[i]].append((labels[j], p_val_wt_mean, "wt"))
            if p_value_var < a or p_val_wt_mean < a:
                printed = True
                print(f"{labels[i]} - {labels[j]} \tvar: {p_value_var:.4E} ({p_value_var > a})\t", end="")
                print(f"\ttt-mu: {p_val_tt_mean:.4E}\twt-mu: {p_val_wt_mean:.4E} ({p_val_wt_mean > a})")
    if not printed:
        print("\tAll mean and vars match!")

    tab = "\t"
    print("\nTest variance results:")
    for key in equal_var.keys():
        print(
            f"{key} ->\t{(os.linesep + tab + tab).join([str(item) for item in sorted(equal_var[key], key=lambda x: x[1], reverse=True)])}")

    print("\nTest mean results:")
    for key in equal_mean.keys():
        print(
            f"{key} ->\t{(os.linesep + tab + tab).join([str(item) for item in sorted(equal_mean[key], key=lambda x: x[1], reverse=True)])}")

    return results


def remove_normal_outliers_plots(data, label, num_sigm):
    (mu_tmp, sigma_tmp) = st.norm.fit(data)
    new_dataset = remove_normal_outliers(data, mu_tmp, sigma_tmp, num_sigm)
    xs = np.linspace(min(data) - 1, max(data) + 1, 50)
    plt.hist(data, color='r', density=True, alpha=0.7, bins=xs)
    plt.hist(new_dataset, color='g', density=True, alpha=0.7, bins=xs)
    (mu, sigma) = st.norm.fit(new_dataset)
    plt.plot(xs, st.norm.pdf(xs, mu_tmp, sigma_tmp), 'r--', linewidth=2)
    plt.plot(xs, st.norm.pdf(xs, mu, sigma), 'g--', linewidth=2)

    plt.title(f"{label} - mu:{mu}, sig:{sigma}")
    plt.show()
    print(f"{label} - mu:{mu}\tsig:{sigma}")
    print(st.kstest(new_dataset, 'norm', st.norm.fit(new_dataset)))

    return mu, sigma, new_dataset


def plot_norm_log_hist(dataset, labels):
    min_val = np.log(min([min(data) for data in dataset]))
    max_val = np.log(max([max(data) for data in dataset]))
    # Fit a normal distribution to the data:
    # dataset = [[np.log(item) for item in data if item] for data in dataset]

    for i, data in enumerate(dataset):
        remove_normal_outliers_plots(data, labels[i], 4)

    # Plot the histogram.
    xs = np.logspace(min_val, max_val, 150)
    for data in dataset:
        plt.hist(data, alpha=0.7, bins=xs)
    # plt.gca().set_xscale("log")
    plt.legend(labels)
    # plt.show()

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 150)
    for i in range(len(all_mu)):
        # dist = lognorm(loc=np.exp(all_mu[i]), s=all_std[i])
        # p = lognorm.pdf(xs, all_mu[i], all_std[i])
        # p = dist.pdf(x)

        plt.plot(x, p, 'k', linewidth=2)

    title = "Fit results: mu = %.2f,  std = %.2f" % (0, 0)
    plt.title(title)
    plt.gca().set_xscale("log")
    plt.show()


def plot_log_hist(dataset, labels, test_type):
    min_val = min([min(data) for data in dataset])
    max_val = max([max(data) for data in dataset])
    num_bins = 2 * math.ceil(math.sqrt(max([len(data) for data in dataset])))

    num_cols = 1
    num_rows = len(dataset)
    fig, axs = plt.subplots(nrows=num_cols, ncols=num_rows,
                            figsize=(3 * len(dataset), 3 + 1),
                            sharex=True, sharey=True)
    for i, data in enumerate(dataset):
        axs[i].hist(
            data, alpha=1, bins=np.linspace(min_val, max_val, num_bins))
        axs[i].set_title(labels[i])
        axs[i].set_xlabel(test_type)

    plt.savefig(f"figures/{test_type}.png", dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()


summarize_all_totals()
summarize_all_diffs()

# if logarithmic:
# axs[curr_row][curr_col].set_xticklabels(axs[curr_row][curr_col].get_xticks())
# labels = axs[curr_row][curr_col].get_xticklabels()
# labels_new = [f"10^{label._text}" for label in labels]
# for i, label_new in enumerate(labels_new):
#     labels[i]._text = label_new
# axs[curr_row][curr_col].set_xticklabels(labels, fontdict={'fontsize': font_size}, minor=False)