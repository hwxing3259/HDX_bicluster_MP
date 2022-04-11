from HDX_MP2_base import *
import pandas
import sys
# take argument from cmd
rep = int(sys.argv[1])
random.seed(314159 + 100000*rep)
np.random.seed(314159 + 100000*rep)


if __name__ == "__main__":
    # ############### Getting the data #####################################################################################
    protein1 = pandas.read_csv("./protein/N64184_1a2_state.csv")[["Start", "End", "State", "Exposure", "Uptake", "Sequence"]]
    protein2 = pandas.read_csv("./protein/N64184_1b_state.csv")[["Start", "End", "State", "Exposure", "Uptake", "Sequence"]]
    protein2 = protein2[protein2.State != 'apo']  # duplicated apo measurements, throw away for now
    protein = pandas.concat([protein1, protein2], axis=0)
    truncated_sequence_length = [len(_[:-2].replace("P", "")) for _ in list(protein['Sequence'])]
    protein["Uptake"] = protein["Uptake"]/truncated_sequence_length  # uptake per exchangeable residue
    protein = protein.sort_values(by=['Start', 'End', 'State', 'Exposure'])
    protein = protein.reset_index(drop=True)

    interval_list = (protein[["Start", "End"]]).values.tolist()
    unique_interval_list = []
    unique_interval_idx = []
    for i, interval in enumerate(interval_list):
        if len(unique_interval_list)==0 or str(interval) not in unique_interval_list:
            unique_interval_list += [str(interval)]
            unique_interval_idx += [i]
    unique_elements = len(unique_interval_list)

    label_to_interval = {i: interval for i, interval in enumerate(unique_interval_list)}
    interval_to_label = {interval: i for i, interval in enumerate(unique_interval_list)}

    label_list = list(label_to_interval.keys())
    protein["label"] = [interval_to_label[str(_)] for _ in interval_list]
    protein = protein.drop(["Start", "End", "Sequence"], axis=1)
    incomplete = np.where(np.array(unique_interval_idx[1:] + [len(interval_list)]) - np.array(unique_interval_idx) != 54)[0]

    # final data set has dimension 111 proteins * 18 treatments * length 2 time series
    protein = protein.pivot_table(values="Uptake", index=["label", "Exposure"], columns="State")
    protein_np2 = np.swapaxes(protein.to_numpy().reshape((111, 3, 18)), 1, 2)

    # actually, since all responses starts at 0, should we just ignore them and only work with t=0.5,5?
    protein_np2 = protein_np2[:, :, 1:]
    # it is a 111*18*2 array with some NaN in it
    # since there is no repeated experiments, row and columns links are simple
    row_link = [[i] for i in range(len(protein_np2))]
    column_link = [[i] for i in range(len(protein_np2[0]))]

    check_point = 'check_point_constrained_{}'.format(rep)

    partition_parallel_tempering(protein_np2, [0.77, 0.79, 0.81, 0.83, 0.85, 0.87, 0.89, 0.92, 0.94, 0.96, 0.98, 1.0], 
                                 row_link=row_link, col_link=column_link,
                                 maxGibbsIteration_coord=1, thin=1, budget=1.0, max_init_cut=2, scale=True,
                                 burnin1=1000, burnin2=1000, p_row=0.2, p_col=0.2, random_MP_init=True, maxIteration_step=2,
                                 maxIteration=500, new_start=True, likelihoodFileName=None,
                                 representationFileName=None, sigFileName=None, budgetFileName=None,
                                 fixed_ordering_col=False, fixed_ordering_row=True,
                                 check_point=check_point, budget_update=True)  # CONSTRAINED

