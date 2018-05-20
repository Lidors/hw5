import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr


class VisualStimData:
    """
    Data and methods for the visual stimulus ePhys experiment.
    The data table itself is held in self.data, an `xarray` object.
    Inputs:
        data: xr.Dataset

    Methods:
        plot_electrode- see explanation below
        experimenter_bias-see explanation below


    """
    def __init__(self, data ):
        self.data = data


    def plot_electrode(self, rep_number=1 , rat_id=1, elec_number=(1,3)):
        """
        Plots the voltage of the electrodes in "elec_number" for the rat "rat_id" in the repetition
        "rep_number". Shows a single figure with subplots.

        -"elec_number" should be an tuple variable containing tow integers
         both values should be between 1 and 10 and will represent the rang of electrodes one would wish to expect
        -"rat_id" should be an int variable between 1 to 3 (as the rat_id rang)
        -"rep_number" should be an int variable between 1 to 4 (as the number of repetitions)

        """

        elec = np.arange(elec_number[0], elec_number[1] + 1) # the electrodes that will be ploted
        x_data = np.arange(0, 2, 2 / 1000)                   # time
        if len(elec)==1:
            y_data = self.data[rat_id][rep_number, elec].values # volt
            plt.plot(x_data,y_data)
        else:

            fig, axs = plt.subplots(len(elec), 1, sharex='all', sharey='all')
            for p, electrode in enumerate(elec):
                x_data = np.arange(0, 2, 2 / 1000)
                y_data = np.transpose(self.data[rat_id][rep_number , elec].values)
                axs[p].plot(x_data, y_data[:, p], linewidth=0.2)
                axs[p].set_title(f'Electrode num' + str(electrode))
        plt.show()

    def experimenter_bias(self):
        """ Shows the statistics of the average recording across all experimenters """

        # arranging the data in to 3 variables (mean_data,std_data,median_data)
        # each variable containg data on the 3 experimenters
        mean_data = self.data.mean(dim='elec_num').to_dataframe().mean(0)
        std_data = self.data.std(dim='elec_num').to_dataframe().mean(0)
        median_data = self.data.median(dim='elec_num').to_dataframe().mean(0)

        # ploting the data
        fig, axs = plt.subplots(3, 1, sharex='all', sharey='all')
        axs[0].bar(['Kandel', 'Huxley', 'Hodgkin'], mean_data, width=0.4)
        axs[0].set_title(f'mean')
        axs[1].bar(['Kandel', 'Huxley', 'Hodgkin'], std_data, width=0.4)
        axs[1].set_title(f'std')
        axs[2].bar(['Kandel', 'Huxley', 'Hodgkin'], median_data, width=0.4)
        axs[2].set_title(f'median')
        plt.show()


def mock_stim_data() -> VisualStimData:
    """ Creates a new VisualStimData instance with mock data """
    # crating the exp features(e.g rat ID,room temp...)

    data_feat = np.array([[1, 22, 40, 'Kandel', 1], [2, 23, 44, 'Huxley', 0], [3, 20, 41, 'Hodgkin', 0]])
    idx = ['rat_id', 'room_temp', 'room_hum', 'ex_name', 'rat_gen']

    # arranging the exp features in to Dict, later on each Dict will be add as an attribute to a data array

    exp1 = dict(pd.Series(data=data_feat[0], index=idx))
    exp2 = dict(pd.Series(data=data_feat[1], index=idx))
    exp3 = dict(pd.Series(data=data_feat[2], index=idx))
    all_feat = np.array([exp1, exp2, exp3])

    # creating a neural data for each exp
    ds = xr.Dataset()
    for j in range(3):

        spike = pd.Series([20, 60, 120, 80, 60, -25])  # A spike wave form
        volt = np.random.normal(-2, 2, 40000)  # random noise
        num_spike = np.random.randint(15, 105,
                                      1)  # chosing randomly the number of spikes that will acure during 4 trails
        spike_time = np.random.randint(1, 39939, num_spike)  # chosing randomly the time when spikes will acure

        for i in spike_time:
            idx = np.arange(i, i + 6)
            volt[idx] = volt[idx] + spike
        volt1 = np.reshape(volt, (4, 10, 1000))
        trail_num = ['rep1', 'rep2', 'rep3', 'rep4']
        elec_num = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10']
        stimuli = np.concatenate((np.zeros(500), np.ones(50), np.ones(450) * 2))
        da = xr.DataArray(volt1, coords=[trail_num, elec_num, stimuli]
                          , dims=['rep_num', 'elec_num', 'stimuli'], attrs=all_feat[j])

        ds[j] = da
    return ds

if __name__ == '__main__':
    data = mock_stim_data()
    VSdata = VisualStimData(data)
    VSdata.plot_electrode(2,1,(1,5))  # add necessary vars
    VSdata.experimenter_bias()
