from LM.lin_method_util import lm, dm

from tabulate import tabulate
from prefixed import Float
import numpy as np
import datetime
import os

def handle_results(SC, ENOB):
    NR = NumpyResults()

    NR.add(DC=SC.qconfig,       # DAC configuration
           DM=SC.dac.model, 
           LM=SC.lin.method, 
           fs=SC.fs, 
           fc=SC.fc, 
           f0=SC.carrier_freq, 
           Ncyc=SC.Ncyc, 
           ENOB=ENOB)
    
    NR.print(SC.qconfig, SC.dac.model, SC.lin.method)
    NR.save()
    NR.save_to_html()

class NumpyResults():
    def __init__(self, **kwargs) -> None:

        self.headers = ['Time (UTC)', 'Config', 'Method', 'Model', 'Fs', 'Fc', 'Fx', 'Ncyc', 'ENOB']

        main_script_path = os.path.abspath(__file__) 
        parent_dir = os.path.dirname(os.path.dirname(main_script_path))

        self.root = os.path.join(parent_dir, 'results')
        self.npy_path = os.path.join(self.root, 'npy')

        self.results_array = None
        self.results_dict = {}
        self.load()
        
    def load(self, filename='all_results'):
        results_file_path = os.path.join(self.npy_path, filename + '.npy')
        results_file_exist = os.path.exists(results_file_path)
        if results_file_exist:
            self.results_file_path = results_file_path
            self.results_array = np.load(self.results_file_path, allow_pickle=True)
            self.results_dict = self.array_to_dict(self.results_array)
            return True
        else:
            if (self.results_array is None):
                self.results_file_path = results_file_path

        return False

    def save(self, filename=None, path=None):
        if (self.results_dict is None):
            pass
        else:
            self.results_array = self.dict_to_array(self.results_dict)
            np.save(self.results_file_path, self.results_array)

    def array_to_dict(self, array=None):
        temp_dict = {}
        if (array is not None):

            for item in array:
                temp_dict[item[0]] = item[1]

        return temp_dict
        
    def dict_to_array(self, dictionary):
        result = dictionary.items()
        data = list(result)
        np_arr = np.empty(len(data), dtype=object)

        for i in range(len(data)):
            np_arr[i] = data[i]

        return np_arr

    def add(self, **kwargs):
        time_and_date = int(datetime.datetime.now(datetime.UTC).timestamp())
        DC = kwargs.get('DC', -1)
        DM = kwargs.get('DM', -1)
        LM = kwargs.get('LM', -1)
        fs = kwargs.get('fs', -1)
        fc = kwargs.get('fc', -1)
        f0 = kwargs.get('f0', -1)
        Ncyc = kwargs.get('Ncyc', -1)
        ENOB = kwargs.get('ENOB', -1)

        if (DC not in self.results_dict):
            results_array = np.zeros((2, 9, 9))
            self.results_dict[DC] = results_array

        DM_index = DM - 1 # Count starts from 1, but indexing from 0
        LM_index = LM - 1 # Count starts from 1, but indexing from 0

        self.results_dict[DC][DM_index, LM_index, 0] = time_and_date
        self.results_dict[DC][DM_index, LM_index, 1] = DC
        self.results_dict[DC][DM_index, LM_index, 2] = LM
        self.results_dict[DC][DM_index, LM_index, 3] = DM
        self.results_dict[DC][DM_index, LM_index, 4] = fs
        self.results_dict[DC][DM_index, LM_index, 5] = fc
        self.results_dict[DC][DM_index, LM_index, 6] = f0
        self.results_dict[DC][DM_index, LM_index, 7] = Ncyc
        self.results_dict[DC][DM_index, LM_index, 8] = ENOB

        # Add LM name/number to all rows

        for DM in range(2):
            for LM in range(9):
                self.results_dict[DC][DM, LM, 2] = LM + 1

    def remove(self):
        if (self.results_array is None):
            return False
        else:
            return True

    def data_array(self):
        pass

    def data_list(self):
        pass

    def print(self, DC=-1, DM=-1, LM=-1):
        if (DC not in self.results_dict):
            data_list = ['-1']*len(self.headers)
        else: 
            DM_index = DM - 1 # Count starts from 1, but indexing from 0
            LM_index = LM - 1 # Count starts from 1, but indexing from 0

            data_array = self.results_dict[DC][DM_index, LM_index, :]
            data_list = self.data_array_to_list(data_array)
        
        table_data = [self.headers, data_list]
        table_print = tabulate(table_data)
        print(table_print)
    
    def data_array_to_list(self, data_array):
        timestamp = int(data_array[0])
        if (timestamp == 0):
            time = 'Never'
        else:
            time = datetime.datetime.fromtimestamp(timestamp, tz=datetime.UTC)

        data_list = []
        data_list.append(str(time))
        data_list.append(str(int(data_array[1])))
        data_list.append(str(lm(data_array[2])))
        data_list.append(str(dm(data_array[3])))
        data_list.append(f'{Float(data_array[4]):.2h}')
        data_list.append(f'{Float(data_array[5]):.1h}')
        data_list.append(f'{Float(data_array[6]):.1h}')
        data_list.append(f'{int(data_array[7])}')
        data_list.append(f'{Float(data_array[8]):.3h}')        

        return data_list

    def save_to_html(self):
        html_string = ''
        Include_index = [0, 2, 4, 5, 6, 7, 8] # What data to include in the table

        for item in sorted(self.results_dict.items()):
            data_array = item[1]
            html_string += f'# DAC Configuration: {item[0]} \n'

            for DM_index, DM in enumerate(data_array): # Static or Simulation (Spectre or Ngspice)

                html_string += f'### Model: {str(dm(DM_index + 1))} \n'

                table_data = [[self.headers[i] for i in Include_index]]
                for LM in DM:
                    data_list = self.data_array_to_list(LM)
                    data_list = [data_list[i] for i in Include_index]
                    table_data.append(data_list)

                table_html = tabulate(table_data, headers="firstrow", tablefmt="html")

                html_string += table_html + '\n\n'

            html_string += '\n\n\n\n'

        if (html_string == ''):
            html_string += '# No results\n'

        filename = 'results'
        results_file_path = os.path.join(self.root, filename + '.md')
        f = open(results_file_path, "w")
        f.write(html_string)
        f.close()

if __name__ == '__main__':
    NR = NumpyResults()

    results_array = np.ndarray((2, 9, 9)) # DAC model (DM) (Static or Simulation (Spectre or Ngspice)) | Linearization method (LM) | Results / Data
    results_dict = {}
    results_dict[12] = results_array
    data = list(results_dict.items())

    np_arr = np.empty(len(data), dtype=object)

    for i in range(len(data)):
        np_arr[i] = data[i]

    print(np_arr)
    print('END')
    