from LM.lin_method_util import lm, dm

from tabulate import tabulate
from prefixed import Float
import datetime
import json
import os

def handle_results(SC, ENOB):
    JR = JSON_results()

    JR.add(DC=SC.qconfig,       # DAC configuration
           DM=SC.dac.model, 
           LM=SC.lin.method, 
           fs=SC.fs, 
           fc=SC.fc, 
           f0=SC.carrier_freq, 
           Ncyc=SC.Ncyc, 
           ENOB=ENOB)
    
    JR.print(SC.qconfig, SC.dac.model, SC.lin.method)
    JR.save()
    JR.save_to_html()

class JSON_results():
    def __init__(self, **kwargs) -> None:

        self.headers = ['Time (UTC+00:00)', 'Config', 'Method', 'Model', 'Fs', 'Fc', 'Fx', 'Ncyc', 'ENOB']

        main_script_path = os.path.abspath(__file__) 
        parent_dir = os.path.dirname(os.path.dirname(main_script_path))

        self.root = os.path.join(parent_dir, 'results')
        self.json_path = os.path.join(self.root, 'json')

        self.results_dict = {}
        self.load()
        
    def load(self, filename='all_results'):
        results_file_path = os.path.join(self.json_path, filename + '.json')
        results_file_exist = os.path.exists(results_file_path)
        if results_file_exist:
            self.results_file_path = results_file_path

            # json_str = json.dumps(self.results_dict)
            f = open(self.results_file_path, "r")
            json_str = f.read()
            f.close()

            self.results_dict = json.loads(json_str)
            return True
        else:
            if (len(self.results_dict.items()) == 0):
                self.results_file_path = results_file_path

        return False

    def save(self, filename=None, path=None):
        if (self.results_dict is None):
            pass
        else:
            json_str = json.dumps(self.results_dict, indent=4, sort_keys=True)
            f = open(self.results_file_path, "w")
            f.write(json_str)
            f.close()

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

        DC_JSON_key = str(DC)

        if (DC_JSON_key not in self.results_dict):
            self.results_dict[DC_JSON_key] = self.create_list_array()

        DM_index = DM - 1 # Count starts from 1, but indexing from 0
        LM_index = LM - 1 # Count starts from 1, but indexing from 0

        self.results_dict[DC_JSON_key][DM_index][LM_index][0] = time_and_date
        self.results_dict[DC_JSON_key][DM_index][LM_index][1] = DC
        self.results_dict[DC_JSON_key][DM_index][LM_index][2] = LM
        self.results_dict[DC_JSON_key][DM_index][LM_index][3] = DM
        self.results_dict[DC_JSON_key][DM_index][LM_index][4] = fs
        self.results_dict[DC_JSON_key][DM_index][LM_index][5] = fc
        self.results_dict[DC_JSON_key][DM_index][LM_index][6] = f0
        self.results_dict[DC_JSON_key][DM_index][LM_index][7] = Ncyc
        self.results_dict[DC_JSON_key][DM_index][LM_index][8] = ENOB

        # Add LM name/number to all rows

        for DM in range(2):
            for LM in range(9):
                self.results_dict[DC_JSON_key][DM][LM][2] = LM + 1

    # TODO: Make it so that these ranges, or, the numbers within them are set when this object is instanciated.
    def create_list_array(self):
        # DAC model (DM) (Static or Simulation (Spectre or Ngspice)) | Linearization method (LM) | Results / Data
        # results_array = [[[0]*9]*9]*2 # np.zeros((2, 9, 9))
        empty_list = []
        for DM in range(2):
            DM_list = []
            empty_list.append(DM_list)
            for LM in range(9):
                LM_list = []
                DM_list.append(LM_list)
                for data in range(9):
                    LM_list.append(0)
        return empty_list

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
        DC_JSON_key = str(DC)

        if (DC_JSON_key not in self.results_dict):
            data_list = ['-1']*len(self.headers)
        else: 
            DM_index = DM - 1 # Count starts from 1, but indexing from 0
            LM_index = LM - 1 # Count starts from 1, but indexing from 0

            data_list = self.results_dict[DC_JSON_key][DM_index][LM_index]
            data_list = self.prepare_data_list_for_print(data_list)
        
        table_data = [self.headers, data_list]
        table_print = tabulate(table_data)
        print(table_print)
    
    def prepare_data_list_for_print(self, data_list):
        timestamp = int(data_list[0])
        if (timestamp == 0):
            time_string = 'Never'
        else:
            date_and_time = datetime.datetime.fromtimestamp(timestamp, tz=datetime.UTC)
            time_string = f'{date_and_time.strftime("%Y-%m-%d")} - {date_and_time.strftime("%H:%M:%S")}'

        new_data_list = []
        new_data_list.append(str(time_string))
        new_data_list.append(str(int(data_list[1])))
        new_data_list.append(str(lm(data_list[2])))
        new_data_list.append(str(dm(data_list[3])))
        new_data_list.append(f'{Float(data_list[4]):.2h}')
        new_data_list.append(f'{Float(data_list[5]):.1h}')
        new_data_list.append(f'{Float(data_list[6]):.1h}')
        new_data_list.append(f'{int(data_list[7])}')
        new_data_list.append(f'{Float(data_list[8]):.3h}')        

        return new_data_list

    def save_to_html(self):
        html_string = ''
        Include_index = [0, 2, 4, 5, 6, 7, 8] # What data to include in the table

        for item in sorted(self.results_dict.items(), reverse=True):
            data_array = item[1]
            html_string += f'# DAC Configuration: {item[0]} \n'

            for DM_index, DM in enumerate(data_array): # Static or Simulation (Spectre or Ngspice)

                html_string += f'### Model: {str(dm(DM_index + 1))} \n'

                table_data = [[self.headers[i] for i in Include_index]]
                for LM in DM:
                    data_list = self.prepare_data_list_for_print(LM)
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
    JR = JSON_results()