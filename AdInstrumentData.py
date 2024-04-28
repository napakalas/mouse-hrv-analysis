import numpy as np
import mat73
from datetime import datetime, timedelta
import logging as log
from wfdb import processing
import numpy as np
from operator import itemgetter
import json
import pandas as pd
import os

class AdInstrumentData:
    
    def __init__(self, mat_files):
        # matfiles: can load several mat files
        
        data_dict = mat73.loadmat(mat_files[0])
        # extract start time
        data_starts = data_dict['record_meta']['data_start']
        self.__start_time = self.__days2datetime(data_starts[0]) if isinstance(data_starts, list) else self.__days2datetime(data_starts)
        
        # extract data frequency
        tick_dts = data_dict['record_meta']['tick_dt']
        self.__fs = 1/tick_dts[0] if isinstance(tick_dts, list) else 1/tick_dts

        self.__template = 'data__chan_'
        all_ticks = {}
        
        def extract_ticks(data, all_ticks):
            # combine all ticks
            
            tmp_keys = []
            for key in data.keys():
                if self.__template in key:
                    tmp_keys += [[int(k) for k in key.split('_') if k.isnumeric()] + [key]]
            tmp_keys = sorted(tmp_keys, key=itemgetter(0, 1))

            for l_key in tmp_keys:
                key = self.__template + str(l_key[0])
                if type(data[l_key[-1]]) == np.ndarray:
                    all_ticks[key] = np.concatenate((all_ticks[key], data[l_key[-1]])) if key in all_ticks else data[l_key[-1]]

        extract_ticks(data_dict, all_ticks)
        for file in mat_files[1:]:
            extract_ticks(mat73.loadmat(file), all_ticks)
        self.__all_ticks = all_ticks
        self.__tick_len = len(list(all_ticks.values())[0])
        
        self.__stop_time = self.__start_time + timedelta(seconds=len(self.all_ticks[self.__template+'1'])/self.__fs)
        
        self.__date_format = "%Y-%m-%d %H:%M:%S.%f"
        
        log.info(f'Labchart data from {str(self.__start_time)} to {str(self.__stop_time)}')
        
    def __days2datetime(self, days):
        # days, ex: array(739071.96513903)
        reference_date = datetime(1, 1, 1)
        return reference_date + timedelta(days=float(days)-367)
    
    def __tick_pos2datetime(self, tick_pos:int):
        """
        args:
            tick_pos: position of tick
        return:
            date_time in datetime format
        """
        seconds = tick_pos / self.__fs
        return self.__start_time + timedelta(seconds=seconds)
    
    def __datetime2tick_pos(self, date_time):
        """
        args:
            date_time: date and time in sting or datetime data type
        return:
            position of ticks
        """
        if isinstance(date_time, str):
            date_time = datetime.strptime(date_time, self.__date_format)
        if isinstance(date_time, datetime):
            return round((date_time - self.__start_time).total_seconds() * self.__fs)
        return None
    
    @property
    def start_time(self):
        return self.__start_time
    
    @property
    def finish_time(self):
        return self.__stop_time
    
    @property
    def fs(self):
        return self.__fs
    
    @property
    def all_ticks(self):
        return self.__all_ticks
    
    @property
    def tick_len(self):
        return self.__tick_len
    
    @property
    def channels(self):
        return [c[len(self.__template):] for c in self.__all_ticks.keys()]
        
    def __get_ecg_range(self, channel, from_time=None, to_time=None, duration=3600):
        """
        from_time : time to start
        to_time: time to finish
        duration : duration in seconds, default = 1 hours = 60 * 60 seconds
        channel : start from 1
        return: validity(%), ticks, from_time, to_time
        """
        if from_time is None:
            from_time = self.__start_time
        start_tick = self.__datetime2tick_pos(from_time)
        if to_time is None and duration:
            finish_tick = start_tick + duration
            to_time = self.__tick_pos2datetime(finish_tick)
        elif to_time is not None:
            finish_tick = self.__datetime2tick_pos(to_time)
        if finish_tick > self.__tick_len:
            finish_tick = self.__tick_len
            to_time = self.__tick_pos2datetime(finish_tick)
        
        
        if start_tick >= self.__tick_len or start_tick >= finish_tick:
            return None
        
        tick_channel = self.__template + str(channel)
        validity_channel = self.__template + str(channel + 1 if (channel + 1) % 3 == 0 else channel + 2)
        
        ticks = self.__all_ticks[tick_channel][start_tick:finish_tick]
        validity = self.__all_ticks[validity_channel][start_tick:finish_tick]            
        
        
        # calculate validity level
        val_level = np.mean(validity)
        # remove ticks with invalid recording
        condition = validity < 0.5
        filtered_ticks = ticks[~condition]

        return val_level, filtered_ticks, from_time, to_time
        
    def get_data(self, channel, from_time=None, to_time=None, duration=3600):
        """
        from_time : time to start
        to_time: time to finish
        duration : duration in seconds, default = 1 hours = 60 * 60 seconds
        channel : start from 1
        return: validity(%), ticks, from_time, to_time
        """
        val_level, signal, from_time, to_time = self.__get_ecg_range(channel, from_time, to_time, duration)
        
        # GET PEAKS
        # rqrs config file for mouse
        # using PhysioZoo setup
        hr = 608
        qs = 0.00718
        qt = 0.03
        QRSa = 1090
        QRSamin = 370
        rr_min = 0.05
        rr_max = 0.24
        window_size_sec = 0.005744 # 0.8*QS

        # adjusting peaks location
        peaks_window = 17
        th = 0.5
        # Use the maximum possible bpm as the search radius
        # Ostergaard G, Hansen HN, Ottesen JL. Physiological, Hematological, and Clinical Chemistry Parameters, 
        # Including Conversion Factors. In: Hau J, Schapiro SJ, editors. 
        # Handbook of laboratory animal science, Volume I: Essential Principles and Practices. 3rd ed. Vol. 1. 
        # Boca Raton, FL: CRC Press; 2010. pp. 667–707.
        min_bpm = 310
        max_bpm = 840
        search_radius = int(self.__fs * 60 / max_bpm)

        # Use the GQRS algorithm to detect QRS locations in the first channel
        try:
            qrs_inds = processing.qrs.gqrs_detect(sig=signal, 
                                                  fs=self.__fs, 
                                                  RRmin=rr_min, 
                                                  RRmax=rr_max,
                                                  hr=hr, 
            #                                       QS=qs, 
                                                  QT=qt, 
                                                  QRSa=QRSa, 
            #                                       QRSamin=QRSamin
                                                 )
        except Exception as e:
            log.warning(f'Cannot identify peaks,\n {e}')
            qrs_inds = []
        
        # Correct the peaks shifting them to local maxima
        if len(qrs_inds) > 0:
            try:
                peaks = processing.peaks.correct_peaks(
                    signal,
                    peak_inds=qrs_inds,
                    search_radius=search_radius, 
                    smooth_window_size=peaks_window
                )
            except Exception as e:
                log.warning(f'Cannot correct peaks,\n {e}')
                peaks = qrs_inds
        else:
            return {"val_level": val_level, "nni": [], "from_time": from_time, "to_time":to_time, "activity": activity}
        
    
        # GET NNI, THEN FILTER NNI and PEAKS
        nni = np.diff(peaks)
        condition = (nni >= rr_min*self.__fs) & (nni <= rr_max*self.__fs)
        nni = nni[condition]
        
        # filter nni within mean +- std2x
        nni_mean = np.mean(nni)
        std2x = 2 * np.std(nni)
        condition = (nni >= nni_mean-std2x) & (nni <= nni_mean+std2x)
        nni = nni[condition]
        
        # get activity in lower data channel
        tick_channel = self.__template + str(channel-1)
        start_tick = self.__datetime2tick_pos(from_time)
        end_tick = self.__datetime2tick_pos(to_time)
        activity = self.__all_ticks[tick_channel][start_tick:end_tick]

        return {"val_level": val_level, "nni": nni, "from_time": from_time, "to_time":to_time, "activity": activity}
    
def extract_data(data_source, duration, sliding_window, minimum_data_validity, minimum_nni_count):
    
    # load all calculated and source
    if os.path.exists(data_source['log_file']):
        with open(data_source['log_file'], 'r') as f:
            loaded_files = json.load(f)
    else:
        loaded_files = []
    if os.path.exists(data_source['calculated_results']):
        df_calculated = pd.read_csv(data_source['calculated_results'])
        calculated = {}
        rows = ['hr_means', 'activity', 'val_level', 'from_time', 'to_time', 'channel']
        for row_name, row_data in df_calculated.to_dict().items():
            if row_name in rows:
                calculated[row_name] = list(row_data.values())
    else:
        calculated = {'hr_means':[], 'activity':[], 'val_level':[], 'from_time':[], 'to_time':[], 'channel':[]}
        
    def update_log(file, loaded_files, log_file):
        loaded_files += [file]
        with open(log_file, 'w') as f:
            json.dump(loaded_files, f)

    files = [os.path.join(data_source['directory'], file) for file in os.listdir(data_source['directory']) if file.endswith(".mat")]
    for file in files:
        if file in loaded_files:
            continue

        raw_data = AdInstrumentData([file])

        start_time = raw_data.start_time
        finish_time = start_time + timedelta(seconds=duration)

        timer = 0

        while start_time < raw_data.finish_time:
            for channel in data_source['channels']:
                if finish_time > raw_data.finish_time:
                    finish_time = raw_data.finish_time
                ecg = raw_data.get_data(channel, from_time=start_time, to_time=finish_time)
                calculated['nni'] += [ecg['nni']]
                calculated['activity'] += [np.mean(ecg['activity'])]
                calculated['val_level'] += [ecg['val_level']]
                calculated['from_time'] += [ecg['from_time']]
                calculated['to_time'] += [ecg['to_time']]
                calculated['channel'] += [channel]
            start_time = start_time + timedelta(seconds=sliding_window)
            finish_time = start_time + timedelta(seconds=duration)

            if timer%6==0: print('.', end='')
            timer += 1

        update_log(file, loaded_files)
        df_calculated = pd.DataFrame(calculated)
        df_calculated.to_csv(data_source['calculated_results'])

def main():
    # setup
    log.basicConfig(level=log.INFO)

    # duration = 3600 #1 hour
    # sliding_window = 1800 # 30 minutes
    duration = 300 # 5 minutes
    sliding_window = 0 # 0 minutes

    minimum_data_validity = 0.3
    minimum_nni_count = 10

    data_sources = [
        {
            'directory': '/Volumes/Aswaty Nur/Pilot Project 2 (2023)/Converted_telemetry_20230315_mat',
            'log_file': 'log_project2.txt',
            'calculated_results': 'actogram_2_5min.csv',
            'channels': [2]
        },
        {
            'directory': '/Volumes/Aswaty Nur/Pilot Project 1 (2022)/Converted_telemetry_20221026_mat',
            'log_file': 'log_project1.txt',
            'calculated_results': 'actogram_1_5min.csv',
            'channels': [5,8]
        }
    ]

    for data_source in data_sources:
        extract_data(data_source, duration, sliding_window, minimum_data_validity, minimum_nni_count)

if __name__ == '__main__':
    main()