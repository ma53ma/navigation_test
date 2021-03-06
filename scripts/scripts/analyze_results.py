#!/usr/bin/env python

import time
from nav_scripts.result_analyzer import ResultAnalyzer


if __name__ == "__main__":
    start_time = time.time()
    analyzer = ResultAnalyzer()
    filenames = ['~/simulation_data/results_2021-11-11 21:53:07.554505']
    analyzer.readFiles(filenames=filenames, whitelist={})
    analyzer.generateGenericTable(independent=['controller'], dependent='result')
