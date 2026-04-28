import threading
import subprocess
import os
import signal
import time
import psutil
import numpy as np
import shlex

from lib import utils
from lib.container import Container
from lib import constants

class Workload:
    ''' This class is not meant to be used by itself. It's only purpose
        is to provide definitions that are common to all of its children.
    '''
    # These variables are defined in child classes
    # that inherit from this class. Their definition here is
    # just done for clarity.
    wname = None
    ideal_mem = None
    min_ratio = None
    cpu_req = None
    coeff = None

    def __init__(self, idd, pinned_cpus, mem_ratio=1):

        self.idd = idd  # a unique uint id for this workload

        # process handling
        self.thread = None
        self.popen = None
        self.stdout = None
        self.stderr = None

        # Container creation
        self.mem_ratio = mem_ratio
        self.container = Container(self.get_name(), self.ideal_mem, self.mem_ratio)
        self.container.create()
        
        # Pin CPUs
        self.pinned_cpus = pinned_cpus

        # Get shell command
        procs_path = self.container.get_procs_path()
        self.cmdline = self.get_cmdline(procs_path, pinned_cpus)

        # task timings
        self.ts_start = 0
        self.ts_finish = 0

        # Getting gradient coeffs ready
        self.percent = 0
        self.ratio = 1
        self.gd_coeff = None
        self.mem_gd_coeff = None
        if self.coeff is not None:
            self.get_gradient()

    def __exec(self):
        " execute in self.thread "
        print(self.cmdline)

        self.ts_start = time.time()
        self.popen = subprocess.Popen(self.cmdline, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, shell=True)
        self.stdout, self.stderr = self.popen.communicate()  # blocks process exit
        assert(self.popen.returncode == 0)
        self.ts_finish = time.time()

        self.container.delete()

    def start(self):
        self.thread = threading.Thread(target=self.__exec)
        self.thread.start()

        while not self.is_alive():
            pass

    def modify_ratio(self, new_ratio):
        self.container.set_new_size(new_ratio)

    def get_name(self):
        return self.wname + str(self.idd)

    def get_retcode(self):
        return self.popen.returncode

    def is_alive(self):
        return self.thread.is_alive() and self.popen

    def get_process_duration(self):
        return self.ts_finish - self.ts_start

    def get_usr_bin_time(self):
        ''' Parse the output of /usr/bin/time from stderr'''
        parser = utils.BinTimeParser()
        return parser.parse(self.stderr.decode('utf-8'))

    def kill(self):
        pg_id = os.getpgid(self.popen.pid)
        os.killpg(pg_id, signal.SIGKILL)
        self.thread.join()

    def set_min_ratio(self, new_min_ratio):
        self.min_ratio = new_min_ratio
        self.min_mem = self.min_ratio * self.ideal_mem

    def update(self, el_time, new_ratio, new_idd=None): # ratio = 0 is no remote memory mode
        assert el_time >= 0

        if (new_idd is not None) and self.idd == new_idd:
            assert self.percent == 0
        else:
            self.update_percent(el_time)
        self.ratio = new_ratio
        self.modify_ratio(new_ratio)

    def update_percent(self, el_time):
        self.percent = self.percent + el_time/self.profile(self.ratio)

    def require_coeffs(self):
        if self.coeff is None:
            raise RuntimeError(
                "{} requires coeff profiling when using the optimal shrinking policy".format(
                    self.wname
                )
            )
        if self.gd_coeff is None or self.mem_gd_coeff is None:
            self.get_gradient()

    def profile(self,ratio): 
        self.require_coeffs()
        return self.compute_ratio_from_coeff(self.coeff, ratio)*1000 # from second to millisecond

    def get_gradient(self):
        tmp_coeff = self.coeff + [0]
        self.gd_coeff = np.polyder(self.coeff)
        self.mem_gd_coeff = np.polyder(tmp_coeff)

    def gradient(self, ratio):
        self.require_coeffs()
        return self.compute_ratio_from_coeff(self.gd_coeff, ratio)

    def mem_gradient(self,ratio):
        self.require_coeffs()
        return self.compute_ratio_from_coeff(self.mem_gd_coeff, ratio)

    def compute_ratio_from_coeff(self, coeffs, ratio):
        p = 0
        order = len(coeffs)
        for i in range(order):
            p += coeffs[i] * ratio**(order-1-i)
        return p

    def get_pids(self):
        return self.container.get_pids()

class Quicksort(Workload):
    wname = "quicksort"
    ideal_mem = 10240
    min_ratio = 0.7
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "quicksort"
    cpu_req = 1
    x = [1,      0.9,    0.8,   0.7,    0.6]
    y = [248.75, 260.41, 268.4, 280.11, 300.78]
    coeff = [-895.83333333, 1814.16666667, -719.04166667, -586.04166667,  635.5]

    def get_cmdline(self, procs_path, pinned_cpus):
        prefix = "echo $$ > {} &&".format(procs_path)
        arg = '10240'
        shell_cmd = '/usr/bin/time -v' + ' ' + constants.WORK_DIR + '/quicksort/quicksort {}'.format(arg)
        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)
        full_command = ' '.join((prefix, 'exec', set_cpu, shell_cmd))
        return full_command

class LinearRegression(Workload):
    wname = "linearregression"
    ideal_mem = 6300
    min_ratio = 0.7
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "lr"
    cpu_req = 1
    x = [1,      0.9,    0.8,   0.7,    0.6]
    y = [248.75, 260.41, 268.4, 280.11, 300.78]
    coeff = [-895.83333333, 1814.16666667, -719.04166667, -586.04166667,  635.5]

    def get_cmdline(self, procs_path, pinned_cpus):
        prefix = "echo $$ > {} &&".format(procs_path)
        #arg = '10240'
        shell_cmd = '/usr/bin/time -v' + ' ' + constants.WORK_DIR + '/linearregression/lr'
        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)
        full_command = ' '.join((prefix, 'exec', set_cpu, shell_cmd))
        return full_command

class RandomAccess(Workload):
    wname = "random_acccess"
    ideal_mem = 66000
    min_ratio = 0.1
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "random_acccess"
    cpu_req = 1
    x = [1,      0.9,    0.8,   0.7,    0.6]
    y = [248.75, 260.41, 268.4, 280.11, 300.78]
    coeff = [-895.83333333, 1814.16666667, -719.04166667, -586.04166667,  635.5]

    def get_cmdline(self, procs_path, pinned_cpus):
        cpu_list = list(range(48))
        #cpu_list.extend(range(64, 96))
        #cpu_list.extend(range(32, 64))
        pinned_cpus = ','.join(str(cpu_list[i]) for i in range(0,31))
        print("pinned cpus are {}".format(pinned_cpus))

        prefix = "echo $$ > {} &&".format(procs_path)
        arg = '16384'
        # arg = '64000'
        shell_cmd = '/usr/bin/time -v' + ' ' + constants.WORK_DIR + '/quicksort/random_access {}'.format(arg)
        #pinned_cpus_string = ','.join(map(str, pinned_cpus))
        pinned_cpus_string = pinned_cpus
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)
        full_command = ' '.join((prefix, 'exec', set_cpu, shell_cmd))
        return full_command

class Matrix(Workload):
    wname = "matrix"
    ideal_mem = 2048
    min_ratio = 0.1
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "matrix"
    cpu_req = 1
    x = [1,      0.9,    0.8,   0.7,    0.6]
    y = [248.75, 260.41, 268.4, 280.11, 300.78]
    coeff = [-895.83333333, 1814.16666667, -719.04166667, -586.04166667,  635.5]

    def get_cmdline(self, procs_path, pinned_cpus):
        prefix = "echo $$ > {} &&".format(procs_path)
        arg = '16 2048'
        shell_cmd = '/usr/bin/time -v' + ' ' + constants.WORK_DIR + '/matrix/matrix {}'.format(arg)
        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)
        full_command = ' '.join((prefix, 'exec', set_cpu, shell_cmd))
        return full_command

class WordCount(Workload):
    wname = "wordcount"
    ideal_mem = 33000
    min_ratio = 0.1
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "wordcount"
    cpu_req = 1
    x = [1,      0.9,    0.8,   0.7,    0.6]
    y = [248.75, 260.41, 268.4, 280.11, 300.78]
    coeff = [-895.83333333, 1814.16666667, -719.04166667, -586.04166667,  635.5]

    def get_cmdline(self, procs_path, pinned_cpus):
        prefix = "echo $$ > {} &&".format(procs_path)
        arg = '32768'
        shell_cmd = '/usr/bin/time -v' + ' ' + constants.WORK_DIR + '/wordcount/wordcount {}'.format(arg)
        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)
        full_command = ' '.join((prefix, 'exec', set_cpu, shell_cmd))
        return full_command

class Xgboost(Workload):
    wname = "xgboost"
    ideal_mem = 16300
    min_ratio = 0.5
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "python"
    cpu_req = 2
    x = [1,      0.9,    0.8,    0.7,    0.6,    0.5,    0.4,    0.3,    0.2]
    y = [338.45, 341.90, 347.52, 349.21, 352.98, 356.92, 386.09, 405.70, 430.11]
    coeff = [ -876.04895105,  1878.74643875, -1148.56526807,    25.39511137, 457.79055556]
    def get_cmdline(self, procs_path, pinned_cpus):
        prefix = "echo $$ > {} &&".format(procs_path)
        #arg = '8192'
        shell_cmd = '/usr/bin/time -v' + ' ' + 'python ' + constants.WORK_DIR + '/xgboost/higgs.py'
        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)
        full_command = ' '.join((prefix, 'exec', set_cpu, shell_cmd))
        return full_command

class Kmeans(Workload):
    wname = "kmeans"
    ideal_mem = 4100
    min_ratio = 0.5
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "python"
    cpu_req = 1
    x = [1,      0.9,    0.8,    0.7,    0.6,    0.5,    0.4,    0.3,    0.2]
    y = [338.45, 341.90, 347.52, 349.21, 352.98, 356.92, 386.09, 405.70, 430.11]
    coeff = [ -876.04895105,  1878.74643875, -1148.56526807,    25.39511137, 457.79055556]
    def get_cmdline(self, procs_path, pinned_cpus):
        prefix = "echo $$ > {} &&".format(procs_path)
        #arg = '8192'
        shell_cmd = '/usr/bin/time -v' + ' ' + 'python ' + constants.WORK_DIR + '/kmeans/emails.py'
        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)
        full_command = ' '.join((prefix, 'exec', set_cpu, shell_cmd))
        return full_command

class Snappy(Workload):
    wname = "snappy"
    ideal_mem = 54000
    min_ratio = 0.8
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "compress"
    cpu_req = 1
    x = [1,      0.9,    0.8,    0.7,    0.6]
    y = [138.88, 143.15, 155.37, 211.18, 274.42]
    coeff = [-31583.33333335,  100776.66666673, -118088.66666675,   59796.08333338, -10765.87000001]
    def get_cmdline(self, procs_path, pinned_cpus):
        prefix = "echo $$ > {} &&".format(procs_path)
        arg = '/mydata/merged.xml'
        shell_cmd = '/usr/bin/time -v' + ' ' + 'python' + ' ' + constants.WORK_DIR + '/snappy/compress.py {}'.format(arg) 
        #shell_cmd = '/usr/bin/time -v' + ' ' + constants.WORK_DIR + '/snappy/compress {}'.format(arg) 
        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)
        full_command = ' '.join((prefix, 'exec', set_cpu, shell_cmd))
        return full_command

class Pagerank(Workload):
    wname = "pagerank"
    ideal_mem = 3580
    min_ratio = 1
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "pr"
    cpu_req = 8
    x = [1,      0.9,    0.8]
    y = [221.06, 736.29, 99900000.00]
    coeff = [-1617.416, 3789.953, -2993.734, 1225.477]
    def get_cmdline(self, procs_path, pinned_cpus):
        prefix = "echo $$ > {} &&".format(procs_path)
        #limit_mem = "echo $$ > {} ".format(procs_path)
        #arg = '-f' + ' ' + constants.WORK_DIR + '/pagerank/gapbs/k27output.sg'
        #arg = '-u 27'
        #pr_cmd = constants.WORK_DIR + '/pagerank/gapbs/pr {}'.format(arg)
        pr_cmd = constants.WORK_DIR + '/pagerank/pr'
        shell_cmd = '/usr/bin/time -v' + ' ' + pr_cmd
        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)
        full_command = ' '.join((prefix, 'exec', set_cpu, shell_cmd))
        return full_command


class Memcached(Workload):
    wname = "memcached"
    ideal_mem = 16900
    min_ratio = 0.6
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "memcached"
    port_number = 11500 
    cpu_req = 2
    x = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    y = [1273.74, 1280.33, 1290.85, 1340.63, 1479.72, 1694.631, 1949.36]
    coeff = [-14813.52272727, 38076.50252525, -31906.18749999, 8189.04796176, 1725.16288095]

    def __init__(self, idd, pinned_cpus, mem_ratio=1):
        super().__init__(idd, pinned_cpus, mem_ratio)
        self.port_number = Memcached.port_number
        self.memcached_bench_pids = set()
        Memcached.port_number += 1

    def get_cmdline(self, procs_path, pinned_cpus):
        prefix = 'echo $$ > {} &&'
        memcached_serv = "/usr/bin/time -v memcached -p {} -m {} -c 1024 -t 8".format(
            self.port_number, self.ideal_mem*2)
        
        cpu_list = list(range(48))
        #cpu_list.extend(range(64, 96))
        """
        sever_taskset = ','.join((str(cpu_list[0]), str(cpu_list[1]), str(cpu_list[2]), str(cpu_list[3]), str(cpu_list[4]), str(cpu_list[5]), str(cpu_list[6]), str(cpu_list[7]),
                                  str(cpu_list[8]), str(cpu_list[9]), str(cpu_list[10]), str(cpu_list[11]), str(cpu_list[12]), str(cpu_list[13]), str(cpu_list[14]), str(cpu_list[15]),
                                  str(cpu_list[16]), str(cpu_list[17]), str(cpu_list[18]), str(cpu_list[19]), str(cpu_list[20]), str(cpu_list[21]), str(cpu_list[22]), str(cpu_list[23]),
                                 str(cpu_list[24]), str(cpu_list[25]), str(cpu_list[26]), str(cpu_list[27]), str(cpu_list[28]), str(cpu_list[29]), str(cpu_list[30]), str(cpu_list[31]),
                                 ))
        """
        sever_taskset = ','.join(str(cpu_list[i]) for i in range(0, 8))
        #sever_taskset += ','
        #sever_taskset += ','.join(str(cpu_list[i]) for i in range(44, 64, 2))
        taskset_serv = 'taskset -c {}'.format(sever_taskset)
        memcached_serv = ' '.join((prefix, 'exec', taskset_serv, memcached_serv))
        memcached_serv = memcached_serv.format(procs_path)

        """
        client_taskset = ','.join(( 
                    str(cpu_list[32]), str(cpu_list[33]), str(cpu_list[34]), str(cpu_list[35]), str(cpu_list[36]), str(cpu_list[37]), str(cpu_list[38]), str(cpu_list[39]),
                    str(cpu_list[40]), str(cpu_list[41]), str(cpu_list[42]), str(cpu_list[43]), str(cpu_list[44]), str(cpu_list[45]), str(cpu_list[46]), str(cpu_list[47]),
                    str(cpu_list[48]), str(cpu_list[49]), str(cpu_list[50]), str(cpu_list[51]), str(cpu_list[52]), str(cpu_list[53]), str(cpu_list[54]), str(cpu_list[55]),
                    str(cpu_list[56]), str(cpu_list[57]), str(cpu_list[58]), str(cpu_list[59]), str(cpu_list[60]), str(cpu_list[61]), str(cpu_list[62]), str(cpu_list[63]),
                    ))
        """
        client_taskset = ','.join(str(cpu_list[i]) for i in range(8, 16))
        #client_taskset += ','
        #client_taskset += ','.join(str(cpu_list[i]) for i in range(45, 64, 2))
        taskset_client = 'taskset -c {}'.format(client_taskset)
        #ycsb_load = taskset_ycsb + ' ' + constants.WORK_DIR + "/memcached/ycsb-0.17.0/bin/ycsb.sh load memcached -s -P " + \
        #            constants.WORK_DIR + "/memcached/ycsb-0.17.0/workloads/workloadb -p \"memcached.hosts=localhost:{}\" -p \"operationcount=30000000\" -p \"recordcount=30000000\" -p \"fieldlength=256\" -p \"fieldcount=2\"".format(self.port_number)
        #ycsb_run = taskset_ycsb + ' ' + constants.WORK_DIR + "/memcached/ycsb-0.17.0/bin/ycsb.sh run memcached -s -P " + \
        #           constants.WORK_DIR + "/memcached/ycsb-0.17.0/workloads/workloadb -p \"memcached.hosts=localhost:{}\" -p \"operationcount=30000000\" -p \"requestdistribution=uniform\"".format(self.port_number)
        
        client_cmd = 'exec' + ' ' + taskset_client + ' ' + constants.WORK_DIR + "/memcached/mc {}".format(self.port_number)
        return (memcached_serv, client_cmd)

    def start(self):
        self.thread = threading.Thread(target=self.__exec)
        self.thread.start()
        while not self.is_alive():
            pass

    def __exec(self):
        memcached, client_cmd = self.cmdline

        print(self.cmdline)
        self.ts_start = time.time()

        self.popen = subprocess.Popen(memcached, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, shell=True,
                                      preexec_fn=os.setsid)

        time.sleep(3)  # Wait for memcached to boot


        client_proc = subprocess.Popen(client_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        stdout, stderr = client_proc.communicate()
        print(stdout.decode('utf-8'))
        print(stderr.decode('utf-8'))

        #time.sleep(5)

        #ycsb_run_proc = subprocess.Popen(shlex.split(ycsb_run), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        #stdout, stderr = ycsb_run_proc.communicate()
        #print(stdout.decode('utf-8'))
        #print(stderr.decode('utf-8'))

        os.killpg(os.getpgid(self.popen.pid), signal.SIGINT)

        self.stdout, self.stderr = self.popen.communicate()
        self.ts_finish = time.time()
        print(self.stdout.decode('utf-8'))
        print(self.stderr.decode('utf-8'))

        self.container.delete()

    def get_pids(self):
        pids = list(self.container.get_pids())
        pids.extend(self.memcached_bench_pids)
        return pids


class Redis(Workload):
    wname = "redis"
    ideal_mem = 31800
    min_ratio = 0.6
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "redis-server"
    port_number = 63791
    cpu_req = 2
    x = [1,       0.9,     0.8,     0.7,     0.6,     0.5,      0.4]
    y = [1273.74, 1280.33, 1290.85, 1340.63, 1479.72, 1694.631, 1949.36 ]
    coeff = [-14813.52272727,  38076.50252525, -31906.18749999,   8189.04796176, 1725.16288095]
    def __init__(self, idd, pinned_cpus, mem_ratio=1):
        super().__init__(idd, pinned_cpus, mem_ratio)
        self.port_number = Redis.port_number
        self.redis_bench_pids = set()
        Redis.port_number += 1

    def get_cmdline(self, procs_path, pinned_cpus):
        prefix = 'echo $$ > {} &&'
        redis_serv = "/usr/bin/time -v redis-server --port {} --maxmemory {}mb --maxmemory-policy allkeys-lru".format(self.port_number, 
                                                    self.ideal_mem)
        cpu_list = list(pinned_cpus)
        #pinned_cpus_string1 = ','.join(map(str, pinned_cpus[0:3]))
        #pinned_cpus_string2 = ','.join(map(str, pinned_cpus[4:7]))

        taskset_serv = 'taskset -c {}'.format(cpu_list[0])
        redis_serv = ' '.join((prefix, 'exec', taskset_serv, redis_serv))
        redis_serv = redis_serv.format(procs_path)

        taskset_ycsb = 'taskset -c {}'.format(cpu_list[1])
        # YCSB load data
        ycsb_load = taskset_ycsb + ' ' + constants.WORK_DIR + "/redis/ycsb-0.17.0/bin/ycsb.sh load redis -s -P " + constants.WORK_DIR + "/redis/ycsb-0.17.0/workloads/workloadb -p \"redis.host=localhost\" -p \"redis.port={}\" -p \"operationcount=30000000\" -p \"recordcount=30000000\" -p \"fieldlength=256\" -p \"fieldcount=2\"".format(self.port_number)
        # YCSB run workload with Zipf distribution
        ycsb_run = taskset_ycsb + ' ' + constants.WORK_DIR + "/redis/ycsb-0.17.0/bin/ycsb.sh run redis -s -P " + constants.WORK_DIR + "/redis/ycsb-0.17.0/workloads/workloadb -p \"redis.host=localhost\" -p \"redis.port={}\" -p \"operationcount=30000000\" -p \"requestdistribution=uniform\"".format(self.port_number)
    
        sleep = 'sleep 5'
        ycsb_cmd = ' && '.join((ycsb_load, sleep, ycsb_run))
        return (redis_serv, ycsb_cmd)

    def start(self):
        self.thread = threading.Thread(target=self.__exec)
        self.thread.start()

        while not self.is_alive():
            pass

    def __exec(self):
        redis, ycsb_cmd = self.cmdline

        " execute in self.thread "
        print(self.cmdline)

        self.ts_start = time.time()

        self.popen = subprocess.Popen(redis, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, shell=True,
                                  preexec_fn=os.setsid)

        time.sleep(3) # Wait for redis to boot

        # Split ycsb_cmd into ycsb_load, sleep, ycsb_run
        ycsb_load, _, ycsb_run = ycsb_cmd.split(' && ')

        # Load data
        ycsb_load_proc = subprocess.Popen(shlex.split(ycsb_load), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        stdout, stderr = ycsb_load_proc.communicate()
        print(stdout.decode('utf-8'))
        print(stderr.decode('utf-8'))

        time.sleep(5) # Wait for data to load

        # Run workload
        ycsb_run_proc = subprocess.Popen(shlex.split(ycsb_run), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        stdout, stderr = ycsb_run_proc.communicate()
        print(stdout.decode('utf-8'))
        print(stderr.decode('utf-8'))

        os.killpg(os.getpgid(self.popen.pid), signal.SIGINT)

        self.stdout, self.stderr = self.popen.communicate()
        self.ts_finish = time.time()
        print(self.stdout.decode('utf-8'))
        print(stderr.decode('utf-8'))

        self.container.delete()

    def get_pids(self):
        pids = list(self.container.get_pids())
        pids.extend(self.redis_bench_pids)
        return pids



class Xsbench(Workload):
    wname = "xsbench"
    ideal_mem = 33300
    min_ratio = 1
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "XSBench"
    cpu_req = 8
    x = [1, 0.9, 0.8]
    y = [244.91, 478.54, 10000.0]
    coeff = [-1984.129, 4548.033, -3588.554, 1048.644, 252.997]

    def get_cmdline(self, procs_path, pinned_cpus):
        prefix = "echo $$ > {} &&".format(procs_path)
        arg = '-g 65000 -p 20000000'
        shell_cmd = '/usr/bin/time -v' + ' ' + constants.WORK_DIR + '/xsbench/XSBench/openmp-threading/XSBench {}'.format(arg)
        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)
        full_command = ' '.join((prefix, 'exec', set_cpu, shell_cmd))
        return full_command

    def start(self):
        self.thread = threading.Thread(target=self.__exec)
        self.thread.start()

        while not self.is_alive():
            pass

    def __exec(self):
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '8'
        self.ts_start = time.time()
        self.popen = subprocess.Popen(self.cmdline, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, shell=True, env=env)
        self.stdout, self.stderr = self.popen.communicate()  # blocks process exit
        assert(self.popen.returncode == 0)
        self.ts_finish = time.time()

        self.container.delete()


def get_workload_class(wname):
    return {'quicksort': Quicksort,
            'xgboost': Xgboost,
            'redis': Redis,
            'snappy': Snappy,
            'memcached': Memcached,
            'pagerank': Pagerank,
            'xsbench': Xsbench,
            'random_access': RandomAccess,
            'matrix': Matrix,
            'wordcount': WordCount,
            'kmeans': Kmeans,
            'linearregression': LinearRegression,
            }[wname]
