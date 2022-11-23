import subprocess
from dataclasses import dataclass
import psutil
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from log import setup_logger, logging


@dataclass
class monitor_data:
    data: list[float]
    start_time: float = 0
    stop_time: float = np.inf
    start_index: int = 0
    stop_index: int = np.inf

    def __str__(self):
        return f"start_time: {self.start_time}, stop_time: {self.stop_time}, data: {self.data}"


logger = setup_logger(__name__, filename='iotop.log', level=logging.DEBUG)


class Process:
    def __init__(self, pid_or_process):
        try:
            pid = int(pid_or_process)
            logger.info("Attaching to process {0}".format(pid))
            self.process = None
        except ValueError:
            command = pid_or_process
            logger.info("Starting up command '{0}' and attaching to process"
                        .format(command))
            self.process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            pid = self.process.pid

        self.pid = pid
        self._current_time = time.time()
        self.process = psutil.Process(pid)
        self._all_process = {self.process}
        self.children = set()  # type: set[psutil.Process]
        self.timestamp = []  # type: list[float]
        self.cpu_percent = {pid: monitor_data([])}  # type: dict[int, monitor_data]
        self.memory_real = {pid: monitor_data([])}  # type: dict[int, monitor_data]
        self.memory_virtual = {pid: monitor_data([])}  # type: dict[int, monitor_data]
        self.disk_read = {pid: monitor_data([])}  # type: dict[int, monitor_data]
        self.disk_write = {pid: monitor_data([])}  # type: dict[int, monitor_data]
        self.disk_read_speed = {pid: monitor_data([])}  # type: dict[int, monitor_data]
        self.disk_write_speed = {pid: monitor_data([])}  # type: dict[int, monitor_data]

        self.__index = 0

    def update_children(self):
        try:
            children_of_pr = self.process.children(recursive=True)
        except Exception:
            return

        if diff := set(children_of_pr) - self.children:
            self.cpu_percent.update({child.pid: monitor_data([], start_time=self._current_time, start_index=self.__index) for child in diff})
            self.memory_real.update({child.pid: monitor_data([], start_time=self._current_time, start_index=self.__index) for child in diff})
            self.memory_virtual.update({child.pid: monitor_data([], start_time=self._current_time, start_index=self.__index) for child in diff})
            self.disk_read.update({child.pid: monitor_data([], start_time=self._current_time, start_index=self.__index) for child in diff})
            self.disk_write.update({child.pid: monitor_data([], start_time=self._current_time, start_index=self.__index) for child in diff})
            self.disk_read_speed.update({child.pid: monitor_data([], start_time=self._current_time, start_index=self.__index) for child in diff})
            self.disk_write_speed.update({child.pid: monitor_data([], start_time=self._current_time, start_index=self.__index) for child in diff})

            self.children.update(diff)
            self._all_process.update(diff)

    def get_cpu(self, include_children: bool = False):
        result = np.zeros(len(self._all_process) if include_children else 1)
        for i, child in enumerate(self._all_process if include_children else [self.process]):
            result[i] = child.cpu_percent()

        return result

    def get_memory(self, include_children: bool = False):
        result_real = np.zeros(len(self._all_process) if include_children else 1)
        result_virtual = np.zeros(len(self._all_process) if include_children else 1)
        for i, child in enumerate(self._all_process if include_children else [self.process]):
            mem = child.memory_info()
            result_real[i] = mem.rss
            result_virtual[i] = mem.vms

        return result_real, result_virtual

    def get_disk(self, include_children: bool = False):
        result_read = np.zeros(len(self._all_process) if include_children else 1)
        result_write = np.zeros(len(self._all_process) if include_children else 1)
        for i, child in enumerate(self._all_process if include_children else [self.process]):
            io = child.io_counters()
            result_read[i] = io.read_bytes
            result_write[i] = io.write_bytes

        return result_read, result_write

    def kill(self):
        if self.process:
            self.process.kill()

    def update_timestamp(self, which: str, timestamp: float, include_children: bool = False, include_main: bool = True, pid: psutil.Process = None):
        if which == 'start':
            _attr1 = 'start_time'
            _attr2 = 'start_index'
        elif which == 'stop':
            _attr1 = 'stop_time'
            _attr2 = 'stop_index'
        else:
            raise ValueError("which must be 'start' or 'stop'")

        to_iter = set()  # type: set[psutil.Process]

        if isinstance(pid, psutil.Process):
            to_iter.add(pid)
        elif pid is None:
            if include_children:
                to_iter.update(self.children)

            if include_main:
                to_iter.add(self.process)
        else:
            raise ValueError("pid must be a psutil.Process or None")

        for child in to_iter:
            setattr(self.cpu_percent[child.pid], _attr1, timestamp)
            setattr(self.memory_real[child.pid], _attr1, timestamp)
            setattr(self.memory_virtual[child.pid], _attr1, timestamp)
            setattr(self.disk_read[child.pid], _attr1, timestamp)
            setattr(self.disk_write[child.pid], _attr1, timestamp)
            setattr(self.disk_read_speed[child.pid], _attr1, timestamp)
            setattr(self.disk_write_speed[child.pid], _attr1, timestamp)
            setattr(self.cpu_percent[child.pid], _attr2, self.__index)
            setattr(self.memory_real[child.pid], _attr2, self.__index)
            setattr(self.memory_virtual[child.pid], _attr2, self.__index)
            setattr(self.disk_read[child.pid], _attr2, self.__index)
            setattr(self.disk_write[child.pid], _attr2, self.__index)
            setattr(self.disk_read_speed[child.pid], _attr2, self.__index)
            setattr(self.disk_write_speed[child.pid], _attr2, self.__index)

    def monitor(self, duration: float = None, include_children: bool = False, interval: float = None):
        start_time = time.time()

        while True:
            # Find current time
            self._current_time = time.time()

            if duration is not None and np.floor(self._current_time - start_time) > duration:
                break

            if self.cpu_percent[self.pid].start_time == 0:
                self.update_timestamp('start', self._current_time, include_children)

            if self.process.status() in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                logger.info(f"Process finished ({self._current_time - start_time:.2f} seconds)")
                break

            if include_children:
                for child in self.children:
                    try:
                        if child.status() in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                            self.update_timestamp('stop', self._current_time, pid=child)
                    except psutil.NoSuchProcess:
                        logger.warning(f"Process {child.pid} not found")
                        self.update_timestamp('stop', self._current_time, pid=child)

                self.update_children()

            self.timestamp.append(self._current_time)

            try:
                cpu_percent = self.get_cpu(include_children)
                mem_real, mem_virtual = self.get_memory(include_children)
                disk_read, disk_write = self.get_disk(include_children)
            except psutil.NoSuchProcess:
                logger.warning(f"Process {self.pid} does not exist anymore")
                self.timestamp.pop()
                break
            except psutil.AccessDenied:
                logger.warning(f"Access denied to process {self.pid}")
                self.timestamp.pop()
                break
            else:
                for i, child in enumerate(self._all_process if include_children else [self.process]):
                    self.cpu_percent[child.pid].data.append(cpu_percent[i])
                    self.memory_real[child.pid].data.append(mem_real[i])
                    self.memory_virtual[child.pid].data.append(mem_virtual[i])
                    self.disk_read[child.pid].data.append(disk_read[i])
                    self.disk_write[child.pid].data.append(disk_write[i])

            self.__index += 1
            if interval is not None:
                time.sleep(interval)

    def get_io_speed(self, include_children: bool = False):
        for pid, disk_read in self.disk_read.items():
            if not include_children and pid != self.pid:
                continue

            _length = len(disk_read.data)
            final_data = np.zeros_like(disk_read.data)
            diff_data = np.diff(disk_read.data, prepend=0)

            stop_index = disk_read.stop_index if disk_read.stop_index != np.inf else _length

            pre_index = disk_read.start_index
            non_zero = np.nonzero(diff_data)[0]
            for index in non_zero:
                final_data[pre_index:index] = np.repeat(diff_data[index] / (self.timestamp[index] - self.timestamp[pre_index]), index - pre_index)
                pre_index = index

            final_data[pre_index:stop_index] = np.repeat(final_data[pre_index - 1], stop_index - pre_index)
            self.disk_read_speed[pid].data = final_data

        for pid, disk_write in self.disk_write.items():
            _length = len(disk_write.data)
            final_data = np.zeros_like(disk_write.data)
            diff_data = np.diff(disk_write.data, prepend=0)

            stop_index = disk_write.stop_index if disk_write.stop_index != np.inf else _length

            pre_index = disk_write.start_index
            non_zero = np.nonzero(diff_data)[0]
            for index in non_zero:
                final_data[pre_index:index] = np.repeat(diff_data[index] / (self.timestamp[index] - self.timestamp[pre_index]), index - pre_index)
                pre_index = index

            final_data[pre_index:stop_index] = np.repeat(final_data[pre_index - 1], stop_index - pre_index)
            self.disk_write_speed[pid].data = final_data

    def calc_data(self, data: dict[int, monitor_data], include_children: bool = False, average: bool = False):
        final_data = np.zeros_like(self.timestamp)
        count = np.zeros_like(self.timestamp)

        for pid, _data in data.items():
            if include_children and pid != self.pid:
                if _data.start_index != 0 or _data.stop_index != np.inf:
                    stop_index = _data.stop_index if _data.stop_index != np.inf else len(self.timestamp)
                    final_data[_data.start_index:stop_index] += _data.data
                    count[_data.start_index:stop_index] += 1
                else:
                    final_data += _data.data
                    count += 1
            else:
                if pid == self.pid:
                    final_data += _data.data
                    count += 1

        return final_data / count if average else final_data

    def save(self, filename: str, include_children: bool = False):
        with open(filename, 'w', encoding="utf-8") as f:
            f.write('timestamp,cpu_percent,memory_real,memory_virtual,disk_read_speed,disk_write_speed\n')
            for i, current_time in enumerate(self.timestamp):
                f.write("{:12.3f} {:12.1f} {:12.0f} {:12.0f} {:12.3f} {:12.3f}\n".format(current_time - self.timestamp[0], self.cpu_percent[self.pid].data[i], self.memory_real[self.pid].data[i], self.memory_virtual[self.pid].data[i], self.disk_read_speed[self.pid].data[i], self.disk_write_speed[self.pid].data[i]))

            if include_children:
                for child in self.children:
                    f.write("\n\npid: {}\n".format(child.pid))
                    timestamp = np.array(self.timestamp)
                    index = (timestamp >= self.cpu_percent[child.pid].start_time) & (timestamp <= self.cpu_percent[child.pid].stop_time)
                    for i, current_time in enumerate(timestamp[index]):
                        f.write("{:12.3f} {:12.1f} {:12.0f} {:12.0f} {:12.3f} {:12.3f}\n".format(current_time - self.timestamp[0], self.cpu_percent[child.pid].data[i], self.memory_real[child.pid].data[i], self.memory_virtual[child.pid].data[i], self.disk_read_speed[child.pid].data[i], self.disk_write_speed[child.pid].data[i]))

    def plot(self):
        import matplotlib.pyplot as plt
        _time = np.array(self.timestamp) - self.timestamp[0]

        fig, ax1 = plt.subplots(1, 2)
        ax2 = ax1[0].twinx()
        ax1[0].set_xlabel('Time (s)')
        ax1[0].set_ylabel('CPU (%)')
        ax2.set_ylabel('Memory (MB)')
        l1 = ax1[0].plot(_time, self.calc_data(self.cpu_percent, include_children=True), label='CPU', linestyle='None', marker='o')
        l2 = ax2.plot(_time, self.calc_data(self.memory_real, include_children=True) / 1024**2, label='Memory Real', linestyle='None', marker='x')
        l3 = ax2.plot(_time, self.calc_data(self.memory_virtual, include_children=True) / 1024**2, label='Memory Virtual', linestyle='None', marker='x')

        ax1[0].yaxis.grid(which='major', linestyle='-', color='gray')
        ax2.yaxis.grid(which='major', linestyle='--', color='silver')
        ax1[0].xaxis.grid(which='major', linestyle='-', color='gray')
        ax1[0].xaxis.grid(which='minor', linestyle='--', linewidth=0.5, color='silver')

        labs = [l.get_label() for l in l1 + l2 + l3]
        ax1[0].legend(l1 + l2 + l3, labs, loc='upper left')
        ax1[0].set_title('CPU and Memory')

        ax1[1].plot(_time, self.calc_data(self.disk_read_speed, include_children=True) / 1024**2, label='Disk Read', linestyle='None', marker='o')
        ax1[1].plot(_time, self.calc_data(self.disk_write_speed, include_children=True) / 1024**2, label='Disk Write', linestyle='None', marker='x')
        ax1[1].set_xlabel('Time (s)')
        ax1[1].set_ylabel('Disk (MB/s)')
        ax1[1].xaxis.grid(which='major', linestyle='-', color='gray')
        ax1[1].xaxis.grid(which='minor', linestyle='--', linewidth=0.5, color='silver')
        ax1[1].legend()
        ax1[1].set_title('Disk')

        fig, ax1 = plt.subplots(1, len(self.children) + 1)
        if len(self.children) == 0:
            ax1 = [ax1]
        for i, child in enumerate(self.children.union({self.process})):
            _start = self.cpu_percent[child.pid].start_index
            _stop = self.cpu_percent[child.pid].stop_index if self.cpu_percent[child.pid].stop_index != np.inf else len(self.timestamp)
            ax1[i].plot(_time[_start:_stop], np.array(self.cpu_percent[child.pid].data), label='CPU', linestyle='None', marker='o')
            ax1[i].plot(_time[_start:_stop], np.array(self.memory_real[child.pid].data) / 1024**2, label='Memory Real', linestyle='None', marker='x')
            ax1[i].plot(_time[_start:_stop], np.array(self.memory_virtual[child.pid].data) / 1024**2, label='Memory Virtual', linestyle='None', marker='x')
            ax1[i].set_xlabel('Time (s)')
            ax1[i].set_ylabel('Memory (MB)')
            ax1[i].yaxis.grid(which='major', linestyle='-', color='gray')
            ax1[i].xaxis.grid(which='major', linestyle='-', color='gray')
            ax1[i].xaxis.grid(which='minor', linestyle='--', linewidth=0.5, color='silver')
            ax1[i].legend()
            ax1[i].set_title(f'Process {child.pid}')

        fig, ax1 = plt.subplots(1, len(self.children) + 1)
        if len(self.children) == 0:
            ax1 = [ax1]
        for i, child in enumerate(self.children.union({self.process})):
            _start = self.cpu_percent[child.pid].start_index
            _stop = self.cpu_percent[child.pid].stop_index if self.cpu_percent[child.pid].stop_index != np.inf else len(self.timestamp)
            ax1[i].plot(_time[_start:_stop], np.array(self.disk_read_speed[child.pid].data) / 1024**2, label='Disk Read', linestyle='None', marker='o')
            ax1[i].plot(_time[_start:_stop], np.array(self.disk_write_speed[child.pid].data) / 1024**2, label='Disk Write', linestyle='None', marker='x')
            ax1[i].set_xlabel('Time (s)')
            ax1[i].set_ylabel('Disk (MB/s)')
            ax1[i].xaxis.grid(which='major', linestyle='-', color='gray')
            ax1[i].xaxis.grid(which='minor', linestyle='--', linewidth=0.5, color='silver')
            ax1[i].legend()
            ax1[i].set_title(f'Process {child.pid}')

        plt.show()

    def __repr__(self):
        return f'Process({self.pid})'

    def __str__(self):
        return f'Process({self.pid})'


if __name__ == '__main__':
    # pid = "rsync --bwlimit=128 /home/ethan/ramdisk/testfile /home/ethan/Downloads/testfile"
    pid = "/home/ethan/zyc/TUD/SS/assignment1/target/release/mygrep torvalds /home/ethan/zyc/TUD/SS/linux"
    process = Process(pid)
    process.monitor(include_children=True, interval=None)

    process.get_io_speed(include_children=True)

    process.save('test.log', include_children=True)
    process.plot()
