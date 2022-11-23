import unittest
from main import Process
import time
import numpy as np

class TestProcess(unittest.TestCase):
    def test_grep(self):
        command = "/home/ethan/zyc/TUD/SS/assignment1/target/release/mygrep torvalds /home/ethan/zyc/TUD/SS/linux"
        include_children = True
        p = Process(command)
        p.monitor(include_children=include_children)
        p.get_io_speed(include_children=include_children)
        p.calc_data(p.cpu_percent, include_children=include_children)
        p.calc_data(p.memory_real, include_children=include_children)
        p.calc_data(p.memory_virtual, include_children=include_children)
        p.calc_data(p.disk_read_speed, include_children=include_children)
        p.calc_data(p.disk_write_speed, include_children=include_children)

    def test_rsync(self):
        command = "rsync --bwlimit=128 /home/ethan/ramdisk/testfile /home/ethan/Downloads/testfile"
        include_children = True
        p = Process(command)
        p.monitor(include_children=include_children)
        p.get_io_speed(include_children=include_children)
        p.calc_data(p.cpu_percent, include_children=include_children)
        p.calc_data(p.memory_real, include_children=include_children)
        p.calc_data(p.memory_virtual, include_children=include_children)
        p.calc_data(p.disk_read_speed, include_children=include_children)
        p.calc_data(p.disk_write_speed, include_children=include_children)

if __name__ == '__main__':

    for _ in range(10):
        suite = unittest.TestSuite()
        suite.addTest(TestProcess('test_grep'))
        unittest.TextTestRunner().run(suite)