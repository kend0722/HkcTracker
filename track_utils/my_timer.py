#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 15:22
@Description: timer - 这个 Timer 类提供了一个简单而有效的方法来测量代码片段的执行时间。
通过 start() 和 stop() 方法，可以方便地启动和停止计时器，并获取所需的计时结果。
@Modify: clear() 方法则用于重置计时器，以便在多次测量中保持准确性。
@Contact: tankang0722@gmail.com
"""

import time


class MyTimer(object):
    """ 时间计时器实例"""

    def __init__(self):
        """
        total_time: 累积的总时间。
        calls: 计时器被调用的次数。
        start_time: 开始计时的时间点。
        diff: 单次计时的差值。
        average_time: 平均每次调用的时间。
        duration: 最近一次计时的结果，可以是平均时间或单次时间差。
        """
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.

    # 启动计时器。
    def start(self):
        """ 使用 time.time() 而不是 time.clock()，因为 time.clock() 在多线程环境中可能不准确。"""
        # 记录当前时间作为计时的起点。
        self.start_time = time.time()

    # 停止计时器并计算时间差。
    def stop(self, average=True):
        """ average: 如果为 True，返回平均时间；否则，返回单次时间差。"""
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    # 重置计时器。
    def clear(self):
        """ 将所有计时器相关变量重置为初始状态。"""
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.


if __name__ == '__main__':
    timer = MyTimer()
    # 启动计时器
    timer.start()
    # 模拟一些耗时操作
    time.sleep(2)
    # 停止计时器并获取平均时间
    duration = timer.stop()
    print(f"Duration: {duration} seconds")
    # 再次启动计时器
    timer.start()
    # 模拟另一段耗时操作
    time.sleep(3)
    # 停止计时器并获取单次时间差
    duration = timer.stop(average=False)
    print(f"Single duration: {duration} seconds")
    # 打印平均时间
    print(f"Average duration: {timer.average_time} seconds")
    # 清除计时器
    timer.clear()

    # Duration: 2.000274658203125 seconds
    # Single duration: 3.00106143951416 seconds
    # Average duration: 2.5006680488586426 seconds