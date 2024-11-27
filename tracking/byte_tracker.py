#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Kend
@Date: 2024/11/23
@Time: 14:54
@Description: byte_tracker - 文件描述
@Modify:
@Contact: tankang0722@gmail.com
"""


import numpy as np
from tracking.tracker.base_track import BaseTrack, TrackState
from tracking.tracker import matching
from tracking.tracker.kalman_filter import KalmanFilter
from loguru import logger


class BYTETracker(object):

    """用于多目标跟踪任务，基于检测框的跟踪"""
    # fps = 30
    # track_thresh = 0.5
    # track_buffer = 30
    # match_thresh = 0.8
    # min_box_area = 10
    # aspect_ratio_thresh = 1.6

    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]  # 存储当前帧中活跃的跟踪对象。
        self.lost_stracks = []  # type: list[STrack]    # 存储当前帧中丢失的跟踪对象。
        self.removed_stracks = []  # type: list[STrack]  # 存储当前帧中已删除的跟踪对象。
        self.frame_id = 0   # frame_id: 当前处理的帧ID。
        self.args = args    # 跟踪参数，通常是一个包含各种配置的命名空间对象。
        self.det_thresh = args.track_thresh  # 检测阈值，用于过滤检测结果。
        # self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)  # 缓冲区大小，用于存储跟踪对象的帧数。根据帧率和跟踪缓冲参数计算。
        self.buffer_size = int(frame_rate / args.fps * args.track_buffer)  # 缓冲区大小。根据帧率和跟踪缓冲参数计算。
        self.max_time_lost = self.buffer_size  # 最大丢失时间，超过这个时间的跟踪对象将被移除。
        self.kalman_filter = KalmanFilter()   # 卡尔曼滤波器，用于预测跟踪对象的下一帧位置。


    # 更新跟踪结果
    def update(self, output_results, img_info, img_size=(640, 640)):
        """接受检测头传递的结果数据"""
        # print('output_results=', output_results)
        # 每次传入新的结果都需要更新帧ID:
        self.frame_id += 1
        # 初始化当前帧的跟踪状态:
        activated_starcks = []  # 存储当前帧中新激活的跟踪对象。
        refind_stracks = []     # 存储当前帧中重新激活的跟踪对象。
        lost_stracks = []       # 存储当前帧中丢失的跟踪对象。
        removed_stracks = []    # 存储当前帧中移除的跟踪对象。
        if output_results.shape[1] == 6:  # xyxy score cls
            scale_ = False
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        # print(scale, "scale")
        # print(f"scores:{scores}")  # scores:[0.9605565  0.9561169  0.9507537  0.94548774 0.9430547  0.9289429.....]
        if scale_:  # 如果需要缩放检测框 取决于进来的张量数据
            bboxes /= scale
        # print(f"bboxes:{bboxes}")  # bboxes:[[  422.55       229.04999    548.55       504.44998 ].....

        """step1: 处理检测结果 """
        remain_inds = scores > self.args.track_thresh   # 追踪阈值
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh     # 追踪阈值
        inds_second = np.logical_and(inds_low, inds_high)  # 与
        dets_second = bboxes[inds_second]  # 存储分数在0.1到追踪阈值之间的检测框。
        dets = bboxes[remain_inds]   # 存储分数大于追踪阈值的检测框。
        scores_keep = scores[remain_inds]  # 存储分数大于追踪阈值的检测框的分数。
        scores_second = scores[inds_second]  # 存储分数在0.1到追踪阈值之间的检测框的分数。
        # 如果存在分数大于追踪阈值的目标
        if len(dets) > 0:
            ''' Step 1: 对分数大于追踪阈值的目标进行编码'''
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s)  # top-left, bottom-right -> top-left, width, height
                for(tlbr, s) in zip(dets, scores_keep)  # 取得阈值高的位置和阈值信息
                ]
        else:
            detections = []

        '''处理未确认的跟踪对象'''
        unconfirmed = []    # 存储当前帧中未确认的跟踪对象。
        tracked_stracks = []  # type: list[STrack]  # 存储当前帧中已确认的跟踪对象。
        for track in self.tracked_stracks:  # 遍历当前帧中所有活跃的跟踪对象。
            if not track.is_activated:  # 如果跟踪对象未激活，将其添加到 unconfirmed 列表。
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)   # 如果跟踪对象已激活，将其添加到 tracked_stracks 列表。

        ''' Step 2: 第一次匹配'''
        # 将已确认的跟踪对象和丢失的跟踪对象合并。
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # 使用卡尔曼滤波器预测所有跟踪对象的位置。
        STrack.multi_predict(strack_pool)
        # 计算已确认的跟踪对象和检测框之间的IOU距离。
        dists = matching.iou_distance(strack_pool, detections)
        #  融合得分。
        dists = matching.fuse_score(dists, detections)
        # 使用线性分配算法进行匹配。
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        # 更新匹配成功的跟踪对象
        for itracked, idet in matches: # 遍历匹配结果。
            track = strack_pool[itracked]   # 获取匹配的跟踪对象。
            det = detections[idet]  # 获取匹配的检测结果。
            if track.state == TrackState.Tracked:  # 如果跟踪对象是已激活状态，更新其状态。
                track.update(detections[idet], self.frame_id)   # 更新跟踪对象的状态。
                activated_starcks.append(track) # 将更新后的跟踪对象添加到 activated_starcks 列表。
            else:
                # 如果跟踪对象不是已激活状态，重新激活它。
                track.re_activate(det, self.frame_id, new_id=False) # : 重新激活跟踪对象。
                refind_stracks.append(track)    # 将重新激活的跟踪对象添加到 refind_stracks 列表。

        ''' Step 3: 第二次匹配'''
        # 如果有得分在0.1到 track_thresh 之间的检测框。
        if len(dets_second) > 0:
            '''将检测框和得分转换为 STrack 对象。'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            # 如果没有得分在0.1到 track_thresh 之间的检测框，detections_second 列表为空。
            detections_second = []
        """第二次匹配继续"""
        # 获取未匹配的已激活跟踪对象。
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # 计算未匹配的已激活跟踪对象与低得分检测结果之间的IoU距离。
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # 使用线性分配算法进行匹配。
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.7)
        # 更新第二次匹配成功的跟踪对象
        for itracked, idet in matches:  # 遍历匹配结果。
            track = r_tracked_stracks[itracked] # 获取匹配的跟踪对象。
            det = detections_second[idet]   # 获取匹配的检测结果。
            #  如果跟踪对象是已激活状态，更新其状态。
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)    # 更新跟踪对象的状态。
                activated_starcks.append(track) # 将更新后的跟踪对象添加到 activated_starcks 列表。
            else:   # 如果跟踪对象不是已激活状态，重新激活它。
                track.re_activate(det, self.frame_id, new_id=False) # 重新激活跟踪对象。
                refind_stracks.append(track)    # 将重新激活的跟踪对象添加到 refind_stracks 列表。
        # 遍历未匹配的已激活跟踪对象。
        for it in u_track:
            track = r_tracked_stracks[it] # 获取未匹配的跟踪对象。
            if not track.state == TrackState.Lost:  # 如果跟踪对象不是丢失状态，标记为丢失。
                track.mark_lost()   #  标记跟踪对象为丢失。
                lost_stracks.append(track)  # 将丢失的跟踪对象添加到 lost_stracks 列表。


        '''处理未确认的跟踪对象'''
        detections = [detections[i] for i in u_detection]   # 获取未匹配的高得分检测结果。
        dists = matching.iou_distance(unconfirmed, detections)  # 计算未确认的跟踪对象与未匹配的高得分检测结果之间的IoU距离。
        # 融合得分。
        dists = matching.fuse_score(dists, detections)  # 使用线性分配算法进行匹配。
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        # 更新未确认的跟踪对象
        for itracked, idet in matches: # 遍历匹配结果。
            unconfirmed[itracked].update(detections[idet], self.frame_id)   # 更新未确认的跟踪对象。
            activated_starcks.append(unconfirmed[itracked]) # 将更新后的未确认跟踪对象添加到 activated_starcks 列表。
        # 遍历未匹配的未确认跟踪对象。
        for it in u_unconfirmed:
            track = unconfirmed[it] # 获取未匹配的未确认跟踪对象。
            track.mark_removed()    #  标记未确认的跟踪对象为移除。
            removed_stracks.append(track)   #  将移除的跟踪对象添加到 removed_stracks 列表。

        """ Step 4: 初始化新的跟踪对象"""
        for inew in u_detection:    # 遍历未匹配的高得分检测结果。
            track = detections[inew]    #  获取未匹配的检测结果。
            if track.score < self.det_thresh:   # 如果检测结果的得分低于 det_thresh，跳过。
                continue
            track.activate(self.kalman_filter, self.frame_id)   # 激活新的跟踪对象
            activated_starcks.append(track) # 将激活的跟踪对象添加到 activated_starcks 列表。

        """ Step 5: 更新丢失的跟踪对象"""
        # 遍历丢失的跟踪对象。
        for track in self.lost_stracks: # 如果跟踪对象丢失的时间超过 max_time_lost，标记为移除。
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()    # 标记跟踪对象为移除。
                removed_stracks.append(track)   # 将移除的跟踪对象添加到 removed_stracks 列表。

        # print('Ramained match {} s'.format(t4-t3))
        """last: 更新跟踪状态 """
        # 更新当前帧中活跃的跟踪对象列表。
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # 合并当前帧中活跃的跟踪对象和新激活的跟踪对象。
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        # 合并当前帧中活跃的跟踪对象和重新激活的跟踪对象。
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # 从丢失的跟踪对象中移除已激活的跟踪对象。
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        # 将丢失的跟踪对象添加到 lost_stracks 列表。
        self.lost_stracks.extend(lost_stracks)
        # 从丢失的跟踪对象中移除已移除的跟踪对象。
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        # 将移除的跟踪对象添加到 removed_stracks 列表。
        self.removed_stracks.extend(removed_stracks)
        # 移除重复的跟踪对象。
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # 获取当前帧中活跃的跟踪对象。 返回当前帧中活跃的跟踪对象
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks



""" STrack对象 是一个用于目标跟踪的类，主要使用卡尔曼滤波器来估计目标的状态 """
class STrack(BaseTrack):
    # shared_kalman 是一个静态属性，表示所有 STrack 实例共享同一个卡尔曼滤波器实例。
    shared_kalman = KalmanFilter()  # 卡尔曼滤波器用于估计目标的状态，从而提高跟踪的准确性。

    def __init__(self, tlwh, score):
        """
        Args:
            tlwh: 目标的初始边界框，格式为 (top left x, top left y, width, height)。
            score 是目标的置信度分数。
        function: STrack 类主要用于管理目标跟踪中的轨迹信息，包括轨迹的初始化、预测、更新和重新激活等功能。
        """
        # 等待被激活
        self._tlwh = np.asarray(tlwh, dtype=np.float64)   # 存储初始的边界框信息。
        self.kalman_filter = None   # 用于存储卡尔曼滤波器实例
        self.mean, self.covariance = None, None    # 分别存储卡尔曼滤波器的状态均值和协方差矩阵，初始为 None
        self.is_activated = False   # 表示该轨迹是否已激活，初始为 False
        self.score = score  # 存储目标的置信度分数
        self.tracklet_len = 0   # 记录轨迹的长度，初始为 0

    # 预测下一个时间的目标
    def predict(self):
        mean_state = self.mean.copy()   # mean_state 当前状态的副本
        # 如果当前轨迹状态不是 Tracked，则将 mean_state 的第8个元素（通常表示速度）设为 0
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        # 使用卡尔曼滤波器的 predict 方法更新 self.mean 和 self.covariance
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    # 多轨迹预测方法
    @staticmethod
    def multi_predict(stracks):
        # 批量预测多个轨迹的状态
        if len(stracks) > 0:    # 包含多个 STrack 实例的列表
            # multi_mean 和 multi_covariance 分别存储所有轨迹的状态均值和协方差矩阵
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            # 如果某个轨迹的状态不是 Tracked，则将 multi_mean 的第8个元素设为 0
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            # 使用共享的卡尔曼滤波器的 multi_predict 方法批量预测所有轨迹的状态
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            # 更新每个轨迹的 mean 和 covariance
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    # 用来激活一个新的轨迹
    def activate(self, kalman_filter, frame_id):
        """ 开始一个新的轨迹 """
        self.kalman_filter = kalman_filter # 卡尔曼滤波器实例
        # 获取一个新的轨迹ID
        self.track_id = self.next_id()
        # 卡尔曼滤波器的 initiate 方法初始化轨迹的状态均值和协方差矩阵
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        # 重置轨迹长度 tracklet_len 为 0。
        self.tracklet_len = 0
        # 设置轨迹状态为 Tracked。
        self.state = TrackState.Tracked
        # 如果当前帧编号为 1，则设置 is_activated 为 True。
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        # # 设置当前帧编号 frame_id 和轨迹起始帧编号 start_frame。
        self.frame_id = frame_id
        self.start_frame = frame_id

    """ 重新激活一个已经存在的轨迹 """
    def re_activate(self, new_track, frame_id, new_id=False):
        """
        Args:
            new_track:  new_track 是一个新的 STrack 实例，包含更新后的信息
            frame_id: 当前帧的编号。
            new_id: new_id 是一个布尔值，表示是否为轨迹分配一个新的ID。
        Returns:
        """
        # 使用卡尔曼滤波器的 update 方法更新轨迹的状态均值和协方差矩阵
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))
        # 重置轨迹长度 tracklet_len 为 0
        self.tracklet_len = 0
        # 设置轨迹状态为 Tracked。
        self.state = TrackState.Tracked
        self.is_activated = True
        # 设置当前帧编号 frame_id。
        self.frame_id = frame_id
        # # 如果 new_id 为 True，则使用 next_id 方法获取一个新的轨迹ID。
        if new_id:
            self.track_id = self.next_id()
        # # 更新轨迹的置信度分数 score。
        self.score = new_track.score

    # 更新一个匹配的轨迹
    def update(self, new_track, frame_id):
        """
        Args:
            new_track 是一个新的 STrack 实例，包含更新后的信息。
            frame_id: int 当前帧编号
        :return:
        """
        # 设置当前帧编号 frame_id
        self.frame_id = frame_id
        # 增加轨迹长度 tracklet_len
        self.tracklet_len += 1
        # 获取新的边界框信息 new_tlwh
        new_tlwh = new_track.tlwh
        # 使用卡尔曼滤波器的 update 方法更新轨迹的状态均值和协方差矩阵
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        # 设置轨迹状态为 Tracked
        self.state = TrackState.Tracked
        # 设置 is_activated 为 True
        self.is_activated = True
        # 更新轨迹的置信度分数 score
        self.score = new_track.score


    @property
    # 获取当前轨迹的边界框信息  # @property装饰器 -> 方法属性
    def tlwh(self):
        # 如果 self.mean 为 None，则返回初始的边界框信息 _tlwh。
        if self.mean is None:
            return self._tlwh.copy()
        # 否则，从卡尔曼滤波器的状态均值中提取边界框信息。
        ret = self.mean[:4].copy()
        # 将状态均值中的 (center_x, center_y, aspect_ratio, height) 转换为 (top_left_x, top_left_y, width, height) 格式。
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        # 返回转换后的边界框信息。
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)   # 这个装饰器来自于 Numba 库，Numba 是一个开源的 JIT（Just-In-Time）编译器，用于加速 Python 代码，特别是数值计算和科学计算。
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)



""" 将两个轨迹列表 tlista 和 tlistb 合并成一个新的列表，确保每个轨迹只出现一次。"""
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

"""从轨迹列表 tlista 中移除出现在轨迹列表 tlistb 中的所有轨迹。"""
def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


"""去除两个轨迹列表 stracksa 和 stracksb 中的重复轨迹。重复轨迹的判断依据是 IoU（Intersection over Union）距离小于 0.15。"""
def remove_duplicate_stracks(stracksa, stracksb):
    # IoU 距离：用于多目标跟踪中的轨迹匹配，衡量两个轨迹之间的相似度。成本矩阵中的值越小，表示两个轨迹越相似。
    pdist = matching.iou_distance(stracksa, stracksb)
    # pairs = np.where(pdist < 0.15)
    pairs = np.where(pdist < 0.2)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb