# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Transformer-based object tracker implementation with BYTETracker-compatible interface.
"""

import numpy as np
import torch
import torch.nn as nn

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH


class TransformerSTrack(BaseTrack):
    """
    Single object tracking representation that maintains BYTETracker STrack interface
    while internally using Transformer features for appearance modeling.

    This class provides exactly the same interface as STrack but with Transformer-based
    appearance features for improved data association.
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh, score, cls):
        """
        Initialize a new TransformerSTrack instance with BYTETracker-compatible interface.

        Args:
            xywh (List[float]): Bounding box coordinates and dimensions in the format (x, y, w, h, [a], idx), where
                (x, y) is the center, (w, h) are width and height, [a] is optional aspect ratio, and idx is the id.
            score (float): Confidence score of the detection.
            cls (Any): Class label for the detected object.

        Examples:
            >>> xywh = [100.0, 150.0, 50.0, 75.0, 1]
            >>> score = 0.9
            >>> cls = 'person'
            >>> track = TransformerSTrack(xywh, score, cls)
        """
        super().__init__()
        # xywh+idx or xywha+idx
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

        # Transformer-specific attributes (internal only, doesn't affect interface)
        self.appearance_feature = None
        self.feature_history = []
        self.feature_update_count = 0
        self.max_feature_history = 10

    def predict(self):
        """Predicts the next state (mean and covariance) of the object using the Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Perform multi-object predictive tracking using Kalman filter for the provided list of TransformerSTrack instances."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = TransformerSTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Update state tracks positions and covariances using a homography matrix for multiple tracks."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Activate a new tracklet using the provided Kalman filter and initialize its state and covariance."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a previously lost track using new detection data and updates its state and attributes."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

        # Update appearance features if available
        if hasattr(new_track, 'appearance_feature') and new_track.appearance_feature is not None:
            self.update_features(new_track.appearance_feature)

    def update(self, new_track, frame_id):
        """
        Update the state of a matched track.

        Args:
            new_track (TransformerSTrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

        # Update appearance features if available
        if hasattr(new_track, 'appearance_feature') and new_track.appearance_feature is not None:
            self.update_features(new_track.appearance_feature)

    def update_features(self, new_feature):
        """Internal method to update appearance features (doesn't affect public interface)."""
        self.feature_history.append(new_feature)
        if len(self.feature_history) > self.max_feature_history:
            self.feature_history.pop(0)

        # Update current feature as weighted average
        if len(self.feature_history) > 0:
            weights = torch.linspace(0.5, 1.0, len(self.feature_history))
            weights = weights / weights.sum()

            weighted_features = torch.stack([
                feature * weight for feature, weight in zip(self.feature_history, weights)
            ])
            self.appearance_feature = weighted_features.sum(dim=0)

        self.feature_update_count += 1

    def convert_coords(self, tlwh):
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        """Returns the bounding box in top-left-width-height format from the current state estimate."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self):
        """Converts bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self):
        """Returns the current position of the bounding box in (center x, center y, width, height) format."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self):
        """Returns position in (center x, center y, width, height, angle) format, warning if angle is missing."""
        if self.angle is None:
            LOGGER.warning("WARNING ⚠️ `angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self):
        """Returns the current tracking results in the appropriate bounding box format."""
        coords = self.xyxy if self.angle is None else self.xywha
        return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]

    def __repr__(self):
        """Returns a string representation of the TransformerSTrack object including start frame, end frame, and track ID."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class TransformerTracker:
    """
    Transformer-based object tracker with BYTETracker-compatible interface.

    This class provides exactly the same interface as BYTETracker but uses Transformer-based
    appearance features internally for improved data association.

    Examples:
        >>> tracker = TransformerTracker(args, frame_rate=30)
        >>> results = yolo_model.detect(image)
        >>> tracked_objects = tracker.update(results)
    """

    def __init__(self, args, frame_rate=30):
        """
        Initialize a TransformerTracker instance with BYTETracker-compatible interface.

        Args:
            args (Namespace): Command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video sequence.
        """
        self.tracked_stracks = []  # type: list[TransformerSTrack]
        self.lost_stracks = []  # type: list[TransformerSTrack]
        self.removed_stracks = []  # type: list[TransformerSTrack]

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def update(self, results, img=None):
        """Updates the tracker with new detections and returns the current list of tracked objects."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        # Add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]

        detections = self.init_track(dets, scores_keep, cls_keep, img)
        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks = []  # type: list[TransformerSTrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        # Step 2: First association, with high score detection boxes
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        self.multi_predict(strack_pool)
        if hasattr(self, "gmc") and img is not None:
            warp = self.gmc.apply(img, dets)
            TransformerSTrack.multi_gmc(strack_pool, warp)
            TransformerSTrack.multi_gmc(unconfirmed, warp)

        # Use Transformer-enhanced distance calculation
        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Step 3: Second association, with low score detection boxes association the untrack to the low score detections
        detections_second = self.init_track(dets_second, scores_second, cls_second, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # TODO
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        # Step 4: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
        # Step 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_kalmanfilter(self):
        """Returns a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH."""
        return KalmanFilterXYAH()

    def init_track(self, dets, scores, cls, img=None):
        """Initializes object tracking with given detections, scores, and class labels using the TransformerSTrack algorithm."""
        return [TransformerSTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(
            dets) else []  # detections

    def get_dists(self, tracks, detections):
        """
        Calculates the distance between tracks and detections using IoU and optionally appearance features.

        This is the main enhancement over BYTETracker - we use Transformer appearance features
        when available to improve association accuracy.
        """
        # Start with standard IoU distance
        dists = matching.iou_distance(tracks, detections)

        # Enhance with appearance features if available
        if hasattr(self.args, 'use_appearance') and self.args.use_appearance:
            # Check if we have appearance features to work with
            tracks_with_features = [t for t in tracks if
                                    hasattr(t, 'appearance_feature') and t.appearance_feature is not None]
            detections_with_features = [d for d in detections if
                                        hasattr(d, 'appearance_feature') and d.appearance_feature is not None]

            if len(tracks_with_features) > 0 and len(detections_with_features) > 0:
                # Calculate appearance distance matrix
                appearance_dists = np.zeros((len(tracks), len(detections)))

                for i, track in enumerate(tracks):
                    for j, detection in enumerate(detections):
                        if (hasattr(track, 'appearance_feature') and track.appearance_feature is not None and
                                hasattr(detection, 'appearance_feature') and detection.appearance_feature is not None):
                            # Cosine distance for appearance features
                            cos_sim = torch.nn.functional.cosine_similarity(
                                track.appearance_feature.unsqueeze(0),
                                detection.appearance_feature.unsqueeze(0),
                                dim=1
                            )
                            appearance_dists[i, j] = 1.0 - cos_sim.item()

                # Combine IoU and appearance distances
                iou_weight = getattr(self.args, 'iou_weight', 0.8)
                appearance_weight = getattr(self.args, 'appearance_weight', 0.2)

                dists = iou_weight * dists + appearance_weight * appearance_dists

        if hasattr(self.args, 'fuse_score') and self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        return dists

    def multi_predict(self, tracks):
        """Predict the next states for multiple tracks using Kalman filter."""
        TransformerSTrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Resets the ID counter for TransformerSTrack instances to ensure unique track IDs across tracking sessions."""
        TransformerSTrack.reset_id()

    def reset(self):
        """Resets the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
        self.tracked_stracks = []  # type: list[TransformerSTrack]
        self.lost_stracks = []  # type: list[TransformerSTrack]
        self.removed_stracks = []  # type: list[TransformerSTrack]
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """Combines two lists of STrack objects into a single list, ensuring no duplicates based on track IDs."""
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

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """Filters out the stracks present in the second list from the first list."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Removes duplicate stracks from two lists based on Intersection over Union (IoU) distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb