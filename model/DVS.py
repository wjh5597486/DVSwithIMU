import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import datetime
import numpy as np
import pickle
import os


def filter_data(filter_module, data):
    filter_module.accept(data)
    return filter_module.generateEvents()


class DVS:
    def __init__(self, args):
        print(args)
        # SAVE - hyperparameter
        self.max_event_num = args.max_event_num
        self.fps = args.fps
        self.display_scale = args.display_scale

        # devices
        self.device = dv.io.CameraCapture()
        self.slicer = dv.EventMultiStreamSlicer("events")
        self.slicer.addFrameStream("frames")

        # device parameter
        self.resolution = self.device.getEventResolution()
        self.noise_filter = args.noise_filter
        self.decay_filter = args.decay_filter
        self.remain_time = None
        self.recording = None

        self.visualizer = dv.visualization.EventVisualizer(self.resolution,
                                                           dv.visualization.colors.white(),
                                                           dv.visualization.colors.green(),
                                                           dv.visualization.colors.red())

        # Event Store
        self.event_store = []
        self.frame_store = []

        # set filters
        self.frame_filters = []
        self.event_filter = []

        # initialize functions
        self.add_event_filter()
        self.add_jobs()

    def add_jobs(self):
        """ do every time Interval of timedelta(...) the function."""
        self.slicer.doEveryTimeInterval(timedelta(milliseconds=1000 // self.fps),
                                        self.display)
        self.slicer.doEveryTimeInterval(timedelta(milliseconds=1000 // self.fps),
                                        self.store_data)

    def run(self, recording, remain_time):
        # update state.
        self.recording = recording
        self.remain_time = remain_time

        # get event and frame.
        if self.device.isRunning():
            event = self.device.getNextEventBatch()
            if event is not None:
                for f in self.event_filter:
                    event = filter_data(f, event)
                self.slicer.accept("events", event)

            frame = self.device.getNextFrame()
            if frame is not None:
                for f in self.frame_filters:
                    frame = filter_data(f, frame)
                self.slicer.accept("frames", [frame])

        if not recording:
            self.empty()

    def add_event_filter(self, **kwargs):
        if self.noise_filter:
            events_filter = dv.noise.BackgroundActivityNoiseFilter(resolution=self.resolution,
                                                                   backgroundActivityDuration=timedelta(
                                                                       milliseconds=1))
            self.event_filter.append(events_filter)

        if self.decay_filter:
            events_filter = dv.noise.FastDecayNoiseFilter(resolution=self.resolution,
                                                          halfLife=timedelta(milliseconds=1),
                                                          subdivisionFactor=1,
                                                          noiseThreshold=1.0)
            self.event_filter.append(events_filter)

    def display(self, data):
        frames = data.getFrames("frames")
        events = data.getEvents("events")

        # check frame data is available and check the dimension(color) of the frame
        if len(frames) > 0:
            if len(frames[-1].image.shape) == 3:
                latest_image = frames[-1].image
            else:
                latest_image = cv.cvtColor(frames[-1].images, cv.COLOR_GRAY2BGR)
        else:
            return

        # Overlapping Events on the image
        image = self.visualizer.generateImage(events, latest_image)

        # flipping horizontal
        image = cv.flip(image, 1)

        # add a frame of image and text.
        if self.recording:
            text = f"RECODING {self.remain_time:.2f}"
            color = (0, 0, 255)  # RGB
        else:
            text = f"WAITING {self.remain_time:.2f}"
            color = (255, 0, 0)  # BGR
        image = cv.rectangle(image, (0, 0), self.resolution, color=color, thickness=15)
        image = cv.putText(img=image, text=text, org=(30, 150), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=1.3, color=color, thickness=2)

        # scale
        height, width = image.shape[:2]
        image = cv.resize(image, (int(width*self.display_scale), int(height*self.display_scale)),
                          interpolation=cv.INTER_LINEAR)

        # cv show
        cv.imshow("Preview", image)
        if cv.waitKey(1) == 27:  # 10:enter, 27: esc, 32: space,
            exit(0)

    def store_data(self, data):
        if self.recording:
            frames = data.getFrames("frames")
            events = data.getEvents("events")
            if len(frames) > 0:
                self.frame_store.append(frames[-1].image)
            if events is not None:
                """ cut the amount of events"""
                events = events.numpy()
                events = np.random.choice(events,
                                          size=min(self.max_event_num, len(events)),
                                          replace=False)
                self.event_store.append(events)
        else:
            pass

    def load_data(self):
        return self.event_store, self.frame_store

    def empty(self):
        self.event_store = []
        self.frame_store = []


