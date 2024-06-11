import datetime
import os
import pickle
from datetime import timedelta

import cv2 as cv
import dv_processing as dv
import numpy as np


def filter_data(filter_module, data):
    filter_module.accept(data)
    return filter_module.generateEvents()


class DVSmodel:
    def __init__(self, args):
        # initialize device
        self.device = dv.io.CameraCapture()
        assert self.device.getCameraName(), "No camera found"

        # record setting
        self.record_interval = args.record_interval
        self.record_time = args.record_time
        self.fps = args.fps
        assert self.fps * self.record_time % 1 == 0, "fps * record time must be an integer"

        self.record_start_time = None
        self.record_end_time = None
        self.recording = None

        # set slicer
        self.slicer = dv.EventMultiStreamSlicer("events")
        self.slicer.addFrameStream("frames")

        #
        self.resolution = self.device.getEventResolution()

        # set filters
        self.frame_filters = []
        self.events_filters = []

        # initialize functions
        self.add_event_filter(noise_filter=args.noise_filter,
                              decay_filter=args.decay_filter)

        self.visualizer = dv.visualization.EventVisualizer(self.resolution,
                                                           dv.visualization.colors.white(),
                                                           dv.visualization.colors.green(),
                                                           dv.visualization.colors.red())

        # Event Store
        self.event_store = []
        self.frame_store = []

        #

        # SAVE - setting
        self.max_event_num = args.max_event_num

        # SAVE - parameter
        self.id = args.id
        self.action = args.action
        self.i = args.i
        self.save_folder_path = f'{args.save_folder}/{args.id:03}/'
        self.prefix = "SubClsIdx"
        self.repeat = args.repeat

        # INIT
        self.init()

    def init(self):
        # on window
        cv.namedWindow("Preview", cv.WINDOW_NORMAL)

        # do display function
        self.slicer.doEveryTimeInterval(timedelta(milliseconds=1000 // self.fps),
                                        self.display)
        self.slicer.doEveryTimeInterval(timedelta(milliseconds=1000 // self.fps),
                                        self.save)

    def empty_store(self):
        # Event Store
        self.event_store = []
        self.frame_store = []

    def add_event_filter(self, **kwargs):
        for filter_name, enabled in kwargs.items():
            if enabled:
                events_filter = None
                if filter_name == "noise_filter":
                    events_filter = dv.noise.BackgroundActivityNoiseFilter(resolution=self.resolution,
                                                                           backgroundActivityDuration=timedelta(
                                                                               milliseconds=1))
                elif filter_name == "decay_filter":
                    events_filter = dv.noise.FastDecayNoiseFilter(resolution=self.resolution,
                                                                  halfLife=timedelta(milliseconds=1),
                                                                  subdivisionFactor=1,
                                                                  noiseThreshold=1.0)
                self.events_filters.append(events_filter)

    def save(self, data):
        if self.recording:
            frames = data.getFrames("frames")
            events = data.getEvents("events")
            if len(frames) > 0:
                self.frame_store.append(frames[-1].image)
            if events is not None:
                self.event_store.append(events)

    def display(self, data):
        frames = data.getFrames("frames")
        events = data.getEvents("events")

        latest_image = None
        if len(frames) > 0:
            if len(frames[-1].image.shape) == 3:
                latest_image = frames[-1].image
            else:
                latest_image = cv.cvtColor(frames[-1].images, cv.COLOR_GRAY2BGR)
        else:
            return

        # add rectangle of CV view
        color = (0, 0, 255) if self.recording else (255, 0, 0)  # (R,G,B)
        image = self.visualizer.generateImage(events, latest_image)
        image = cv.rectangle(image, (0, 0), self.resolution, color=color, thickness=15)

        # flip
        image = cv.flip(image, 1)

        # add text
        if self.recording:
            text = "RECODING"
        else:
            text = f"   {self.record_start_time - datetime.datetime.now().timestamp():.2f}"

        image = cv.putText(img=image,
                       text=text,
                       org=(30, 150),
                       fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=2,
                       color=(150, 150, 150),
                       thickness=2)

        cv.imshow("Preview", image)

        if cv.waitKey(1) == 27:  # 10:enter, 27: esc, 32: space,
            exit(0)

    def record(self):
        self.recording = False
        # time setting
        start_time = datetime.datetime.now().timestamp()
        self.record_start_time = start_time + self.record_interval
        self.record_end_time = start_time + self.record_time + self.record_interval + 0.05

        while self.device.isRunning():
            events = self.device.getNextEventBatch()
            if events is not None:
                self.slicer.accept("events", events)

            frame = self.device.getNextFrame()
            if frame is not None:
                for f in self.frame_filters:
                    frame = filter_data(f, frame)
                self.slicer.accept("frames", [frame])

            # compute cur_time
            cur_time = datetime.datetime.now().timestamp()

            # start the recoding
            self.recording = cur_time >= self.record_start_time

            # finish the recoding
            if cur_time >= self.record_end_time:
                break

    def filtering(self):
        for i in range(len(self.event_store)):
            for event_filter in self.events_filters:
                self.event_store[i] = filter_data(event_filter, self.event_store[i])

    def save_data(self):
        # get path
        event_path, frame_path = self.get_paths()

        # transform and cut using max_event_num
        for i in range(len(self.event_store)):
            event_np = self.event_store[i].numpy()
            self.event_store[i] = np.random.choice(event_np,
                                                   size=min(self.max_event_num, len(event_np)),
                                                   replace=False)

        data_len = self.fps * self.record_time
        # check the length of data
        if len(self.frame_store) < data_len or len(self.event_store) < data_len:
            print("The data is too short to save")
            return
        else:
            self.frame_store = self.frame_store[:data_len]
            self.event_store = self.event_store[:data_len]

        # save event
        with open(event_path, 'wb') as f:
            pickle.dump(self.event_store, f)
        print(event_path)

        # save frame
        frames = np.stack(self.frame_store)
        np.save(frame_path, frames)
        print(frame_path)

    def get_paths(self):
        save_folder_path = self.save_folder_path
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
            print(f'{save_folder_path} has been created.')

        while True:
            assert 0 <= self.i < 1000, "Invalid index range"
            frame_path = save_folder_path + f'{self.prefix}_{self.id:03}_{self.action:03}_{self.i:03}.npy'
            event_path = save_folder_path + f'{self.prefix}_{self.id:03}_{self.action:03}_{self.i:03}.pkl'
            if not os.path.exists(frame_path) and not os.path.exists(event_path):
                break
            self.i += 1
        return event_path, frame_path

    def start(self):
        self.record_interval += 5  # initial waiting.
        for i in range(self.repeat):
            if i == 1:
                self.record_interval -= 5
            self.record()
            self.filtering()
            print("Saving...")
            self.save_data()
            self.empty_store()

    def stop(self):
        self.slicer.removeJob(1)
        self.slicer.removeJob(2)
        cv.destroyAllWindows()
        print("fin")
