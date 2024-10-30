import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import numpy as np
import os


def filter_data(filter_module, data):
    """Apply filter to the data."""
    filter_module.accept(data)
    return filter_module.generateEvents()


class DVS:
    def __init__(self, args):
        # Hyperparameters for saving
        self.max_event_num = args.max_event_num
        self.fps = args.fps
        self.display_scale = args.display_scale

        # Initialize device and slicer
        self.device = dv.io.CameraCapture()
        self.slicer = dv.EventMultiStreamSlicer("events")
        self.slicer.addFrameStream("frames")

        # Device settings
        self.resolution = self.device.getEventResolution()
        self.noise_filter = args.noise_filter
        self.decay_filter = args.decay_filter
        self.remain_time = 0.0
        self.recording = False

        # Visualization
        self.visualizer = dv.visualization.EventVisualizer(
            self.resolution,
            dv.visualization.colors.white(),
            dv.visualization.colors.green(),
            dv.visualization.colors.red()
        )

        # Storage for events and frames
        self.frame_image_store = []
        self.event_image_store = []

        # Initialize filters and add jobs
        self.frame_filters = []
        self.event_filters = []
        self.add_event_filters()
        self.add_jobs()

    def add_jobs(self):
        """Run functions at regular intervals based on fps."""
        interval = timedelta(milliseconds=1000 // self.fps)
        self.slicer.doEveryTimeInterval(interval, self.display_and_save_frame)

    def run(self, recording, remain_time):
        """Run the DVS device and update state."""
        self.recording = recording
        self.remain_time = remain_time

        if self.device.isRunning():
            event_batch = self.device.getNextEventBatch()
            if event_batch:
                for f in self.event_filters:
                    event_batch = filter_data(f, event_batch)
                self.slicer.accept("events", event_batch)

            frame = self.device.getNextFrame()
            if frame:
                for f in self.frame_filters:
                    frame = filter_data(f, frame)
                self.slicer.accept("frames", [frame])

        if not recording:
            self.empty()

    def add_event_filters(self):
        """Add noise and decay filters for events."""
        if self.noise_filter:
            noise_filter = dv.noise.BackgroundActivityNoiseFilter(
                resolution=self.resolution,
                backgroundActivityDuration=timedelta(milliseconds=1)
            )
            self.event_filters.append(noise_filter)

        if self.decay_filter:
            decay_filter = dv.noise.FastDecayNoiseFilter(
                resolution=self.resolution,
                halfLife=timedelta(milliseconds=1),
                subdivisionFactor=1,
                noiseThreshold=1.0
            )
            self.event_filters.append(decay_filter)

    def display_and_save_frame(self, data):
        """Display frames and events with overlay."""
        frames = data.getFrames("frames")
        events = data.getEvents("events")

        if not frames:
            return

        frame_image = frames[-1].image  # (260, 346)
        if len(frame_image.shape) != 3:
            frame_image = cv.cvtColor(frame_image, cv.COLOR_GRAY2BGR)  # (260, 346, 3)

        event_image = np.zeros_like(frame_image)
        event_image = self.visualizer.generateImage(events, event_image)

        frame_image = cv.flip(frame_image, 1)
        event_image = cv.flip(event_image, 1)

        self.frame_image_store.append(frame_image)
        self.event_image_store.append(event_image)

        image = np.concatenate((frame_image, event_image), axis=1)
        h, w = self.resolution

        # Add recording status
        text = f"RECORDING {self.remain_time:.2f}" if self.recording else f"WAITING {self.remain_time:.2f}"
        color = (0, 0, 255) if self.recording else (255, 0, 0)
        image = cv.rectangle(image, (0, 0), (h, w), color=color, thickness=15)
        image = cv.putText(image, text, org=(30, 150), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=1.3, color=color, thickness=2)

        # Resize and show image
        height, width = image.shape[:2]
        image = cv.resize(image, (int(width * self.display_scale), int(height * self.display_scale)))
        cv.imshow("Preview", image)

        if cv.waitKey(1) == 27:  # Exit on ESC key
            exit(0)

    def load_data(self):
        """Return the stored events and frames."""
        return self.event_image_store, self.frame_image_store

    def empty(self):
        """Clear stored events and frames."""
        self.event_image_store.clear()
        self.frame_image_store.clear()




    """ abandoned method"""
    def store_data(self, data):
        """Store frames and events during recording."""
        if self.recording:
            frames = data.getFrames("frames")
            events = data.getEvents("events")
            if frames:
                self.frame_image_store.append(frames[-1].image)
            if events is not None:
                # Limit the number of events
                events = np.random.choice(events.numpy(),
                                          size=min(self.max_event_num, len(events)),
                                          replace=False)
                self.event_image_store.append(events)