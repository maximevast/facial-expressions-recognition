#!/usr/bin/env python3

class Frame:
    def __init__(self, timestamp, binary_class):
        self.timestamp    = timestamp
        self.binary_class = binary_class
        self.points       = []

    def __repr__(self):
        if self.binary_class == 1:
            return "{0} contains {1} points, with occurring EFG".format(self.timestamp, len(self.points))
        else:
            return "{0} contains {1} points, without occurring EFG".format(self.timestamp, len(self.points))

    def flatten(self):
        """ Convert a list of points to a list of numbers as classifiers don't handle 3d matrix
        [(0, 1, 2), (3, 4, 5)] becomes [0, 1, 2, 3, 4, 5]
        """
        lst = []
        for point in self.points:
            lst.extend([point.x, point.y, point.z])

        return lst


class Point:
    def __init__(self, point_id, x, y, z):
        """ Initialize instance
        :param  x: Coordinate in pixels
        :type   x: double.
        :param  y: Coordinate in pixels
        :type   y: double.
        :param  z: Coordinate in millimeters
        :type   z: double.
        """
        self.id = point_id
        self.x  = x
        self.y  = y
        self.z  = z

        self.define_position()

    def define_position(self):
        """
        0 - 7   (x,y,z) - left eye
        8 - 15  (x,y,z) - right eye
        16 - 25 (x,y,z) - left eyebrow
        26 - 35 (x,y,z) - right eyebrow
        36 - 47 (x,y,z) - nose
        48 - 67 (x,y,z) - mouth
        68 - 86 (x,y,z) - face contour
        87              (x,y,z) - left iris
        88              (x,y,z) - right iris
        89              (x,y,z) - nose tip
        90 - 94 (x,y,z) - line above left eyebrow
        95 - 99 (x,y,z) - line above right eyebrow
        """

        if self.id >= 0 and self.id <= 7:
            self.position = "left eye"
        elif self.id >= 8 and self.id <= 15:
            self.position = "right eye"
        elif self.id >= 16 and self.id <= 25:
            self.position = "left eyebrow"
        elif self.id >= 26 and self.id <= 35:
            self.position = "right eyebrow"
        elif self.id >= 36 and self.id <= 47:
            self.position = "nose"
        elif self.id >= 48 and self.id <= 67:
            self.position = "mouth"
        elif self.id >= 68 and self.id <= 86:
            self.position = "face contour"
        elif self.id == 87:
            self.position = "left iris"
        elif self.id == 88:
            self.position = "right iris"
        elif self.id == 89:
            self.position = "nose tip"
        elif self.id >= 90 and self.id <= 94:
            self.position = "line above left eyebrow"
        elif self.id >= 95 and self.id <= 99:
            self.position = "line above right eyebrow"

    def __repr__(self):
        return "({1}, {2}, {3})".format(self.x, self.y, self.z)


def build_data_set(path):

    datapoints_file = path + '_datapoints.txt'
    targets_file    = path + '_targets.txt'
    frames          = []

    # Read the target file to load per-frame binary class
    with open(targets_file) as f:
        binary_classes = f.readlines()

    binary_classes = [cl.strip() for cl in binary_classes]

    with open(datapoints_file) as f:
        # Bypass the headers
        for _ in range(1):
            next(f)

        for line_number, line in enumerate(f):
            tokens    = line.strip().split(' ')
            timestamp = tokens[0]
            tokens    = tokens[1:]  # Remove the timestamp from the coordinates list

            frame = Frame(timestamp, int(binary_classes[line_number]))

            point_id = 0
            counter  = 0
            coords   = {
                'x': None,
                'y': None,
                'z': None
            }

            for token in tokens:
                if counter == 0:
                    coords['x'] = token
                    counter    += 1

                elif counter == 1:
                    coords['y'] = token
                    counter    += 1

                elif counter == 2:
                    coords['z'] = token
                    point = Point(point_id, float(coords['x']), float(coords['y']), float(coords['z']))
                    frame.points.append(point)

                    counter     = 0
                    point_id   += 1

            frames.append(frame)

    return frames
