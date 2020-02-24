""" This file demonstrates how to write classes in python.
    This Time object is inspired by Allen B. Downey and Think Python.
"""
__version__ = '0.3'
__author__ = 'Cesar Cesarotti'

class Point(object):
    """ Represents a point in 2-D space. """

def point_demo():
    """ Demonstrates bits about classes using Point class """
    dot = Point()
    dot.xval = 4
    dot.yval = -5
    print("What happens when you print p?", dot)
    print("What type is p?", type(dot))
    print("Does p have an 'x' attribute?", hasattr(dot, 'xval'))
    print("Does p have a 'z' attribute?", hasattr(dot, 'zval'))
    print("What are all of p's attributes?", dot.__dict__)


class Time(object):
    """ Represents the time of day.

        attributes: hour, minute, second

        NOTES:
        Class variables (called attributes in python) are not declared as
            part of the class definition,and new attributes can be set on
            an object at any time.

        Nothing is private in python classes!

        All method definitions have 'self' as their first input parameter,
        but you don't include it when you call the method. It is much like
        the 'this' keyword in Java.
    """

    # __init__ is the name of the Constructor method in Python.
    def __init__(self, hour=0, minute=0, second=0):
        self.hour = hour
        self.minute = minute
        self.second = second

    # __str__ is the name of the toString method in Python.
    def __str__(self):
        return '%.2d:%.2d:%.2d' % (self.hour, self.minute, self.second)

    def increment(self, seconds):
        """ Modifies this Time object to be the sum of this time and seconds."""
        self.second = self.second + seconds
        self.minute = self.minute + self.second // 60
        self.hour = self.hour + self.minute // 60
        self.second = self.second % 60
        self.minute = self.minute % 60

    # Because python allows public access to the internal data of objects,
    #   it can be valuable to create a method that implements and validates
    #   the data rules we would normally include in the setters of a class.
    #   Typically you would also include this kind of check in the __init__
    #   method.
    def is_valid(self):
        """ Checks whether a Time object satisfies the data rules(invariants)."""
        if self.hour < 0 or self.minute < 0 or self.second < 0:
            return False
        if self.minute >= 60 or self.second >= 60:
            return False
        return True

    def time_to_int(self):
        """ Computes the number of seconds since midnight."""
        minutes = self.hour * 60 + self.minute
        seconds = minutes * 60 + self.second
        return seconds

    # You can use assert statements to validate that the objects you are about
    #   to work with are following the data rules for those objects.
    def add_time(self, other):
        """ Modifies this Time object to be the sum of this time and other time."""
        assert self.is_valid() and other.is_valid()
        seconds = self.time_to_int() + other.time_to_int()
        self.hour = seconds // 3600
        self.minute = (seconds // 60) % 60
        self.second = seconds % 60

    # Python supports Operator Overloading (much like C++), which allows us
    #   to define what happens when you use various operators like '+' and '>'
    #   with this object. For a full list of operators you can overload and
    #   special method names, look here:
    #       https://docs.python.org/3/reference/datamodel.html#specialnames
    def __add__(self, other):
        """ Returns a new Time object that is the sum of two Time objects
            or a Time object and a number.

            other: Time object or number of seconds
        """
        new_time = Time(self.hour, self.minute, self.second)
        if isinstance(other, Time):
            #return Time(self.hour, self.minute, self.second).add_time(other)
            new_time.add_time(other)
        else:
            new_time.increment(other)
        return new_time

    def __radd__(self, other):
        """Adds two Time objects or a Time object and a number."""
        return self.__add__(other)


def time_demo():
    """ Demonstrates other bits about classes using Time class """
    start = Time(9, 45, 00)
    print(start)

    start.increment(1337)
    print(start)

    start = Time(9, 45)
    duration = Time(1, 35)
    print(start + duration)
    print(start + 1337)
    print(1337 + start)


if __name__ == '__main__':
    point_demo()
    #time_demo()
