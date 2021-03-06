""" This module is python practice exercises for programmers new to Python.
    All exercises can be found at: http://www.practicepython.org/exercises/
    You may submit them there to get feedback on your solutions.
    Put the code for your solutions to each exercise in the appropriate function.
    DON'T change the names of the functions!
    You may change the names of the input parameters.
    Put your code that tests your functions in the if __name__ == "__main__": section
    Don't forget to regularly commit and push to github.
    Please include an __author__ comment.
"""

import json
import requests
from bs4 import BeautifulSoup

# 5: List Overlap
def list_overlap(list1, list2):
    # olap = []

    # for x in list1:
    #     if x in list2:
    #         olap.append(x)

    olap = list(set(list1) & set(list2))
    return olap


# 7: List Comprehensions
def even_list_elements(input_list):

    # for x in input_list:
    #     if x%2 == 0:
    #         even.append(x)
    even = [x for x in input_list if x % 2 == 0]
    return even


# 10: List Overlap Comprehensions
def list_overlap_comp(list1, list2):

    return [x for x in list1 if x in list2]


# 14: List Remove Duplicates
# Use a SET for this exercise!
def remove_dups(input_list):

    return list(set(input_list))

# 15: Reverse Word Order
def reverse_words(sentence):

    splitsent = sentence.split()
    splitsent.reverse()
    return " ".join(splitsent)

# 21: Write To A File 
# Name of the file should be exer21.txt
def write_file():
 
    base_url = 'http://www.nytimes.com'
    r = requests.get(base_url)
    soup = BeautifulSoup(r.text, "lxml")
     
    with open("exer21.txt", "w") as open_file:
        for story_heading in soup.find_all(class_="story-heading"): 
            if story_heading.a: 
                open_file.write(story_heading.a.text.replace("\n", " ").strip())
            else: 
                open_file.write(story_heading.contents[0].strip())



# 22: Read From File 
def count_names_in_file():


    namedict = {}
    with open('listnames.txt', 'r') as open_file:
        for line in open_file.readlines():
            line = line.strip()
            namedict[line] = namedict.get(line, 0) + 1
    return namedict
 

# 33: Birthday Dictionaries
# Your function should know at least three birthdays
def birthdays(): 

    birthdict = {'Michael Scott' : 'March 15, 1964', 'Jim Halpert' : 'October 1, 1978', 'Dwight Schrute' : 'January 20, 1977'}
    print("Welcome to the birthday dictionary! We know the birthdays of:")
    print(birthdict.keys())

    person = input("Who's birthday would you like to look up?: ")

    print(person + "'s birthday is " + birthdict[person])


# 34: Birthday Json
def birthdays_json():

    birthdict = {'Michael Scott' : 'March 15, 1964', 'Jim Halpert' : 'October 1, 1978', 'Dwight Schrute' : 'January 20, 1977'}

    with open("birth.json", "w") as open_file:
        json.dump(birthdict, open_file)

    with open("birth.json", "r") as open_file:
        info = json.load(open_file)
        person = input("Who's birthday would you like to look up?: ")
        print(person + "'s birthday is " + info[person])


        


# # 35: Birthday Months
# def birthdays_months():


# Extra Credit! - Class Exercises
# found at https://www.hackerrank.com/domains/python/py-classes

# Class 2 - Find the Torsional Angle
# class Points(object):
#     def __init__(self, x, y, z):

#     def __sub__(self, no):

#     def dot(self, no):

#     def cross(self, no):
        
#     def absolute(self):
#         return pow((self.x ** 2 + self.y ** 2 + self.z ** 2), 0.5)

# #Classes: Dealing with Complex Numbers
# class Complex(object):
#     def __init__(self, real, imaginary):
        
#     def __add__(self, no):

#     def __sub__(self, no):
        
#     def __mul__(self, no):

#     def __div__(self, no):
        
#     def mod(self):

#     def __str__(self):
#         if self.imaginary == 0:
#             result = "%.2f+0.00i" % (self.real)
#         elif self.real == 0:
#             if self.imaginary >= 0:
#                 result = "0.00+%.2fi" % (self.imaginary)
#             else:
#                 result = "0.00-%.2fi" % (abs(self.imaginary))
#         elif self.imaginary > 0:
#             result = "%.2f+%.2fi" % (self.real, self.imaginary)
#         else:
#             result = "%.2f-%.2fi" % (self.real, abs(self.imaginary))
#         return result


if __name__ == "__main__":
    # put your test code here
    birthdays()
