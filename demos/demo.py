""" This file demonstrates many of the basic syntax of python.
    This is an example of a module docstring.
    You can create multi-line string by using triple quotes.
    A multi-line string that is not assigned to a variable
    or used in a statement is considered as a comment.
"""
__version__ = '0.4'
__author__ = 'Cesar Cesarotti'

# is the single line comment character, like // in java
print("Hello world!")  # comments work here too

def fubar(num):
    """ This function alernates printing Fu and Bar for a total of n lines.
        A multiline comment immediately after a function declaration
        is a function docstring.
    """
    # Python for statements are the same as java for each statements
    for i in range(num): # range(x) creates a list of numbers from zero to x-1
        if i % 2 == 0:
            print("Fu")
        elif i % 3 == 0:
            print("Fubar")
        else:
            print("bar")
fubar(10)

# input parameters can be given defaults:
def doubler(num=1):
    """ This functin doubles its input """
    return 2*num
print("Doubling 5 =", doubler(5))
print("Doubling the default =", doubler())

def misc_demo():
    """ This function demonstrates user input, exceptions, and misc operators."""
    def mystery(num=1):
        """ Demonstrates various other operators """
        expo = -1
        while num > 0:
            num = num // 2  # // is floor or integer division
            expo += 1
        return 2**expo  # ** is exponetiation

    # You can data from the user as a str using input:
    unum = input("Pick a number:")
    # Don't forget to check your user input!
    try:
        fnum = float(unum)
        print(mystery(fnum))
    except ValueError as verror:
        print(verror)

def list_demo():
    """This function demonstrates various features of lists"""
    squares = [1, 4, 9, 16, 25, 36, 49]
    print(squares)
    print("squares[2]=", squares[2])
    print("squares[-1]=", squares[-1])
    print("squares[-3]=", squares[-3])
    print("squares[2:4]=", squares[2:4])
    print("squares[3:]", squares[3:])
    print("len(squares)=", len(squares))
    squares.append(64)
    print("squares.append(64)=", squares)
    print("len(squares)=", len(squares))
    print("sum(squares)=", sum(squares)) # min() and max() are also built-ins
    print("squares.index(25)=", squares.index(25))
    print("squares.count(25)=", squares.count(25))
    squares.append("carla")
    print("squares.append(carla)=", squares)
    print("sum(squares)=", sum(squares)) # min() and max() are also built-ins


def string_demo():
    """This function demonstrates various features of strings"""
    squares = "language"
    print("squares now =", squares)
    print("squares[2]=", squares[2])
    print("However strings are immutable, so squares[2]='g' doesn't work")
    #squares[2] = 'g'
    print("squares[-1]=", squares[-1])
    print("squares[-3]=", squares[-3])
    print("squares[2:4]=", squares[2:4])
    print("squares[3:]", squares[3:])
    print("len(squares)=", len(squares))
    print("looping on each letter in a string is easy!")
    for letter in squares:
        print(letter)
    squares += "s"
    print("You can concatenate two strings with the + sign, so squares + 's'=", squares)
    print("len(squares)=", len(squares))
    print("methods are invoked the sameway in java, with object.method :")
    print("squares.find('a')=", squares.find('a'))
    print("'a' in squares =", 'a' in squares)
    print("squares.find('a',3)=", squares.find('a', 3))
    print("squares.count('a')=", squares.count('a'))
    print("squares.find('guag')=", squares.find('guag'))
    print("== works on strings!, so 'languages' == squares =", squares == 'languages')
    print("you can multiply strings too! squares*3=", squares*3)

if __name__ == '__main__':
    #misc_demo()
    #list_demo()
    #string_demo()
    print()
