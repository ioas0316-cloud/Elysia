```python
"""
A simple module demonstrating code refactoring for readability and PEP8 compliance.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import random
import sys


# ==============================================================================
# FUNCTIONS
# ==============================================================================
def do_something(x, y):
    """Calculates the sum of two numbers and prints a size description.

    Args:
        x (int | float): The first number.
        y (int | float): The second number.

    Returns:
        int | float: The sum of x and y.
    """
    # Calculate the sum of the two input numbers.
    result = x + y

    # Check if the result is greater than 10 and print a description.
    if result > 10:
        print("big")
    else:
        print("small")

    return result


# ==============================================================================
# CLASSES
# ==============================================================================
class CleanClass:
    """A simple class to demonstrate refactoring.

    This class serves as a basic container for a list of data items.

    Attributes:
        data (list): A list to store items.
    """

    def __init__(self):
        """Initializes a new CleanClass instance."""
        self.data = []

    def add(self, item):
        """Adds an item to the instance's data list.

        Args:
            item: The item to be added to the data list.
        """
        self.data.append(item)

```